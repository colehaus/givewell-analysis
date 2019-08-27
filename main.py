#!/usr/bin/env python3

import collections
import moral_weights
import cash
import smc
import utility
import worms
import pymc3 as pm
import seaborn as sns
import functools
from functools import partial
import distance
import numpy as np
from SALib.sample import fast_sampler
from SALib.analyze import fast
from matplotlib import pyplot as plt
import math
import itertools
from textwrap import wrap

sns.set(style="darkgrid")

givewell = {
    "Moral weights": moral_weights.givewell,
    "GiveDirectly": cash.cash_transfers_givewell,
    "Deworming": {
        "Income effects": worms.long_term_income_effects_givewell,
        "END": worms.end_givewell,
        "DTW": worms.dtw_givewell,
        "SCI": worms.sci_givewell,
        "Sightsavers": worms.sightsavers_givewell,
    },
    "Malaria Consortium": {
        "Shared": smc.smc_givewell,
        "Adherence adjustment": smc.adherence_adjustment_givewell,
        "Coverage adjustment": smc.coverage_adjustment_givewell,
        "Effectiveness": smc.smc_effectiveness_givewell,
        "Treated population": smc.treated_population_givewell,
        "Untreated population": smc.untreated_population_givewell,
        "Income increase": smc.income_increase_ages_14_and_under_givewell,
        "Results": smc.results_givewell,
    },
}

Model = collections.namedtuple("Model", "calculation parameters")

smc_effectiveness = Model(
    smc.smc_effectiveness,
    [
        ["Malaria Consortium", "Effectiveness"],
        Model(smc.coverage_adjustment, [["Malaria Consortium", "Coverage adjustment"]]),
        Model(
            smc.adherence_adjustment, [["Malaria Consortium", "Adherence adjustment"]]
        ),
    ],
)

smc_treated_population = Model(
    smc.mortality_reduction_in_treated_population,
    [
        ["Moral weights"],
        ["Malaria Consortium", "Shared"],
        ["Malaria Consortium", "Treated population"],
        smc_effectiveness,
        Model(smc.costs, [["Malaria Consortium", "Shared"]]),
    ],
)

worms_long_term_income_effects = Model(
    worms.long_term_income_effects, [["Moral weights"], ["Deworming", "Income effects"]]
)

models = {
    "GiveDirectly": Model(cash.cash_transfers, [["Moral weights"], ["GiveDirectly"]]),
    "END": Model(
        partial(worms.charity_specific, "END"),
        [["Moral weights"], ["Deworming", "END"], worms_long_term_income_effects],
    ),
    "DTW": Model(
        partial(worms.charity_specific, "DTW"),
        [["Moral weights"], ["Deworming", "DTW"], worms_long_term_income_effects],
    ),
    "SCI": Model(
        partial(worms.charity_specific, "SCI"),
        [["Moral weights"], ["Deworming", "SCI"], worms_long_term_income_effects],
    ),
    "Sightsavers": Model(
        partial(worms.charity_specific, "Sightsavers"),
        [
            ["Moral weights"],
            ["Deworming", "Sightsavers"],
            worms_long_term_income_effects,
        ],
    ),
    "Malaria Consortium": Model(
        smc.results,
        [
            ["Malaria Consortium", "Results"],
            Model(
                smc.income_increases_age_14_and_under,
                [
                    ["Moral weights"],
                    ["Malaria Consortium", "Shared"],
                    ["Malaria Consortium", "Income increase"],
                    smc_effectiveness,
                ],
            ),
            smc_treated_population,
            Model(
                smc.mortality_reduction_in_untreated_population,
                [
                    ["Moral weights"],
                    ["Malaria Consortium", "Shared"],
                    ["Malaria Consortium", "Untreated population"],
                    smc_treated_population,
                    Model(smc.costs, [["Malaria Consortium", "Shared"]]),
                ],
            ),
        ],
    ),
}


def sanitize_label(s):
    return trim_prefix(s).replace(" ", "_").replace(".", "").replace("/", "")


def trim_prefix(key):
    parts = key.partition(": ")
    if parts[1] is ": ":
        return parts[2]
    else:
        return key


# TODO: Audit duplicate variables
def register_calculations(model_context, fn, args_list):
    def inner(k, v):
        try:
            return model_context.__getitem__(k)
        except KeyError:
            return pm.Deterministic(k, v)

    # For different deworming charities, we call the same code with different parameters. We make the parameters unique in the pymc3 computation graph with a per-charity prefix which we strip when actually calling the python code.
    args = {
        sanitize_label(k): v
        for k, v in utility.merge_dicts(map(pluck_out, args_list)).items()
    }
    with model_context:
        return {
            k: inner(k, v)
            for k, v in utility.call_with_only_required_arguments(fn, args).items()
        }


def register_params(model_context, params):
    return utility.parameters_to_distribution(model_context, params)


def apply_if_model(fn, piece):
    if is_model(piece):
        (l, r) = piece
        return fn(l, r)
    else:
        return piece


def apply_if_not_model(fn, piece):
    if is_model(piece):
        return piece
    else:
        return fn(piece)


def apply_if_list(fn, piece):
    if type(piece) is list:
        return fn(piece)
    else:
        return piece


def apply_if_dict(fn, piece):
    if type(piece) is dict:
        return fn(piece)
    else:
        return piece


def lookup_by_path(d, path):
    return functools.reduce(lambda acc, key: acc[key], path, d)


def apply_fn_to_arg_list(fn, args_list):
    args = utility.merge_dicts(args_list)
    return utility.call_with_only_required_arguments(fn, args)


def map_tree_with_context(f, tree):
    def wrapper(piece):
        if is_model(piece):
            (fn, args) = piece
            return Model(f(fn, args), args)
        else:
            return f(piece, None)

    return cata(wrapper, tree)


def map_tree(f, tree):
    def wrapper(*args):
        return f(args[0])

    return map_tree_with_context(wrapper, tree)


def is_model(t):
    return type(t).__name__ == "Model"


def cata(alg, tree):
    if is_model(tree):
        (fn, args) = tree
        return alg(Model(fn, [cata(alg, arg) for arg in args]))
    else:
        return alg(tree)


def inline_params(model, params):
    return map_tree(partial(apply_if_list, partial(lookup_by_path, params)), model)


def partition(fn, xs):
    return (list(filter(fn, xs)), list(filter(lambda x: not fn(x), xs)))


def merge_dicts_in_model(model):
    def inner(fn, args):
        models, params = partition(is_model, args)
        return Model(fn, list(models) + [utility.merge_dicts(list(params))])

    return cata(partial(apply_if_model, inner), model)


def prune_args_in_model(model):
    def inner(fn, args):
        models, params = partition(is_model, args)
        return Model(
            fn,
            list(models)
            + [
                utility.filter_for_required_arguments(fn, p, sanitize_label)
                for p in params
            ],
        )

    return cata(partial(apply_if_model, inner), model)


def sanitize_keys(d):
    return {sanitize_label(k): v for k, v in d.items()}


def run_calculations(fn, args, num_samples):
    homogenized_args = list(map(pluck_out, args))
    labels = [
        sanitize_label(k)
        for k in utility.flatten_lists([k for k, v in homogenized_args])
    ]
    runs = np.concatenate([v for k, v in homogenized_args], axis=1)

    def inner(X):
        return {
            k: utility.try_eval(v)
            for k, v in utility.call_with_only_required_arguments(
                fn, dict(zip(labels, X))
            ).items()
        }

    return (
        sorted(inner(runs[0]).keys()),
        [utility.values_sorted_by_key(inner(run)) for run in runs],
    )


# def sample_model(model, num_samples):
#     def inner(params):
#         samples = fast_sampler.sample(
#             utility.parameters_to_sensitivity_analysis_problem(params),
#             num_samples,
#         )
#         ordered_keys = sorted(params.keys())
#         return ordered_keys, samples

#     model_with_sampled_params = map_tree(partial(apply_if_dict, inner), model)
#     return map_tree_with_context(
#         lambda fn, args: run_calculations(fn, args, num_samples)
#         if args is not None
#         else fn,
#         model_with_sampled_params,
#     )


def register_model(model_context, model):
    model_with_registered_params = map_tree(
        partial(apply_if_dict, partial(register_params, model_context)), model
    )
    return map_tree_with_context(
        lambda fn, args: register_calculations(model_context, fn, args)
        if args is not None
        else fn,
        model_with_registered_params,
    )


def strip_vars(model):
    return map_tree(lambda x: x.keys(), model)


def pluck_out(piece):
    if is_model(piece):
        return piece.calculation
    else:
        return piece


ChartSpec = collections.namedtuple("ChartSpec", "ins outs")


def big_step_chart_spec(model):
    return ChartSpec(all_inputs_from_model(model), set(model.calculation))


def small_step_chart_specs(model):
    specs = []

    def inner(out, ins):
        if ins is not None:
            spec = ChartSpec(set(utility.flatten_lists(map(pluck_out, ins))), set(out))
            if spec not in specs:
                specs.append(spec)
        return out

    map_tree_with_context(inner, model)
    return specs


def eval_model(model, params):
    model_with_params = map_tree(
        partial(apply_if_list, partial(lookup_by_path, params)), model
    )
    return cata(partial(apply_if_model, apply_fn_to_arg_list), model_with_params)


# We compute these from the trace instead of registering them in the model because pmyc3/theano doesn't like some of the operations we have to perform during the computation.
def compute_distances(models, trace):
    result_traces = {k: trace[v] for k, v in top_level_results(models).items()}
    runs = np.transpose(utility.values_sorted_by_key(result_traces))

    reference_vector = utility.values_sorted_by_key(distance.value_per_dollar)
    angles = [distance.angle_between(reference_vector, run) for run in runs]

    vector_keys = sorted(distance.value_per_dollar.keys())
    run_rankings = [
        utility.keys_sorted_by_value(dict(zip(vector_keys, run))) for run in runs
    ]
    taus = [
        distance.kendall_tau(distance.ranked_list, ranking) for ranking in run_rankings
    ]
    footrules = [
        distance.spearman_footrule(distance.ranked_list, ranking)
        for ranking in run_rankings
    ]

    return np.array(angles), np.array(taus), np.array(footrules)


def all_outputs_from_model(model):
    def inner(piece):
        if is_model(piece):
            fn, args = piece
            return utility.merge_dicts(args + [fn], no_clobber=False)
        else:
            return dict()

    return cata(inner, model)


def all_inputs_from_model(model):
    return cata(
        partial(
            apply_if_model, lambda fn, args: utility.merge_dicts(args, no_clobber=False)
        ),
        model,
    )


def all_inputs_from_models(models):
    return functools.reduce(
        lambda acc, x: utility.merge_dicts(
            [all_inputs_from_model(x), acc], no_clobber=False
        ),
        models.values(),
        dict(),
    )


def top_level_results(models):
    return {
        k: utility.extract_only_key(model.calculation) for k, model in models.items()
    }


def test():
    print(
        {
            model_name: {
                result_name: utility.try_eval(result)
                for result_name, result in eval_model(model, givewell).items()
            }
            for model_name, model in models.items()
        }
    )


def plot_uncertainty_small_multiples(trace, results):
    rows, cols = grid_dims(len(results))

    fig = plt.figure(tight_layout=True, figsize=(3 * cols, 3 * (1 + rows)))

    # st = fig.suptitle(
    #     "Uncertainty for key outputs in Givewell's cost-effectiveness estimates"
    # )

    gs = fig.add_gridspec(ncols=cols, nrows=rows + 1)
    distances = fig.add_subplot(gs[0, :])
    result_plots = [
        (fig.add_subplot(gs[1 + math.floor(i / cols), i % cols]), v)
        for i, v in enumerate(results.items())
    ]

    sns.kdeplot(
        trace.angle,
        ax=distances,
        label="Angle (in radians)",
        gridsize=500,
        clip=(0, math.pi),
    )
    sns.kdeplot(
        trace.tau,
        ax=distances,
        bw=0.8,
        label="Kendall's tau",
        gridsize=500,
        clip=(0, 1),
    )
    sns.kdeplot(
        trace.footrule,
        ax=distances,
        bw=0.8,
        label="Spearman's footrule",
        gridsize=500,
        clip=(0, 1),
    )
    distances.set_xlim(-0.5, 1)

    for (ax, (charity, var)) in result_plots:
        sns.kdeplot(trace[var], ax=ax, label=charity, gridsize=500)
        ax.set_xlim(0, 0.25)

    # fig.tight_layout(rect=[0, 0.01, 1, 0.97])


def plot_uncertainty_overlaid(trace, results):
    ax = None
    for charity, var in results.items():
        mean = np.mean(trace[var])
        normed = trace[var] / mean
        ax = sns.kdeplot(normed, label=charity)
        ax.set_xlim(-0, 2)
    # ax.set_title(
    #     "Normalized uncertainty for value per dollar of GiveWell top charities"
    # )


def grid_dims(n):
    cols = 4
    rows = math.ceil(n / cols)
    return rows, cols


def plot_regressions(charity, df, step):
    combos = list(itertools.product(step.ins, step.outs))
    rows, cols = grid_dims(len(combos))

    print(charity)
    fig = plt.figure(tight_layout=True, figsize=(3 * cols, 3 * rows))
    # Title with `tight_layout` requires `rect`. `tight_layout` with many rows (?) crashes. So we omit the title for now.
    # st = fig.suptitle("Visual sensitivity analysis of results for " + charity)

    gs = fig.add_gridspec(ncols=cols, nrows=rows)

    plots = [
        (fig.add_subplot(gs[math.floor(i / cols), i % cols]), v)
        for i, v in enumerate(combos)
    ]

    for (ax, (i, o)) in plots:
        sns.regplot(x=i, y=o, data=df, ax=ax)
        ax.set_xlim(min(df[i]) * 0.9, max(df[i]) * 1.1)
        ax.set_ylim(min(df[o]) * 0.9, max(df[o]) * 1.1)
        ax.set_xlabel("\n".join(wrap(trim_prefix(i), 20)))
        ax.set_ylabel("\n".join(wrap(trim_prefix(o), 20)))

    # fig.tight_layout()


def plot_all_big_step_regressions(models_with_variables, df):
    for charity, model in models_with_variables.items():
        plot_regressions(charity, df, big_step_chart_spec(model))


def plot_all_small_step_regressions(models_with_variables, df):
    for charity, model in models_with_variables.items():
        for spec in small_step_chart_specs(model):
            # This is a hacky way to remove the second unwanted (for graphing) output from one of the calculations
            outs = filter(
                lambda x: x
                != "SMC: unadjusted deaths averted per 1000 under 5s targeted",
                spec.outs,
            )
            plot_regressions(charity, df, ChartSpec(ins=spec.ins, outs=outs))


def plot_inputs_vs_angle(models_with_variables, df):
    inputs = all_inputs_from_models(models_with_variables)
    plot_regressions("overall ranking", df, ChartSpec(ins=inputs, outs=["angle"]))


def main(parameters):
    model_context = pm.Model()
    models_with_params = {
        k: prune_args_in_model(
            inline_params(v, utility.numbers_to_log_normal_params(parameters, 0.5))
        )
        for k, v in models.items()
    }
    models_with_variables = {
        k: register_model(model_context, v) for k, v in models_with_params.items()
    }
    with model_context:
        trace = pm.sample(1000)

    angles, taus, footrules = compute_distances(models_with_variables, trace)
    trace.add_values({"angle": angles, "tau": taus, "footrule": footrules})

    df = pm.trace_to_dataframe(trace)
    # Have to add these manually because pymc3 doesn't seem to do so in the line above
    df["angle"] = angles
    df["tau"] = taus
    df["footrule"] = footrules

    results = top_level_results(models_with_variables)

    plt.figure()
    plot_uncertainty_small_multiples(trace, results)
    plt.figure()
    plot_uncertainty_overlaid(trace, results)
    plot_all_small_step_regressions(models_with_variables, df)
    plot_all_big_step_regressions(models_with_variables, df)
    plot_inputs_vs_angle(models_with_variables, df)

    # for k, v in models.items():
    #     params = dict(collections.ChainMap(*[parameters[p] for p in v.parameters]))
    #     sa = utility.run_sensitivity_analysis(params, v.calculation, 1)
    #     utility.plot_sensitivity_analysis(sa, params)
    #     for i in params.keys():
    #         sns.jointplot(i, k, data=frame, kind="reg")
