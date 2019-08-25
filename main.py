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


def trim_prefix(key):
    parts = key.partition(": ")
    if parts[1] is ": ":
        return parts[2]
    else:
        return key


# TODO: Audit duplicate variables
def register_calculations(model_context, fn, argsList):
    def inner(k, v):
        try:
            return model_context.__getitem__(k)
        except KeyError:
            return pm.Deterministic(k, v)

    argsListMunged = [
        arg.calculation if type(arg) is Model else arg for arg in argsList
    ]
    # For different deworming charities, we call the same code with different parameters. We make the parameters unique in the pymc3 computation graph with a per-charity prefix which we strip when actually calling the python code.
    args = {trim_prefix(k).replace(" ", "_"): v for k, v in utility.merge_dicts(argsListMunged).items()}
    with model_context:
        return {
            k: inner(k, v)
            for k, v in utility.call_with_only_required_arguments(fn, args).items()
        }


def register_params(model_context, params):
    return utility.parameters_to_distribution(model_context, params)


def apply_if_model(fn, piece):
    if type(piece) is Model:
        (l, r) = piece
        return fn(l, r)
    else:
        return piece


def apply_if_not_model(fn, piece):
    if type(piece) is Model:
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


def apply_fn_to_arg_list(fn, argsList):
    args = utility.merge_dicts(argsList)
    return utility.call_with_only_required_arguments(fn, args)


def map_tree_with_context(f, tree):
    def wrapper(piece):
        if type(piece) is Model:
            (fn, args) = piece
            return Model(f(fn, args), args)
        else:
            return f(piece, None)

    return cata(wrapper, tree)


def map_tree(f, tree):
    def wrapper(*args):
        return f(args[0])

    return map_tree_with_context(wrapper, tree)


def cata(alg, tree):
    if type(tree) is Model:
        (fn, args) = tree
        return alg(Model(fn, [cata(alg, arg) for arg in args]))
    else:
        return alg(tree)


def register_model(model_context, model, params):
    model_with_params = map_tree(
        partial(apply_if_list, partial(lookup_by_path, params)), model
    )
    model_with_registered_params = map_tree(
        partial(apply_if_dict, partial(register_params, model_context)),
        model_with_params,
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
    if type(piece) is Model:
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
    runs = np.transpose([v for k, v in sorted(result_traces.items())])

    reference_vector = [v for k, v in sorted(distance.value_per_dollar.items())]
    angles = [distance.angle_between(reference_vector, run) for run in runs]

    vector_keys = [k for k, v in sorted(distance.value_per_dollar.items())]
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

    return angles, taus, footrules

def all_outputs_from_model(model):
    def inner(piece):
        if type(piece) is Model:
            fn, args = piece
            return utility.unions(args).union(set(fn))
        else:
            return set()
    return cata(inner, model)


def all_inputs_from_model(model):
    return cata(
        partial(apply_if_model, lambda fn, args: utility.unions(args)), model
    )

def all_inputs_from_models(models):
    return functools.reduce(lambda acc, x: acc.union(all_inputs_from_model(x)), models.values(), set())


def top_level_results(models):
    return {
        k: utility.extract_only_value(model.calculation) for k, model in models.items()
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


def main(parameters):
    model = pm.Model()
    for k, v in models.items():
        cata(register, tree)

    # with model:
    #     trace = pm.sample(1)
    # frame = pm.trace_to_dataframe(trace)

    # pm.plot_posterior(trace, varnames=models.keys(), kde_plot=True)

    # for k, v in models.items():
    #     params = dict(collections.ChainMap(*[parameters[p] for p in v.parameters]))
    #     sa = utility.run_sensitivity_analysis(params, v.calculation, 1)
    #     utility.plot_sensitivity_analysis(sa, params)
    #     for i in params.keys():
    #         sns.jointplot(i, k, data=frame, kind="reg")
