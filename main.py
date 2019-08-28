#!/usr/bin/env python3

from functools import partial
from itertools import product
from math import ceil, floor, pi
from textwrap import wrap

from SALib.analyze import delta
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import pymc3 as pm
import seaborn as sns

from distance import (
    angle_between,
    kendall_tau,
    ranked_list,
    spearman_footrule,
    value_per_dollar,
)
from models import models
from utility import (
    call_with_only_required_args,
    filter_for_required_args,
    flatten_lists,
    grid_dims,
    keys_sorted_by_value,
    lookup_by_path,
    merge_dicts,
    bounds_to_log_normal_params,
    params_to_distribution,
    partition,
    sanitize_keys,
    sanitize_label,
    transpose,
    trim_prefix,
    values_sorted_by_key,
)
from tree import (
    ChartSpec,
    Model,
    all_params_from_models,
    apply_if_dict,
    apply_if_list,
    apply_if_model,
    big_step_chart_spec,
    cata,
    is_model,
    map_tree,
    map_tree_with_context,
    pluck_out,
    small_step_chart_specs,
    results_from_models,
)

# Munging


def inline_params(model, params):
    return map_tree(partial(apply_if_list, partial(lookup_by_path, params)), model)


def prune_inputs_in_model(model):
    def inner(fn, args):
        models, params = partition(is_model, args)
        return Model(
            fn,
            list(models)
            + [filter_for_required_args(fn, p, sanitize_label) for p in params],
        )

    return cata(partial(apply_if_model, inner), model)


# pymc3 setup


def register_model(model_context, model):

    model_with_registered_params = map_tree(
        partial(apply_if_dict, partial(params_to_distribution, model_context)), model
    )
    return map_tree_with_context(
        lambda fn, args: register_outputs(model_context, fn, args)
        if args is not None
        else fn,
        model_with_registered_params,
    )


def register_outputs(model_context, fn, args_list):
    def inner(k, v):
        try:
            return model_context.__getitem__(k)
        except KeyError:
            return pm.Deterministic(k, v)

    # For different deworming charities, we call the same code with different parameters. We make the parameters unique in the pymc3 computation graph with a per-charity prefix which we strip when actually calling the python code.
    args = sanitize_keys(merge_dicts(map(pluck_out, args_list)))
    with model_context:
        return {
            k: inner(k, v) for k, v in call_with_only_required_args(fn, args).items()
        }


# Distance

# We compute these from the trace instead of registering them in the model because pmyc3/theano doesn't like some of the operations we have to perform during the computation.
def compute_distances(models, trace):
    result_traces = {k: trace[v] for k, v in results_from_models(models).items()}
    runs = np.transpose(values_sorted_by_key(result_traces))

    reference_vector = values_sorted_by_key(value_per_dollar)
    angles = [angle_between(reference_vector, run) for run in runs]

    vector_keys = sorted(value_per_dollar.keys())
    run_rankings = [keys_sorted_by_value(dict(zip(vector_keys, run))) for run in runs]
    taus = [kendall_tau(ranked_list, ranking) for ranking in run_rankings]
    footrules = [spearman_footrule(ranked_list, ranking) for ranking in run_rankings]

    return np.array(angles), np.array(taus), np.array(footrules)


# Uncertainty


def plot_uncertainties_small_multiples(trace, results):
    rows, cols = grid_dims(len(results))

    fig = plt.figure(tight_layout=True, figsize=(3 * cols, 3 * (1 + rows)))

    # st = fig.suptitle(
    #     "Uncertainty for key outputs in Givewell's cost-effectiveness estimates"
    # )

    gs = fig.add_gridspec(ncols=cols, nrows=rows + 1)
    distances = fig.add_subplot(gs[0, :])
    result_plots = [
        (fig.add_subplot(gs[1 + floor(i / cols), i % cols]), v)
        for i, v in enumerate(results.items())
    ]

    sns.kdeplot(
        trace.angle,
        ax=distances,
        label="Angle (in radians)",
        gridsize=500,
        clip=(0, pi),
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


def plot_uncertainties_overlaid(trace, results):
    for charity, var in results.items():
        mean = np.mean(trace[var])
        normed = trace[var] / mean
        ax = sns.kdeplot(normed, label=charity, gridsize=500)
        ax.set_xlim(-0, 2)


# Regressions


def plot_regressions(ordering, charity, df, spec, should_trim_prefix=True):
    combos = list(product(spec.ins, spec.outs))
    # The `ordering` dict only includes input variables (rather than intermediate calculations) so we don't worry about sorting anything else
    combos_sorted = (
        sorted(combos, key=lambda x: ordering[x[0]])
        if set(spec.ins).issubset(set(ordering.keys()))
        else combos
    )
    rows, cols = grid_dims(len(combos_sorted))

    print(charity)
    fig = plt.figure(tight_layout=True, figsize=(3 * cols, 3 * rows))
    # Title with `tight_layout` requires `rect`. `tight_layout` with many rows (?) crashes. So we omit the title for now.
    # st = fig.suptitle("Visual sensitivity analysis of results for " + charity)

    gs = fig.add_gridspec(ncols=cols, nrows=rows)

    plots = [
        (fig.add_subplot(gs[floor(i / cols), i % cols]), v)
        for i, v in enumerate(combos_sorted)
    ]

    def trim(s):
        if should_trim_prefix:
            return trim_prefix(s)
        else:
            return s

    for (ax, (i, o)) in plots:
        sns.regplot(x=i, y=o, data=df, ax=ax)
        ax.set_xlim(min(df[i]) * 0.9, max(df[i]) * 1.1)
        ax.set_ylim(min(df[o]) * 0.9, max(df[o]) * 1.1)
        ax.set_xlabel("\n".join(wrap(trim(i), 20)))
        ax.set_ylabel("\n".join(wrap(trim(o), 20)))

    # fig.tight_layout()


def plot_big_step_regressions(ordering, models_with_variables, df):
    for charity, model in models_with_variables.items():
        plot_regressions(ordering, charity, df, big_step_chart_spec(model))


def plot_small_step_regressions(ordering, models_with_variables, df):
    for charity, model in models_with_variables.items():
        for spec in small_step_chart_specs(model):
            # This is a hacky way to remove the second unwanted (for graphing) output from one of the calculations
            outs = filter(
                lambda x: x
                != "SMC: unadjusted deaths averted per 1000 under 5s targeted",
                spec.outs,
            )
            plot_regressions(ordering, charity, df, ChartSpec(ins=spec.ins, outs=outs))


def plot_angle_regressions(ordering, models_with_variables, df):
    params = all_params_from_models(models_with_variables)
    plot_regressions(
        ordering,
        "overall ranking",
        df,
        ChartSpec(ins=params, outs=["angle"]),
        should_trim_prefix=False,
    )


# Sensitivity analysis


def calculate_sensitivities(trace, spec):
    ins_list = list(spec.ins)
    ins = np.stack([trace[i] for i in ins_list], axis=1)
    problem = {"num_vars": len(spec.ins), "names": ins_list}
    return [
        sort_sensitivities(delta.analyze(problem, ins, trace[out])) for out in spec.outs
    ]


def sort_sensitivities(a):
    transposed = transpose(a.values())
    original = transpose(sorted(transposed, reverse=True))
    return dict(zip(a.keys(), original))


def calculate_small_step_sensitivities(trace, models):
    def analyses(v):
        return flatten_lists(
            [calculate_sensitivities(trace, spec) for spec in small_step_chart_specs(v)]
        )

    return {k: analyses(v) for k, v in models.items()}


def calculate_big_step_sensitivities(trace, models):
    return {
        k: calculate_sensitivities(trace, big_step_chart_spec(v))[0]
        for k, v in models.items()
    }


def calculate_angle_sensitivities(trace, models_with_variables):
    params = all_params_from_models(models_with_variables)
    return sort_sensitivities(
        calculate_sensitivities(trace, ChartSpec(ins=params, outs=["angle"]))[0]
    )


def sensitivities_to_dataframe(sa, should_trim_prefix=True):
    def trim(s):
        if should_trim_prefix:
            return trim_prefix(s)
        else:
            return s

    def inner(d, typ, k, v, t):
        d["variable"].append(k)
        d[typ].append(v)
        d["tag"].append(t)

    S1 = {"variable": [], "S1 sensitivity": [], "tag": []}
    for v, c, k in zip(sa["S1"], sa["S1_conf"], map(trim, sa["names"])):
        inner(S1, "S1 sensitivity", k, v, "S1")
        inner(S1, "S1 sensitivity", k, v - c / 2, "S1_l")
        inner(S1, "S1 sensitivity", k, v + c / 2, "S1_h")

    delta = {"variable": [], "delta sensitivity": [], "tag": []}
    for v, c, k in zip(sa["delta"], sa["delta_conf"], map(trim, sa["names"])):
        inner(delta, "delta sensitivity", k, v, "delta")
        inner(delta, "delta sensitivity", k, v - c / 2, "delta_l")
        inner(delta, "delta sensitivity", k, v + c / 2, "delta_h")

    return DataFrame(S1), DataFrame(delta)


def plot_sensitivities(sa, should_trim_prefix=True):
    S1, delta = sensitivities_to_dataframe(sa, should_trim_prefix)
    plt.figure(figsize=(6, ceil(len(sa["names"]) / 3)))
    sns.pointplot(x="delta sensitivity", y="variable", data=delta, join=False)
    plt.figure(figsize=(6, ceil(len(sa["names"]) / 3)))
    sns.pointplot(x="S1 sensitivity", y="variable", data=S1, join=False)


# Main


def main(params, num_samples):
    sns.set(style="darkgrid")

    model_context = pm.Model()
    models_with_params = {
        k: prune_inputs_in_model(inline_params(v, bounds_to_log_normal_params(params)))
        for k, v in models.items()
    }
    models_with_variables = {
        k: register_model(model_context, v) for k, v in models_with_params.items()
    }
    with model_context:
        trace = pm.sample(num_samples)

    angles, taus, footrules = compute_distances(models_with_variables, trace)
    trace.add_values({"angle": angles, "tau": taus, "footrule": footrules})

    df = pm.trace_to_dataframe(trace)
    # Have to add these manually because pymc3 doesn't seem to do so in the line above
    df["angle"] = angles
    df["tau"] = taus
    df["footrule"] = footrules

    results = results_from_models(models_with_variables)

    plt.figure()
    plot_uncertainties_small_multiples(trace, results)
    plt.figure()
    plot_uncertainties_overlaid(trace, results)

    small_step_sensitivities = calculate_small_step_sensitivities(
        trace, models_with_variables
    )
    big_step_sensitivities = calculate_big_step_sensitivities(
        trace, models_with_variables
    )
    angle_sensitivities = calculate_angle_sensitivities(trace, models_with_variables)
    ordering = dict(
        map(lambda x: (x[1], x[0]), enumerate(angle_sensitivities["names"]))
    )

    plot_small_step_regressions(ordering, models_with_variables, df)
    plot_big_step_regressions(ordering, models_with_variables, df)
    plot_angle_regressions(ordering, models_with_variables, df)

    plot_sensitivities(angle_sensitivities, should_trim_prefix=False)
    for sensitivities in list(big_step_sensitivities.values()) + flatten_lists(
        list(small_step_sensitivities.values())
    ):
        plot_sensitivities(sensitivities)
