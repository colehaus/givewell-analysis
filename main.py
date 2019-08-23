#!/usr/bin/env python3

import collections
import moral_weights
import cash
import utility
import worms
import pymc3 as pm
import seaborn as sns

sns.set(style="darkgrid")

params = {
    "moral weights": moral_weights.staff_aggregates,
    "GiveDirectly": cash.staff_aggregates,
    "deworming": worms.deworming_staff_aggregates,
    "END": worms.end_staff_aggregates,
    "DTW": worms.dtw_staff_aggregates,
    "SCI": worms.sci_staff_aggregates,
    "Sightsavers": worms.sightsavers_staff_aggregates,
}

Model = collections.namedtuple("Model", "calculation parameters")

models = {
    "GiveDirectly": Model(cash.cash_transfers, ["moral weights", "GiveDirectly"]),
    "END": Model(worms.combined, ["moral weights", "deworming", "END"]),
    "DTW": Model(worms.combined, ["moral weights", "deworming", "DTW"]),
    "SCI": Model(worms.combined, ["moral weights", "deworming", "SCI"]),
    "Sightsavers": Model(worms.combined, ["moral weights", "deworming", "Sightsavers"]),
}


def main(parameters):
    model = pm.Model()
    charities = dict()
    for k, v in models.items():
        params = dict(collections.ChainMap(*[parameters[p] for p in v.parameters]))
        charities[k] = utility.register_rvs(model, params, k, v.calculation)

    with model:
        trace = pm.sample(2000)
    frame = pm.trace_to_dataframe(trace)

    pm.plot_posterior(trace, varnames=models.keys(), kde_plot=True)

    for k, v in models.items():
        params = dict(collections.ChainMap(*[parameters[p] for p in v.parameters]))
        sa = utility.run_sensitivity_analysis(params, v.calculation, 20)
        utility.plot_sensitivity_analysis(sa, params)
        for i in params.keys():
            sns.jointplot(i, k, data=frame, kind="reg")
