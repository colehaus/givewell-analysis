#!/usr/bin/env python3

import collections
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import numpy as np
import pandas
import scipy
import inspect
import seaborn as sns

LogNormal = collections.namedtuple("LogNormal", "mu sd")


def present_value_of_annuity(rate, num_periods, payment_amount):
    return payment_amount * (1 - (1 + rate) ** -num_periods) / rate


def log_normal_params_from_90_percent_ci(lo, hi):
    mu = 1 / 2 * (math.log(hi) - math.log(1 / lo))
    sd = (math.log(hi) + math.log(1 / lo)) / (
        2 * math.sqrt(2) * scipy.special.erfcinv(1 / 10)
    )
    return LogNormal(mu, sd)


def parameters_to_distribution(model, parameters):
    def parameter_name_to_constructor(p):
        return {"LogNormal": pm.Lognormal}[type(p).__name__]

    def inner(k, v):
        try:
            return model.__getitem__(k)
        except KeyError:
            return parameter_name_to_constructor(v)(k, **v._asdict())

    with model:
        return {k: inner(k, v) for k, v in parameters.items()}


def parameters_to_sensitivity_analysis_problem(parameters):
    def to_sa_string(parameters):
        return {"LogNormal": "lognorm"}[type(parameters).__name__]

    return {
        "num_vars": len(parameters),
        "names": list(parameters.keys()),
        "bounds": list(map(list, parameters.values())),
        "dists": list(map(to_sa_string, parameters.values())),
    }


def run_sensitivity_analysis(parameters, model_calculation, num_samples):
    # Some calculations return TF values even when given ordinary numbers because of the use of pymc3.math
    def try_eval(x):
        try:
            return x.eval()
        except AttributeError:
            return x

    problem = parameters_to_sa_problem(parameters)
    param_values = saltelli.sample(problem, num_samples, calc_second_order=False)
    Y = np.array(
        [
            try_eval(model_calculation(**dict(zip(parameters.keys(), vs))))
            for vs in param_values
        ]
    )
    return sobol.analyze(problem, Y, calc_second_order=False)


Bounds = collections.namedtuple("Bounds", "lo hi")


def create_bounds(point, percent):
    return Bounds(lo=point * (1 - percent), hi=point * (1 + percent))


def register_rvs(model, parameters, calculation_name, model_calculation):
    rvs = parameters_to_distribution(model, parameters)
    with model:
        return pm.Deterministic(calculation_name, model_calculation(**rvs))


def staff_aggregates_to_params(nums, percent):
    return {
        k: log_normal_params_from_90_percent_ci(**create_bounds(v, percent)._asdict())
        for k, v in nums.items()
    }


def all_staff_aggregates_to_params(nums, percent):
    return {k: staff_aggregates_to_params(v, percent) for k, v in nums.items()}


def call_with_only_required_arguments(fn, d):
    required_args = set(inspect.getargspec(fn).args)
    args = {k: v for k, v in d.items() if k in required_args}
    return fn(**args)


def sensitivity_analysis_to_dataframe(sa, parameters):
    S1 = {"variable": [], "sensitivity": [], "tag": []}

    keys = list(parameters.keys())
    for v, c, k in zip(sa["S1"], sa["S1_conf"], keys):
        S1["variable"].append(k)
        S1["sensitivity"].append(v)
        S1["tag"].append("S1")

        S1["variable"].append(k)
        S1["sensitivity"].append(v - c)
        S1["tag"].append("S1l")

        S1["variable"].append(k)
        S1["sensitivity"].append(v + c)
        S1["tag"].append("S1h")

    ST = {"variable": [], "sensitivity": [], "tag": []}
    for v, c, k in zip(sa["ST"], sa["ST_conf"], keys):
        ST["variable"].append(k)
        ST["sensitivity"].append(v)
        ST["tag"].append("ST")

        ST["variable"].append(k)
        ST["sensitivity"].append(v - c)
        ST["tag"].append("STl")

        ST["variable"].append(k)
        ST["sensitivity"].append(v + c)
        ST["tag"].append("STh")

    return pandas.DataFrame(S1), pandas.DataFrame(ST)


def plot_sensitivity_analysis(sa, parameters):
    S1, ST = sa_to_dataframe(sa, parameters)
    plt.figure()
    sns.pointplot(x="sensitivity", y="variable", data=S1, join=False)
    plt.figure()
    sns.pointplot(x="sensitivity", y="variable", data=ST, join=False)
