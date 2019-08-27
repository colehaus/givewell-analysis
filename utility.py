#!/usr/bin/env python3

import collections
import math
import pymc3 as pm
import scipy
import inspect
import itertools

LogNormal = collections.namedtuple("LogNormal", "mu sd")


def present_value_of_annuity(rate, num_periods, payment_amount):
    return payment_amount * (1 - (1 + rate) ** -num_periods) / rate


def log_normal_params_from_90_percent_ci(lo, hi):
    mu = 1 / 2 * (math.log(hi) - math.log(1 / lo))
    sd = (math.log(hi) + math.log(1 / lo)) / (
        2 * math.sqrt(2) * scipy.special.erfcinv(1 / 10)
    )
    return LogNormal(mu, sd)


def parameters_to_distribution(model_context, parameters):
    def parameter_name_to_constructor(p):
        return {"LogNormal": pm.Lognormal}[type(p).__name__]

    def inner(k, v):
        try:
            return model_context.__getitem__(k)
        except KeyError:
            return parameter_name_to_constructor(v)(k, **v._asdict())

    with model_context:
        return {k: inner(k, v) for k, v in parameters.items()}


def parameters_to_sensitivity_analysis_problem(parameters):
    def to_sa_string(parameters):
        return {"LogNormal": "lognorm"}[type(parameters).__name__]

    items = sorted(parameters.items())

    val = {
        "num_vars": len(items),
        "names": [k for k, v in items],
        "bounds": [list(v) for k, v in items],
        "dists": [to_sa_string(v) for k, v in items],
    }
    return val


def try_eval(x):
    try:
        return x.eval()
    except AttributeError:
        return x


Bounds = collections.namedtuple("Bounds", "lo hi")


def create_bounds(point, percent):
    return Bounds(lo=point * (1 - percent), hi=point * (1 + percent))


def register_rvs(model, parameters, calculation_name, model_calculation):
    rvs = parameters_to_distribution(model, parameters)
    with model:
        return pm.Deterministic(calculation_name, model_calculation(**rvs))


def number_to_log_normal_params(num, percent):
    return log_normal_params_from_90_percent_ci(**create_bounds(num, percent)._asdict())


def numbers_to_log_normal_params(params, percent):
    return {
        k: numbers_to_log_normal_params(v, percent)
        if type(v) is dict
        else number_to_log_normal_params(v, percent)
        for k, v in params.items()
    }


def filter_for_required_arguments(fn, d, transform):
    required_args = set(inspect.getargspec(fn).args)
    return {k: v for k, v in d.items() if transform(k) in required_args}


def call_with_only_required_arguments(fn, d):
    required_args = set(inspect.getargspec(fn).args)
    args = {k: v for k, v in d.items() if k in required_args}
    return fn(**args)


def merge_dicts(dicts, no_clobber=True):
    # This is just because python does a ridiculous thing where calling `list(a)` on an iterator twice gives different results
    dicts_list = list(dicts)
    if no_clobber:
        keys = list(itertools.chain.from_iterable([d.keys() for d in dicts_list]))
        if len(keys) != len(list(set(keys))):
            raise AssertionError(
                "Duplicate keys in dictionary merge: " + str(keys.sort())
            )
    return dict(collections.ChainMap(*dicts_list))


def flatten_lists(lists):
    return list(itertools.chain.from_iterable(lists))


def extract_only_value(d):
    vals = d.values()
    if len(vals) == 1:
        return list(vals)[0]
    else:
        raise AssertionError("Not just a single value in dict")


def extract_only_key(d):
    keys = d.keys()
    if len(keys) == 1:
        return list(keys)[0]
    else:
        raise AssertionError("Not just a single value in dict")


def keys_sorted_by_value(d):
    return [k for k, v in sorted(d.items(), key=lambda x: x[1])]


def values_sorted_by_key(d):
    return [v for k, v in sorted(d.items())]


# transpose_dict
