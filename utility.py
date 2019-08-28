#!/usr/bin/env python3

from collections import ChainMap, namedtuple
from functools import reduce
import inspect
from itertools import chain
from math import ceil, log, sqrt

import scipy
import pymc3 as pm


# Distributions

LogNormal = namedtuple("LogNormal", "mu sd")


def create_bounds(point, percent):
    return {"lo": point * (1 - percent), "hi": point * (1 + percent)}


def numbers_to_bounds(params, percent):
    return {
        k: numbers_to_bounds(v, percent)
        if type(v) is dict
        else create_bounds(v, percent)
        for k, v in params.items()
    }


def bounds_to_log_normal_params(params):
    return {
        k: log_normal_params_from_90_percent_ci(**v)
        if set(v.keys()) == {"lo", "hi"}
        else bounds_to_log_normal_params(v)
        for k, v in params.items()
    }


def log_normal_params_from_90_percent_ci(lo, hi):
    mu = 1 / 2 * (log(hi) - log(1 / lo))
    sd = (log(hi) + log(1 / lo)) / (2 * sqrt(2) * scipy.special.erfcinv(1 / 10))
    return LogNormal(mu, sd)


def params_to_distribution(model_context, params):
    def parameter_name_to_constructor(p):
        return {"LogNormal": pm.Lognormal}[type(p).__name__]

    def inner(k, v):
        try:
            return model_context.__getitem__(k)
        except KeyError:
            return parameter_name_to_constructor(v)(k, **v._asdict())

    with model_context:
        return {k: inner(k, v) for k, v in params.items()}


# Collections


def lookup_by_path(d, path):
    return reduce(lambda acc, key: acc[key], path, d)


def partition(fn, xs):
    return (list(filter(fn, xs)), list(filter(lambda x: not fn(x), xs)))


def transpose(lists):
    return [list(x) for x in zip(*lists)]


def keys_sorted_by_value(d):
    return [k for k, v in sorted(d.items(), key=lambda x: x[1])]


def values_sorted_by_key(d):
    return [v for k, v in sorted(d.items())]


def merge_dicts(dicts, no_clobber=True):
    # This is just because python does a ridiculous thing where calling `list(a)` on an iterator twice gives different results
    dicts_list = list(dicts)
    if no_clobber:
        keys = list(chain.from_iterable([d.keys() for d in dicts_list]))
        if len(keys) != len(list(set(keys))):
            raise AssertionError(
                "Duplicate keys in dictionary merge: " + str(keys.sort())
            )
    return dict(ChainMap(*dicts_list))


def flatten_lists(lists):
    return list(chain.from_iterable(lists))


def extract_only_key(d):
    keys = d.keys()
    if len(keys) == 1:
        return list(keys)[0]
    else:
        raise AssertionError("Not just a single key in dict")


# String manipulation


def sanitize_label(s):
    return trim_prefix(s).replace(" ", "_").replace(".", "").replace("/", "")


def trim_prefix(key):
    parts = key.partition(": ")
    if parts[1] == ": ":
        return parts[2]
    else:
        return key


def sanitize_keys(d):
    return {sanitize_label(k): v for k, v in d.items()}


# Misc


def present_value_of_annuity(rate, num_periods, payment_amount):
    return payment_amount * (1 - (1 + rate) ** -num_periods) / rate


def try_eval(x):
    try:
        return x.eval()
    except AttributeError:
        return x


def filter_for_required_args(fn, d, transform):
    required_args = set(inspect.getargspec(fn).args)
    return {k: v for k, v in d.items() if transform(k) in required_args}


def call_with_only_required_args(fn, d):
    return fn(**filter_for_required_args(fn, d, lambda x: x))


def grid_dims(n, max_cols=4):
    cols = max_cols
    rows = ceil(n / cols)
    return rows, cols
