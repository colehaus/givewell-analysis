#!/usr/bin/env python3

from collections import namedtuple
from functools import partial, reduce

from utility import extract_only_key, flatten_lists, merge_dicts


Model = namedtuple("Model", "calculation parameters")


# Tree recursion


def cata(alg, tree):
    if is_model(tree):
        (fn, args) = tree
        return alg(Model(fn, [cata(alg, arg) for arg in args]))
    else:
        return alg(tree)


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


# Helpers


def apply_if_model(fn, piece):
    if is_model(piece):
        (l, r) = piece
        return fn(l, r)
    else:
        return piece


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


def is_model(t):
    return type(t).__name__ == "Model"


def pluck_out(piece):
    if is_model(piece):
        return piece.calculation
    else:
        return piece


# Extractors

ChartSpec = namedtuple("ChartSpec", "ins outs")


def all_params_from_model(model):
    return cata(
        partial(apply_if_model, lambda fn, args: merge_dicts(args, no_clobber=False)),
        model,
    )


def all_params_from_models(models):
    return reduce(
        lambda acc, x: merge_dicts([all_params_from_model(x), acc], no_clobber=False),
        models.values(),
        dict(),
    )


def results_from_models(models):
    return {k: extract_only_key(model.calculation) for k, model in models.items()}


def small_step_chart_specs(model):
    specs = []

    def inner(out, ins):
        if ins is not None:
            spec = ChartSpec(set(flatten_lists(map(pluck_out, ins))), set(out))
            if spec not in specs:
                specs.append(spec)
        return out

    map_tree_with_context(inner, model)
    return specs


def big_step_chart_spec(model):
    return ChartSpec(all_params_from_model(model), set(model.calculation))
