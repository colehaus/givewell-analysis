#!/usr/bin/env python3

from collections import namedtuple
from functools import partial, reduce

from utility import extract_only_key, flatten_lists, merge_dicts


Model = namedtuple("Model", "calculation parameters")


# Tree recursion
# I don't necessarily love how this all worked out. It's not nearly as nice without types/not in Haskell.


def cata(alg, tree):
    """Catamorphisms generalize the idea of `reduce`/`fold`. In this case, `alg` is applied from the bottom of the tree and works its way up. Can be used to reduce a tree to a single summary value."""
    if is_model(tree):
        (fn, args) = tree
        return alg(Model(fn, [cata(alg, arg) for arg in args]))
    else:
        return alg(tree)


def map_tree_with_context(f, tree):
    """This uses the `cata` machinery to define mapping over a tree. The context for a model is its arguments."""

    def wrapper(piece):
        if is_model(piece):
            (fn, args) = piece
            return Model(f(fn, args), args)
        else:
            return f(piece, None)

    return cata(wrapper, tree)


def map_tree(f, tree):
    """This uses the `cata` machinery to define mapping over a tree."""

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
    """Returns flat dictionary of all parameters used in model."""
    return cata(
        partial(apply_if_model, lambda fn, args: merge_dicts(args, no_clobber=False)),
        model,
    )


def all_params_from_models(models):
    """For each model, returns flat dictionary of all parameters used in model."""
    return reduce(
        lambda acc, x: merge_dicts([all_params_from_model(x), acc], no_clobber=False),
        models.values(),
        dict(),
    )


def results_from_models(models):
    return {k: extract_only_key(model.calculation) for k, model in models.items()}


def small_step_chart_specs(model):
    """Returns specification of inputs and outputs for a "small step" in a model. Each function in a model represents a small step. The goal here was to allow more granular analysis when desired."""
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
    """Returns specification of inputs and outputs for a big step in a model. A big step collapses all the small steps and turns the full set of input parameters into the final output value(s)."""
    return ChartSpec(all_params_from_model(model), set(model.calculation))
