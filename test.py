#!/usr/bin/env python3

from functools import partial

from models import givewell, models
from tree import apply_if_list, apply_if_model, cata, map_tree
from utility import (
    call_with_only_required_args,
    merge_dicts,
    lookup_by_path,
    sanitize_keys,
    try_eval,
)


def eval_model(model, params):
    model_with_params = map_tree(
        partial(apply_if_list, partial(lookup_by_path, params)), model
    )
    return cata(
        partial(
            apply_if_model,
            lambda fn, args: call_with_only_required_args(
                fn, sanitize_keys(merge_dicts(args))
            ),
        ),
        model_with_params,
    )


def test():
    results = {
        model_name: {
            result_name: try_eval(result)
            for result_name, result in eval_model(model, givewell).items()
        }
        for model_name, model in models.items()
    }
    for k, v in results.items():
        print(k, list(v.values())[0])
