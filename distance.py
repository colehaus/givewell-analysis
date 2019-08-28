#!/usr/bin/env python3

import numpy as np
import itertools
import utility

ten_k = 10000
value_per_dollar = {
    "GiveDirectly": 38 / ten_k,
    "END": 222 / ten_k,
    "DTW": 738 / ten_k,
    "SCI": 378 / ten_k,
    "Sightsavers": 394 / ten_k,
    "Malaria Consortium": 326 / ten_k,
    "AMF": 247 / ten_k,
    "HKI": 223 / ten_k,
}

ranked_list = utility.keys_sorted_by_value(value_per_dollar)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))


def kendall_tau(o1, o2):
    assert set(o1) == set(o2)

    def greater_than(l, r, o):
        return o.index(l) > o.index(r)

    def concordant_or_discordant(l, r):
        if greater_than(l, r, o1) == greater_than(l, r, o2):
            return 0
        else:
            return 1

    n = len(o1)
    return sum(
        concordant_or_discordant(l, r) for l, r in itertools.combinations(o1, 2)
    ) / (n * (n - 1) / 2)


def spearman_footrule(o1, o2):
    assert set(o1) == set(o2)
    n = len(o1)
    if n % 2 == 0:
        norm = n ** 2 / 2
    else:
        norm = (n + 1) * (n - 1) / 2
    return sum(abs(o1.index(a) - o2.index(a)) for a in o1) / norm
