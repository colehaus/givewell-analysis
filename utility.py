#!/usr/bin/env python3

from collections import namedtuple
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import scipy

LogNormal = namedtuple('LogNormal', 'mu sd')

def present_value_of_annuity(rate, num_periods, payment_amount):
    return payment_amount * (1 - (1 + rate) ** -num_periods) / rate

def log_normal_params_from_90_percent_ci(lo, hi):
    mu = 1/2 * (math.log(hi) - math.log(1/lo))
    sd = (math.log(hi) + math.log(1/lo)) / (2 * math.sqrt(2) * scipy.special.erfcinv(1/10))
    return LogNormal(mu=mu, sd=sd)

def log_normalify_parameters(model, parameters):
    with model:
        return {k: pm.Lognormal(k, **v._asdict()) for k, v in parameters.items()}
