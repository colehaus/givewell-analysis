#!/usr/bin/env python3

from functools import partial

import cash
import moral_weights
import nets
import smc
from tree import Model
import vas
import worms

smc_effectiveness = Model(
    smc.smc_effectiveness,
    [
        ["Malaria Consortium", "Effectiveness"],
        Model(smc.coverage_adjustment, [["Malaria Consortium", "Coverage adjustment"]]),
        Model(
            smc.adherence_adjustment, [["Malaria Consortium", "Adherence adjustment"]]
        ),
    ],
)

smc_treated_population = Model(
    smc.mortality_reduction_in_treated_population,
    [
        ["Moral weights"],
        ["Malaria Consortium", "Shared"],
        ["Malaria Consortium", "Treated population"],
        smc_effectiveness,
        Model(smc.costs, [["Malaria Consortium", "Shared"]]),
    ],
)

worms_long_term_income_effects = Model(
    worms.long_term_income_effects, [["Moral weights"], ["Deworming", "Income effects"]]
)

amf_cost_per_yr = Model(nets.cost_per_yr_of_protection, [["AMF", "Cost"]])

amf_pre_existing_nets = Model(nets.pre_existing_nets, [["AMF", "Pre-existing"]])

amf_under_5 = Model(
    nets.deaths_averted_children_under_5,
    [
        ["Moral weights"],
        amf_cost_per_yr,
        Model(nets.effectiveness, [["AMF", "Shared"], ["AMF", "Effectiveness"]]),
        amf_pre_existing_nets,
    ],
)

models = {
    "GiveDirectly": Model(cash.cash_transfers, [["Moral weights"], ["GiveDirectly"]]),
    "END": Model(
        partial(worms.charity_specific, "END"),
        [["Moral weights"], ["Deworming", "END"], worms_long_term_income_effects],
    ),
    "DTW": Model(
        partial(worms.charity_specific, "DTW"),
        [["Moral weights"], ["Deworming", "DTW"], worms_long_term_income_effects],
    ),
    "SCI": Model(
        partial(worms.charity_specific, "SCI"),
        [["Moral weights"], ["Deworming", "SCI"], worms_long_term_income_effects],
    ),
    "Sightsavers": Model(
        partial(worms.charity_specific, "Sightsavers"),
        [
            ["Moral weights"],
            ["Deworming", "Sightsavers"],
            worms_long_term_income_effects,
        ],
    ),
    "Malaria Consortium": Model(
        smc.results,
        [
            ["Malaria Consortium", "Results"],
            Model(
                smc.income_increases_age_14_and_under,
                [
                    ["Moral weights"],
                    ["Malaria Consortium", "Shared"],
                    ["Malaria Consortium", "Income increase"],
                    smc_effectiveness,
                ],
            ),
            smc_treated_population,
            Model(
                smc.mortality_reduction_in_untreated_population,
                [
                    ["Moral weights"],
                    ["Malaria Consortium", "Shared"],
                    ["Malaria Consortium", "Untreated population"],
                    smc_treated_population,
                    Model(smc.costs, [["Malaria Consortium", "Shared"]]),
                ],
            ),
        ],
    ),
    "HKI": Model(
        vas.results,
        [
            ["Moral weights"],
            ["HKI", "Results"],
            Model(
                vas.cost_of_covering_a_hypothetical_cohort_with_vitamin_a_supplementation,
                [["HKI", "Shared"], ["HKI", "Cost"]],
            ),
            Model(
                vas.mortality_reduction_in_hypothetical_cohort,
                [
                    ["HKI", "Mortality reduction"],
                    Model(
                        vas.meta_analysis_finding_on_relative_rate_of_mortality_reduction,
                        [["HKI", "Meta-analysis"]],
                    ),
                ],
            ),
            Model(
                vas.development_benefits,
                [
                    ["Moral weights"],
                    ["HKI", "Shared"],
                    ["HKI", "Development"],
                    worms_long_term_income_effects,
                ],
            ),
        ],
    ),
    "AMF": Model(
        nets.results,
        [
            ["AMF", "Results"],
            Model(
                nets.income_increase_under_15,
                [
                    ["Moral weights"],
                    ["AMF", "Income increase"],
                    amf_cost_per_yr,
                    Model(
                        nets.reduction_in_prevalence_from_net_distributions,
                        [
                            ["AMF", "Shared"],
                            ["AMF", "Prevalence reduction"],
                            amf_pre_existing_nets,
                        ],
                    ),
                ],
            ),
            amf_under_5,
            Model(
                nets.deaths_averted_over_5,
                [["Moral weights"], ["AMF", "Deaths over 5"], amf_under_5],
            ),
        ],
    ),
}

givewell = {
    "Moral weights": moral_weights.givewell,
    "GiveDirectly": cash.cash_transfers_givewell,
    "Deworming": {
        "Income effects": worms.long_term_income_effects_givewell,
        "END": worms.end_givewell,
        "DTW": worms.dtw_givewell,
        "SCI": worms.sci_givewell,
        "Sightsavers": worms.sightsavers_givewell,
    },
    "Malaria Consortium": {
        "Shared": smc.smc_givewell,
        "Adherence adjustment": smc.adherence_adjustment_givewell,
        "Coverage adjustment": smc.coverage_adjustment_givewell,
        "Effectiveness": smc.smc_effectiveness_givewell,
        "Treated population": smc.treated_population_givewell,
        "Untreated population": smc.untreated_population_givewell,
        "Income increase": smc.income_increase_ages_14_and_under_givewell,
        "Results": smc.results_givewell,
    },
    "HKI": {
        "Shared": vas.vas_givewell,
        "Cost": vas.cost_of_covering_a_hypothetical_cohort_with_vitamin_a_supplementation_givewell,
        "Meta-analysis": vas.meta_analysis_finding_on_relative_rate_of_mortality_reduction_givewell,
        "Mortality reduction": vas.mortality_reduction_in_hypothetical_cohort_givewell,
        "Development": vas.development_benefits_givewell,
        "Results": vas.results_givewell,
    },
    "AMF": {
        "Shared": nets.nets_givewell,
        "Effectiveness": nets.effectiveness_givewell,
        "Pre-existing": nets.pre_existing_nets_givewell,
        "Prevalence reduction": nets.reduction_in_prevalence_from_net_distributions_givewell,
        "Cost": nets.cost_per_yr_of_protection_givewell,
        "Deaths over 5": nets.deaths_averted_over_5_givewell,
        "Income increase": nets.income_increase_under_15_givewell,
        "Results": nets.results_givewell,
    },
}
