#!/usr/bin/env python3

import utility
from utility import present_value_of_annuity
import pymc3 as pm

deworming_staff_aggregates = {
    "treatment_effect_on_ln_income_in_MK_study_population": 0.143,
    "average_num_of_years_between_deworming_and_beginning_of_benefits": 8,
    "duration_of_benefits_of_deworming": 40,
    "multiplier_for_resource_sharing_within_households": 2.0,
    "adjustment_for_el_nino": 0.65,
    "adjustment_for_years_of_treatment_in_MK_vs_charities_programs": 0.90,
    "replicability_adjustment_for_deworming": 0.11,
    "additional_years_of_treatment_assigned_to_treatment_group_from_MK": 1.69,
}


def long_term_income_effects(
    treatment_effect_on_ln_income_in_MK_study_population,
    discount_rate,
    average_num_of_years_between_deworming_and_beginning_of_benefits,  # TODO
    duration_of_benefits_of_deworming,
    multiplier_for_resource_sharing_within_households,
    adjustment_for_el_nino,
    adjustment_for_years_of_treatment_in_MK_vs_charities_programs,
    replicability_adjustment_for_deworming,
    additional_years_of_treatment_assigned_to_treatment_group_from_MK,
):

    benefit_on_one_years_income = (
        treatment_effect_on_ln_income_in_MK_study_population
        / (
            (1 + discount_rate)
            ** average_num_of_years_between_deworming_and_beginning_of_benefits
        )
    )
    present_value_of_lifetime_benefits_from_year_of_deworming = present_value_of_annuity(  # TODO: end of period
        discount_rate, duration_of_benefits_of_deworming, benefit_on_one_years_income
    )
    adjusted_benefits_per_year_of_deworming = (
        present_value_of_lifetime_benefits_from_year_of_deworming
        * multiplier_for_resource_sharing_within_households
        * adjustment_for_el_nino
        * adjustment_for_years_of_treatment_in_MK_vs_charities_programs
        * replicability_adjustment_for_deworming
        / additional_years_of_treatment_assigned_to_treatment_group_from_MK
    )
    return adjusted_benefits_per_year_of_deworming


dtw_staff_aggregates = {
    "worm_intensity_adjustment": 0.12,
    "cost_per_capita_per_annum": 0.61,
    "total_additional_expected_value_from_leverage_and_funging": 0.71,
}

sci_staff_aggregates = {
    "worm_intensity_adjustment": 0.09,
    "cost_per_capita_per_annum": 0.99,
    "total_additional_expected_value_from_leverage_and_funging": 1.31,
}

sightsavers_staff_aggregates = {
    "worm_intensity_adjustment": 0.09,
    "cost_per_capita_per_annum": 0.95,
    "total_additional_expected_value_from_leverage_and_funging": 0.91,
}

end_staff_aggregates = {
    "worm_intensity_adjustment": 0.07,
    "cost_per_capita_per_annum": 0.81,
    "total_additional_expected_value_from_leverage_and_funging": 0.39,
}


def charity_specific(
    adjusted_benefits_per_year_of_deworming,
    worm_intensity_adjustment,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    cost_per_capita_per_annum,
    total_additional_expected_value_from_leverage_and_funging,
):
    proportion_of_deworming_going_to_children = 1.00
    present_value_of_lifetime_benefits_from_year_of_deworming = (
        adjusted_benefits_per_year_of_deworming
        * proportion_of_deworming_going_to_children
        * worm_intensity_adjustment
    )
    value_from_each_year_of_deworming = (
        present_value_of_lifetime_benefits_from_year_of_deworming
        * value_of_increasing_ln_consumption_per_capita_per_annum
    )
    value_per_dollar = value_from_each_year_of_deworming / cost_per_capita_per_annum
    value_per_dollar_after_accounting_for_leverage_and_funging = value_per_dollar * (
        1 + total_additional_expected_value_from_leverage_and_funging
    )
    return value_per_dollar_after_accounting_for_leverage_and_funging


def combined(**kwargs):
    adjusted_benefits_per_year_of_deworming = utility.call_with_only_required_arguments(
        long_term_income_effects, kwargs
    )
    args = {
        "adjusted_benefits_per_year_of_deworming": adjusted_benefits_per_year_of_deworming,
        **kwargs,
    }
    return utility.call_with_only_required_arguments(charity_specific, args)
