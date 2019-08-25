#!/usr/bin/env python3

import utility

long_term_income_effects_givewell = {
    "Deworming: treatment_effect_on_ln_income_in_MK_study_population": 0.143,
    "Deworming: average_num_of_years_between_deworming_and_beginning_of_benefits": 8,
    "Deworming: duration_of_benefits_of_deworming": 40,
    "Deworming: multiplier_for_resource_sharing_within_households": 2.0,
    "Deworming: adjustment_for_el_nino": 0.65,
    "Deworming: adjustment_for_years_of_treatment_in_MK_vs_charities_programs": 0.90,
    "Deworming: replicability_adjustment_for_deworming": 0.11,
    "Deworming: additional_years_of_treatment_assigned_to_treatment_group_from_MK": 1.69,
}


def long_term_income_effects(
    treatment_effect_on_ln_income_in_MK_study_population,
    discount_rate,
    average_num_of_years_between_deworming_and_beginning_of_benefits,
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
    present_value_of_lifetime_benefits_from_year_of_deworming = utility.present_value_of_annuity(  # TODO: end of period
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
    return {
        "Deworming: adjusted_benefits_per_year_of_deworming": adjusted_benefits_per_year_of_deworming
    }


dtw_givewell = {
    "DTW: worm_intensity_adjustment": 0.12,
    "DTW: cost_per_capita_per_annum": 0.61,
    "DTW: total_additional_expected_value_from_leverage_and_funging": 0.71,
}

sci_givewell = {
    "SCI: worm_intensity_adjustment": 0.09,
    "SCI: cost_per_capita_per_annum": 0.99,
    "SCI: total_additional_expected_value_from_leverage_and_funging": 1.31,
}

sightsavers_givewell = {
    "SS: worm_intensity_adjustment": 0.09,
    "SS: cost_per_capita_per_annum": 0.95,
    "SS: total_additional_expected_value_from_leverage_and_funging": 0.91,
}

end_givewell = {
    "END: worm_intensity_adjustment": 0.07,
    "END: cost_per_capita_per_annum": 0.81,
    "END: total_additional_expected_value_from_leverage_and_funging": 0.39,
}


def charity_specific(
    name,
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
    return {
        name
        + ": value_per_dollar_after_accounting_for_leverage_and_funging": value_per_dollar_after_accounting_for_leverage_and_funging
    }
