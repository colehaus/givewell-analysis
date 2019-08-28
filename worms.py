#!/usr/bin/env python3

from utility import present_value_of_annuity

long_term_income_effects_givewell = {
    "Deworming: effect on ln income in MK pop": 0.143,
    "Deworming: num yrs between deworming and benefits": 8,
    "Deworming: duration of benefits": 40,
    "Deworming: multiplier for sharing within households": 2.0,
    "Deworming: adjustment for el nino": 0.65,
    "Deworming: adjustment for yrs of treatment in MK vs programs": 0.90,
    "Deworming: replicability adjustment": 0.11,
    "Deworming: additional yrs for treatment group in MK": 1.69,
}


def long_term_income_effects(
    effect_on_ln_income_in_MK_pop,
    discount_rate,
    num_yrs_between_deworming_and_benefits,
    duration_of_benefits,
    multiplier_for_sharing_within_households,
    adjustment_for_el_nino,
    adjustment_for_yrs_of_treatment_in_MK_vs_programs,
    replicability_adjustment,
    additional_yrs_for_treatment_group_in_MK,
):

    benefit_on_one_years_income = effect_on_ln_income_in_MK_pop / (
        (1 + discount_rate) ** num_yrs_between_deworming_and_benefits
    )
    present_value_of_lifetime_benefits_from_year_of_deworming = present_value_of_annuity(  # TODO: end of period
        discount_rate, duration_of_benefits, benefit_on_one_years_income
    )
    adjusted_benefits_per_yr_of_deworming_in_ln_consumption = (
        present_value_of_lifetime_benefits_from_year_of_deworming
        * multiplier_for_sharing_within_households
        * adjustment_for_el_nino
        * adjustment_for_yrs_of_treatment_in_MK_vs_programs
        * replicability_adjustment
        / additional_yrs_for_treatment_group_in_MK
    )
    return {
        "Deworming: adjusted benefits per yr of deworming in ln consumption": adjusted_benefits_per_yr_of_deworming_in_ln_consumption
    }


dtw_givewell = {
    "DTW: worm intensity adjustment": 0.12,
    "DTW: cost per capita per annum": 0.61,
    "DTW: expected value from lev/fun": 0.71,
}

sci_givewell = {
    "SCI: worm intensity adjustment": 0.09,
    "SCI: cost per capita per annum": 0.99,
    "SCI: expected value from lev/fun": 1.31,
}

sightsavers_givewell = {
    "SS: worm intensity adjustment": 0.09,
    "SS: cost per capita per annum": 0.95,
    "SS: expected value from lev/fun": 0.91,
}

end_givewell = {
    "END: worm intensity adjustment": 0.07,
    "END: cost per capita per annum": 0.81,
    "END: expected value from lev/fun": 0.39,
}


def charity_specific(
    name,
    adjusted_benefits_per_yr_of_deworming_in_ln_consumption,
    worm_intensity_adjustment,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    cost_per_capita_per_annum,
    expected_value_from_levfun,
):
    proportion_of_deworming_going_to_children = 1.00
    present_value_of_lifetime_benefits_from_year_of_deworming = (
        adjusted_benefits_per_yr_of_deworming_in_ln_consumption
        * proportion_of_deworming_going_to_children
        * worm_intensity_adjustment
    )
    value_from_each_year_of_deworming = (
        present_value_of_lifetime_benefits_from_year_of_deworming
        * value_of_increasing_ln_consumption_per_capita_per_annum
    )
    value_per_dollar = value_from_each_year_of_deworming / cost_per_capita_per_annum
    value_per_dollar_w_levfun = value_per_dollar * (1 + expected_value_from_levfun)
    return {name + ": value per dollar w/ lev/fun": value_per_dollar_w_levfun}
