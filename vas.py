#!/usr/bin/env python3

num_children_in_cohort = 10000

vas_givewell = {"VAS: cost per child per round": 1.35, "VAS: rounds per yr": 2}

cost_of_covering_a_hypothetical_cohort_with_vitamin_a_supplementation_givewell = {
    "VAS: coverage in RCTs": 0.873
}


def cost_of_covering_a_hypothetical_cohort_with_vitamin_a_supplementation(
    cost_per_child_per_round, rounds_per_yr, coverage_in_RCTs
):
    cost_for_cohort_per_year = (
        cost_per_child_per_round
        * rounds_per_yr
        * num_children_in_cohort
        * coverage_in_RCTs
    )
    return {"VAS: cost for cohort per yr": cost_for_cohort_per_year}


meta_analysis_finding_on_relative_rate_of_mortality_reduction_givewell = {
    "VAS: relative risk of all-cause mortality for young children in programs": 0.76
}


def meta_analysis_finding_on_relative_rate_of_mortality_reduction(
    relative_risk_of_all_cause_mortality_for_young_children_in_programs
):
    reduction_in_all_cause_mortality = (
        1 - relative_risk_of_all_cause_mortality_for_young_children_in_programs
    )
    return {"VAS: reduction in all-cause mortality": reduction_in_all_cause_mortality}


mortality_reduction_in_hypothetical_cohort_givewell = {
    "VAS: baseline_deaths_per_thousand_child_years_for_young_children": 12.1,
    "VAS: internal_validity_adjustment": 0.95,
    "VAS: external_validity_adjustment": 0.23,
}


def mortality_reduction_in_hypothetical_cohort(
    baseline_deaths_per_thousand_child_years_for_young_children,
    reduction_in_all_cause_mortality,
    internal_validity_adjustment,
    external_validity_adjustment,
):
    expected_deaths_in_cohort_over_one_year_in_absence_of_VAS_program = (
        baseline_deaths_per_thousand_child_years_for_young_children
        * num_children_in_cohort
        / 1000
    )
    expected_deaths_averted_in_cohort_due_to_program_unadjusted = (
        expected_deaths_in_cohort_over_one_year_in_absence_of_VAS_program
        * reduction_in_all_cause_mortality
    )
    deaths_averted_in_cohort_due_to_program = (
        expected_deaths_averted_in_cohort_due_to_program_unadjusted
        * internal_validity_adjustment
        * external_validity_adjustment
    )
    return {
        "VAS: deaths averted in cohort due to program": deaths_averted_in_cohort_due_to_program
    }


development_benefits_givewell = {
    "VAS: value of development from yr of VAS relative to yr of deworming": 0.07
}


def development_benefits(
    value_of_development_from_yr_of_VAS_relative_to_yr_of_deworming,
    adjusted_benefits_per_yr_of_deworming_in_ln_consumption,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    cost_per_child_per_round,
    rounds_per_yr,
):
    value_due_to_development_per_dollar_wout_levfun = (
        value_of_development_from_yr_of_VAS_relative_to_yr_of_deworming
        * adjusted_benefits_per_yr_of_deworming_in_ln_consumption
        * value_of_increasing_ln_consumption_per_capita_per_annum
    ) / (cost_per_child_per_round * rounds_per_yr)
    return {
        "VAS: value due to development per dollar w/out lev/fun": value_due_to_development_per_dollar_wout_levfun
    }


results_givewell = {"VAS: expected value from lev/fun": 0.18}


def results(
    cost_for_cohort_per_yr,
    deaths_averted_in_cohort_due_to_program,
    value_of_averting_death_of_a_young_child,
    value_due_to_development_per_dollar_wout_levfun,
    expected_value_from_levfun,
):
    cost_per_young_childs_death_averted_wout_levfun = (
        cost_for_cohort_per_yr / deaths_averted_in_cohort_due_to_program
    )
    value_per_dollar_wout_levfun = (
        1
        / cost_per_young_childs_death_averted_wout_levfun
        * value_of_averting_death_of_a_young_child
        + value_due_to_development_per_dollar_wout_levfun
    )
    value_per_dollar_w_levfun = value_per_dollar_wout_levfun * (
        1 + expected_value_from_levfun
    )
    return {"VAS: value per dollar w/ lev/fun": value_per_dollar_w_levfun}
