#!/usr/bin/env python3

import utility
from pymc3 import math

smc_givewell = {
    "cost_per_child_targeted": 4.54,
    "proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season": 0.7,
    "internal_validity_adjustment_SMC": 0.95,
    "external_validity_adjustment_SMC": 1.00,
    "unadjusted_expected_reduction_in_malaria_cases_in_untreated_population_as_percentage_of_reduction_in_treated_population": 0.43,
    "adjustment_to_account_for_higher_proportion_of_people_being_covered_in_trial_than_ACCESS_SMC": 0.31,
}

adherance_adjustment_givewell = {
    "percent_of_first_doses_that_arent_directly_observed": 0.30,
    "adherence_rate_to_first_doses_that_arent_directly_observed": 0.84,
    "adherence_rate_to_second_and_third_doses": 0.84,
    "reduction_in_efficacy_from_non_adherence_to_2nd_and_3rd_doses": 0.50,
}


def adherence_adjustment(
    percent_of_first_doses_that_arent_directly_observed,
    adherence_rate_to_first_doses_that_arent_directly_observed,
    adherence_rate_to_second_and_third_doses,
    reduction_in_efficacy_from_non_adherence_to_2nd_and_3rd_doses,
):
    overall_adherence_adjustment = (
        1
        - (
            percent_of_first_doses_that_arent_directly_observed
            - percent_of_first_doses_that_arent_directly_observed
            * adherence_rate_to_first_doses_that_arent_directly_observed
        )
        - (
            (1 - adherence_rate_to_second_and_third_doses)
            * reduction_in_efficacy_from_non_adherence_to_2nd_and_3rd_doses
        )
    )
    return {"overall_adherence_adjustment": overall_adherence_adjustment}


coverage_adjustment_givewell = {
    "percent_of_targeted_children_who_received_at_least_1_round_of_treatment": 0.92,
    "percent_of_targeted_children_who_received_at_least_2_rounds_of_treatment": 0.83,
    "percent_of_targeted_children_who_received_at_least_3_rounds_of_treatment": 0.71,
    "percent_of_targeted_children_who_received_all_4_rounds_of_treatment": 0.71,
    "estimated_coverage_in_trials_considered_in_givewells_meta_analysis": 0.55,
}


def coverage_adjustment(
    percent_of_targeted_children_who_received_at_least_1_round_of_treatment,
    percent_of_targeted_children_who_received_at_least_2_rounds_of_treatment,
    percent_of_targeted_children_who_received_at_least_3_rounds_of_treatment,
    percent_of_targeted_children_who_received_all_4_rounds_of_treatment,
    estimated_coverage_in_trials_considered_in_givewells_meta_analysis,
):
    percent_of_children_who_received_exactly_1_round = (
        percent_of_targeted_children_who_received_at_least_1_round_of_treatment
        - percent_of_targeted_children_who_received_at_least_2_rounds_of_treatment
    )
    percent_of_children_who_received_exactly_2_round = (
        percent_of_targeted_children_who_received_at_least_2_rounds_of_treatment
        - percent_of_targeted_children_who_received_at_least_3_rounds_of_treatment
    )
    percent_of_children_who_received_exactly_3_round = (
        percent_of_targeted_children_who_received_at_least_3_rounds_of_treatment
        - percent_of_targeted_children_who_received_all_4_rounds_of_treatment
    )
    average_number_of_treatments = (
        percent_of_targeted_children_who_received_all_4_rounds_of_treatment * 4
        + percent_of_children_who_received_exactly_3_round * 3
        + percent_of_children_who_received_exactly_2_round * 2
        + percent_of_children_who_received_exactly_1_round * 1
    )
    coverage_in_ACCESS_SMC_program = average_number_of_treatments / 4
    coverage_in_ACCESS_SMC_program_relative_to_coverage_in_RCTs = (
        coverage_in_ACCESS_SMC_program
        / estimated_coverage_in_trials_considered_in_givewells_meta_analysis
    )
    return coverage_in_ACCESS_SMC_program_relative_to_coverage_in_RCTs


smc_effectiveness_givewell = {
    "relative_risk_for_malaria_cases_intention_to_treat_effect": 0.25
}


def smc_effectivness(
    relative_risk_for_malaria_cases_intention_to_treat_effect,
    coverage_in_ACCESS_SMC_program_relative_to_coverage_in_RCTs,
    overall_adherence_adjustment,
):
    corresponding_reduction_in_clinical_malaria = (
        1 - relative_risk_for_malaria_cases_intention_to_treat_effect
    )
    total_adjustment_to_meta_analysis_finding_due_to_imperfect_adherence_and_coverage = (
        coverage_in_ACCESS_SMC_program_relative_to_coverage_in_RCTs
        * overall_adherence_adjustment
    )
    expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments = (
        corresponding_reduction_in_clinical_malaria
        * total_adjustment_to_meta_analysis_finding_due_to_imperfect_adherence_and_coverage
    )
    return {
        "expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments": expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments
    }


def costs(cost_per_child_targeted):
    cost_to_cover_hypothetical_cohort = (
        cost_per_child_targeted * hypothetical_cohort_size
    )
    return {"cost_to_cover_hypothetical_cohort": cost_to_cover_hypothetical_cohort}


hypothetical_cohort_size = 1000000

treated_population_givewell = {
    "young_all_cause_mortality_per_1000_per_annum": 14.2,
    "percent_of_young_deaths_due_to_malaria": 0.27,
    "ratio_of_indirect_to_direct_malaria_deaths": 0.5,
    "ratio_of_reduction_in_malaria_mortality_to_reduction_in_malaria_incidence": 1.0,
}


def mortality_reduction_in_treated_population(
    cost_to_cover_hypothetical_cohort,
    expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments,
    young_all_cause_mortality_per_1000_per_annum,
    percent_of_young_deaths_due_to_malaria,
    ratio_of_indirect_to_direct_malaria_deaths,
    proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season,
    ratio_of_reduction_in_malaria_mortality_to_reduction_in_malaria_incidence,
    internal_validity_adjustment_SMC,
    external_validity_adjustment_SMC,
    value_of_averting_death_of_a_young_child,
):
    estimated_percent_of_young_deaths_attributable_to_malaria = (
        percent_of_young_deaths_due_to_malaria
        * (1 + ratio_of_indirect_to_direct_malaria_deaths)
    )
    malaria_attributable_deaths_in_ACCESS_SMCs_target_population_per_1000_under_5_person_years = (
        estimated_percent_of_young_deaths_attributable_to_malaria
        * young_all_cause_mortality_per_1000_per_annum
    )
    malaria_attributable_deaths_per_1000_under_5s_during_high_transmission_season = (
        malaria_attributable_deaths_in_ACCESS_SMCs_target_population_per_1000_under_5_person_years
        * proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season
    )
    number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments = (
        malaria_attributable_deaths_per_1000_under_5s_during_high_transmission_season
        * ratio_of_reduction_in_malaria_mortality_to_reduction_in_malaria_incidence
        * expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments
    )
    expected_deaths_averted_in_treated_cohort_after_adjustments = (
        hypothetical_cohort_size
        * number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments
        * internal_validity_adjustment_SMC
        * external_validity_adjustment_SMC
        / 1000
    )
    cost_per_young_death_averted_before_accounting_for_leveraging_and_funging = (
        cost_to_cover_hypothetical_cohort
        / expected_deaths_averted_in_treated_cohort_after_adjustments
    )
    value_from_under_5_deaths_averted_per_dollar_before_accounting_for_leverage_and_funging = (
        value_of_averting_death_of_a_young_child
        / cost_per_young_death_averted_before_accounting_for_leveraging_and_funging
    )
    return {
        "number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments": number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments,
        "value_from_under_5_deaths_averted_per_dollar_before_account_for_leverage_and_funging": value_from_under_5_deaths_averted_per_dollar_before_accounting_for_leverage_and_funging,
    }


untreated_population_givewell = {
    "ratio_of_malaria_deaths_at_all_ages_to_malaria_deaths_in_young": 1.34
}


def mortality_reduction_in_untreated_population(
    unadjusted_expected_reduction_in_malaria_cases_in_untreated_population_as_percentage_of_reduction_in_treated_population,
    ratio_of_malaria_deaths_at_all_ages_to_malaria_deaths_in_young,
    number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments,
    internal_validity_adjustment_SMC,
    external_validity_adjustment_SMC,
    adjustment_to_account_for_higher_proportion_of_people_being_covered_in_trial_than_ACCESS_SMC,
    cost_to_cover_hypothetical_cohort,
    value_of_averting_death_of_an_individual_5_or_older,
):
    number_of_malaria_attributable_deaths_prevented_in_untreated_population_per_1000_under_5s_targeted_prior_to_adjustments = (
        number_of_malaria_attributable_deaths_prevented_per_1000_under_5s_targeted_prior_to_adjustments
        * unadjusted_expected_reduction_in_malaria_cases_in_untreated_population_as_percentage_of_reduction_in_treated_population
        * (ratio_of_malaria_deaths_at_all_ages_to_malaria_deaths_in_young - 1)
    )
    number_of_malaria_attributable_deaths_prevented_in_community_per_1000_under_5s_targeted_after_adjustments = (
        number_of_malaria_attributable_deaths_prevented_in_untreated_population_per_1000_under_5s_targeted_prior_to_adjustments
        * internal_validity_adjustment_SMC
        * external_validity_adjustment_SMC
        * adjustment_to_account_for_higher_proportion_of_people_being_covered_in_trial_than_ACCESS_SMC
    )
    expected_deaths_averted_in_untreated_population_after_adjustments = (
        number_of_malaria_attributable_deaths_prevented_in_community_per_1000_under_5s_targeted_after_adjustments
        * hypothetical_cohort_size
        / 1000
    )
    cost_per_death_averted_in_untreated_population_before_accounting_for_leverage_and_funging = (
        cost_to_cover_hypothetical_cohort
        / expected_deaths_averted_in_untreated_population_after_adjustments
    )
    value_from_over_5_deaths_averted_for_each_dollar_donated_before_accounting_for_leverage_and_funging = (
        value_of_averting_death_of_an_individual_5_or_older
        / cost_per_death_averted_in_untreated_population_before_accounting_for_leverage_and_funging
    )

    return {
        "value_from_over_5_deaths_averted_for_each_dollar_donated_before_accounting_for_leverage_and_funging": value_from_over_5_deaths_averted_for_each_dollar_donated_before_accounting_for_leverage_and_funging
    }


income_increase_ages_14_and_under_givewell = {
    "malaria_prevalance_young": 0.29,
    "malaria_prevalence_old": 0.29,
    "increase_in_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14": 0.023,
    "additional_replicability_adjustment_for_relationship_between_malaria_and_income": 0.52,
    "average_number_of_years_between_program_implementation_and_beginning_of_long_term_benefits": 10,
    "duration_of_long_term_benefits_of_SMC": 40,
    "multiplier_for_resource_sharing_within_households": 2,
    "value_of_increasing_ln_consumption_per_capita_per_annum": 1.44,
}


def income_increases_age_14_and_under(
    malaria_prevalance_young,
    malaria_prevalence_old,
    proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season,
    expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments,
    internal_validity_adjustment_SMC,
    external_validity_adjustment_SMC,
    unadjusted_expected_reduction_in_malaria_cases_in_untreated_population_as_percentage_of_reduction_in_treated_population,
    adjustment_to_account_for_higher_proportion_of_people_being_covered_in_trial_than_ACCESS_SMC,
    increase_in_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14,
    additional_replicability_adjustment_for_relationship_between_malaria_and_income,
    average_number_of_years_between_program_implementation_and_beginning_of_long_term_benefits,
    discount_rate,
    duration_of_long_term_benefits_of_SMC,
    multiplier_for_resource_sharing_within_households,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    cost_per_child_targeted,
):
    percentage_reduction_in_malaria_prevalance_in_treated_population_after_adherence_and_coverage_adjustments_young = (
        malaria_prevalance_young
        * proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season
        * expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments
        * internal_validity_adjustment_SMC
        * external_validity_adjustment_SMC
    )
    percentage_reduction_in_malaria_prevalance_in_untreated_population_after_adherence_and_coverage_adjustments_old = (
        malaria_prevalence_old
        * proportion_of_annual_direct_malaria_mortality_occuring_in_high_transmission_season
        * expected_reduction_in_malaria_cases_after_adherence_and_coverage_adjustments
        * internal_validity_adjustment_SMC
        * external_validity_adjustment_SMC
        * unadjusted_expected_reduction_in_malaria_cases_in_untreated_population_as_percentage_of_reduction_in_treated_population
        * adjustment_to_account_for_higher_proportion_of_people_being_covered_in_trial_than_ACCESS_SMC
    )
    reduction_in_number_of_people_infected_with_malaria_at_point_in_time_per_1000_under_5s_targeted_young = (
        percentage_reduction_in_malaria_prevalance_in_treated_population_after_adherence_and_coverage_adjustments_young
        * 1000
    )
    reduction_in_number_of_people_infected_with_malaria_at_point_in_time_per_1000_under_5s_targeted_old = (
        percentage_reduction_in_malaria_prevalance_in_untreated_population_after_adherence_and_coverage_adjustments_old
        * 1000
    )
    increase_in_ln_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14 = (
        (
            math.log(
                1
                + increase_in_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
            )
            - math.log(1)
        )
        * additional_replicability_adjustment_for_relationship_between_malaria_and_income
    )
    benefit_on_one_years_income = (
        increase_in_ln_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        / (1 + discount_rate)
        ** average_number_of_years_between_program_implementation_and_beginning_of_long_term_benefits
    )
    present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14 = utility.present_value_of_annuity(
        discount_rate,
        duration_of_long_term_benefits_of_SMC,
        benefit_on_one_years_income,
    )  # TODO: end of period
    present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14 = (
        present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14
        * multiplier_for_resource_sharing_within_households
    )
    total_increase_in_annual_ln_income_for_young_per_1000_under_5s_targeted = (
        present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        * reduction_in_number_of_people_infected_with_malaria_at_point_in_time_per_1000_under_5s_targeted_young
    )
    total_increase_in_annual_ln_income_for_old_per_1000_under_5s_targeted = (
        present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        * reduction_in_number_of_people_infected_with_malaria_at_point_in_time_per_1000_under_5s_targeted_old
    )
    value_from_development_benefits_per_1000_under_5s_targeted = (
        value_of_increasing_ln_consumption_per_capita_per_annum
        * (
            total_increase_in_annual_ln_income_for_old_per_1000_under_5s_targeted
            + total_increase_in_annual_ln_income_for_young_per_1000_under_5s_targeted
        )
    )
    value_from_development_benefits_per_dollar_before_accounting_for_leverage_and_funging = (
        value_from_development_benefits_per_1000_under_5s_targeted
        / (cost_per_child_targeted * 1000)
    )
    return {
        "value_from_development_benefits_per_dollar_before_accounting_for_leverage_and_funging": value_from_development_benefits_per_dollar_before_accounting_for_leverage_and_funging
    }


results_givewell = {
    "total_additional_expected_value_from_leverage_funging_SMC": -0.1074
}


def results(
    value_from_development_benefits_per_dollar_before_accounting_for_leverage_and_funging,
    value_from_under_5_deaths_averted_per_dollar_before_accounting_for_leverage_and_funging,
    value_from_over_5_deaths_averted_for_each_dollar_donated_before_accounting_for_leverage_and_funging,
    total_additional_expected_value_from_leverage_funging_SMC,
):
    value_per_dollar_before_accounting_for_leverage_and_funging = (
        value_from_development_benefits_per_dollar_before_accounting_for_leverage_and_funging
        * value_from_under_5_deaths_averted_per_dollar_before_accounting_for_leverage_and_funging
        * value_from_over_5_deaths_averted_for_each_dollar_donated_before_accounting_for_leverage_and_funging
    )
    value_per_dollar_after_accounting_for_leverage_and_funging = (
        value_per_dollar_before_accounting_for_leverage_and_funging
        * (1 + total_additional_expected_value_from_leverage_funging_SMC)
    )
    return {
        "value_per_dollar_after_accounting_for_leverage_and_funging": value_per_dollar_after_accounting_for_leverage_and_funging
    }
