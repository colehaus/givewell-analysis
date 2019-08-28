#!/usr/bin/env python3

from pymc3.math import log

from utility import present_value_of_annuity

nets_givewell = {
    "net use adjustment": 0.90,
    "internal validity adjustment": 0.95,
    "percent of mortality due to malaria in AMF areas vs trials": 1.00,
    "efficacy reduction due to insecticide resistance": 0.25,
}

effectiveness_givewell = {
    "deaths averted per protected child under 5": 0.00553,
    "under 5 all cause mortality in trials": 34.8,
    "under 5 all cause mortality in AMF areas": 13.8,
    "percent of mortality differerence due to ITNs": 0.25,
}


def effectiveness(
    deaths_averted_per_protected_child_under_5,
    under_5_all_cause_mortality_in_trials,
    under_5_all_cause_mortality_in_AMF_areas,
    percent_of_mortality_differerence_due_to_ITNs,
    net_use_adjustment,
    internal_validity_adjustment,
    percent_of_mortality_due_to_malaria_in_AMF_areas_vs_trials,
    efficacy_reduction_due_to_insecticide_resistance,
):
    mortality_in_AMF_vs_study_areas = (
        under_5_all_cause_mortality_in_AMF_areas / under_5_all_cause_mortality_in_trials
    )
    mortality_in_AMF_areas_vs_study_wout_ITN_effects = (
        mortality_in_AMF_vs_study_areas
        + (1 - mortality_in_AMF_vs_study_areas)
        * percent_of_mortality_differerence_due_to_ITNs
    )
    deaths_averted_per_child_protected_after_adjusting_for_reduced_mortality_today = (
        deaths_averted_per_protected_child_under_5
        * mortality_in_AMF_areas_vs_study_wout_ITN_effects
    )
    adjusted_deaths_averted_per_child_under_5_targeted = (
        deaths_averted_per_child_protected_after_adjusting_for_reduced_mortality_today
        * net_use_adjustment
        * (1 - efficacy_reduction_due_to_insecticide_resistance)
        * internal_validity_adjustment
        * percent_of_mortality_due_to_malaria_in_AMF_areas_vs_trials
    )
    return {
        "adjusted deaths averted per child under 5 targeted": adjusted_deaths_averted_per_child_under_5_targeted
    }


pre_existing_nets_givewell = {
    "percent of people own nets w/out distribution": 0.20,
    "percent of extra nets distributed used": 0.50,
}


def pre_existing_nets(
    percent_of_people_own_nets_wout_distribution, percent_of_extra_nets_distributed_used
):
    adjustment_for_pre_existing_nets = (
        1
        - percent_of_people_own_nets_wout_distribution
        * (1 - percent_of_extra_nets_distributed_used)
    )
    return {"adjustment for pre-existing nets": adjustment_for_pre_existing_nets}


reduction_in_prevalence_from_net_distributions_givewell = {
    "reduction in incidence for children under 5": 0.50,
    "malaria prevalance in AMF areas under 5": 0.14,
    "malaria prevalance in AMF areas 5 to 14": 0.18,
    "increase in malaria prevalance in absence of LLIN distribution": 0.20,
}


def reduction_in_prevalence_from_net_distributions(
    reduction_in_incidence_for_children_under_5,
    net_use_adjustment,
    internal_validity_adjustment,
    percent_of_mortality_due_to_malaria_in_AMF_areas_vs_trials,
    efficacy_reduction_due_to_insecticide_resistance,
    adjustment_for_pre_existing_nets,
    malaria_prevalance_in_AMF_areas_under_5,
    malaria_prevalance_in_AMF_areas_5_to_14,
    increase_in_malaria_prevalance_in_absence_of_LLIN_distribution,
):
    reduction_in_prevalence_in_distributions = (
        reduction_in_incidence_for_children_under_5
        * net_use_adjustment
        * internal_validity_adjustment
        * percent_of_mortality_due_to_malaria_in_AMF_areas_vs_trials
        * (1 - efficacy_reduction_due_to_insecticide_resistance)
        * adjustment_for_pre_existing_nets
    )
    adjusted_malaria_prevalence_in_AMF_areas_under_5 = (
        malaria_prevalance_in_AMF_areas_under_5
        * (1 + increase_in_malaria_prevalance_in_absence_of_LLIN_distribution)
    )
    adjusted_malaria_prevalence_in_AMF_areas_5_to_14 = (
        malaria_prevalance_in_AMF_areas_5_to_14
        * (1 + increase_in_malaria_prevalance_in_absence_of_LLIN_distribution)
    )
    reduction_in_prob_of_covered_child_being_infected_under_5 = (
        reduction_in_prevalence_in_distributions
        * adjusted_malaria_prevalence_in_AMF_areas_under_5
    )
    reduction_in_prob_of_covered_child_being_infected_5_to_14 = (
        reduction_in_prevalence_in_distributions
        * adjusted_malaria_prevalence_in_AMF_areas_5_to_14
    )
    return {
        "reduction in prob. of covered child being infected under 5": reduction_in_prob_of_covered_child_being_infected_under_5,
        "reduction in prob. of covered child being infected 5 to 14": reduction_in_prob_of_covered_child_being_infected_5_to_14,
    }


cost_per_yr_of_protection_givewell = {
    "pre-distribution wastage": 0.05,
    "cost per LLIN": 4.525938148,
    "num LLINs distributed per person": 0.5555556,
    "percent of pop. under 5": 0.17,
    "percent of pop. 5 to 14": 0.31,
    "lifespan of an LLIN": 2.22,
}

arbitrary_donation_size = 1000000


def cost_per_yr_of_protection(
    pre_distribution_wastage,
    cost_per_LLIN,
    num_LLINs_distributed_per_person,
    percent_of_pop_under_5,
    percent_of_pop_5_to_14,
    lifespan_of_an_LLIN,
):
    remaining_dollars_available_for_purchasing_LLIN = arbitrary_donation_size * (
        1 - pre_distribution_wastage
    )
    cost_per_person_covered_in_universal_distribution = (
        cost_per_LLIN * num_LLINs_distributed_per_person
    )
    number_of_people_covered = (
        remaining_dollars_available_for_purchasing_LLIN
        / cost_per_person_covered_in_universal_distribution
    )
    children_under_5_covered = number_of_people_covered * percent_of_pop_under_5
    children_5_to_14_covered = number_of_people_covered * percent_of_pop_5_to_14
    person_yrs_of_coverage_for_under_5s = children_under_5_covered * lifespan_of_an_LLIN
    person_yrs_of_coverage_for_5_to_14s = children_5_to_14_covered * lifespan_of_an_LLIN
    person_yrs_of_coverage_for_under_15s = (
        person_yrs_of_coverage_for_under_5s + person_yrs_of_coverage_for_5_to_14s
    )
    cost_per_person_yr_of_protection_for_under_15s = (
        arbitrary_donation_size / person_yrs_of_coverage_for_under_15s
    )
    return {
        "person-yrs of coverage for under 5s": person_yrs_of_coverage_for_under_5s,
        "person-yrs of coverage for 5 to 14s": person_yrs_of_coverage_for_5_to_14s,
        "cost per person-yr of protection for under 15s": cost_per_person_yr_of_protection_for_under_15s,
    }


def deaths_averted_children_under_5(
    person_yrs_of_coverage_for_under_5s,
    adjusted_deaths_averted_per_child_under_5_targeted,
    adjustment_for_pre_existing_nets,
    value_of_averting_death_of_a_young_child,
):
    under_5_deaths_averted_per_million_dollars = (
        person_yrs_of_coverage_for_under_5s
        * adjusted_deaths_averted_per_child_under_5_targeted
        * adjustment_for_pre_existing_nets
    )
    cost_per_under_5_death_averted_wout_levfun = (
        arbitrary_donation_size / under_5_deaths_averted_per_million_dollars
    )
    value_from_under_5_deaths_averted_per_dollar_wout_levfun = (
        value_of_averting_death_of_a_young_child
        / cost_per_under_5_death_averted_wout_levfun
    )
    return {
        "under 5 deaths averted per million dollars": under_5_deaths_averted_per_million_dollars,
        "value from under 5 deaths averted per dollar w/out lev/fun": value_from_under_5_deaths_averted_per_dollar_wout_levfun,
    }


deaths_averted_over_5_givewell = {
    "ratio_of_5_and_over_to_under_5_malaria_deaths": 0.38,
    "relative_efficacy_of_LLINs_for_people_over_5": 0.80,
}


def deaths_averted_over_5(
    ratio_of_5_and_over_to_under_5_malaria_deaths,
    relative_efficacy_of_LLINs_for_people_over_5,
    value_of_averting_death_of_a_person_5_or_older,
    under_5_deaths_averted_per_million_dollars,
):
    total_number_of_over_5_deaths_averted_per_million_donated = (
        under_5_deaths_averted_per_million_dollars
        * ratio_of_5_and_over_to_under_5_malaria_deaths
        * relative_efficacy_of_LLINs_for_people_over_5
    )
    cost_per_over_5_death_averted_wout_levfun = (
        arbitrary_donation_size
        / total_number_of_over_5_deaths_averted_per_million_donated
    )
    value_from_over_5_deaths_averted_per_dollar_wout_levfun = (
        value_of_averting_death_of_a_person_5_or_older
        / cost_per_over_5_death_averted_wout_levfun
    )
    return {
        "value from over 5 deaths averted per dollar w/out lev/fun": value_from_over_5_deaths_averted_per_dollar_wout_levfun
    }


income_increase_under_15_givewell = {
    "AMF: increase in income from eliminating prob. of infection in youth": 0.023,
    "AMF: replicability adjustment for malaria vs income": 0.52,
    "AMF: num yrs between program and long-term benefits": 10,
    "AMF: duration of long-term benefits": 40,
    "AMF: multiplier for sharing w/in households": 2,
}


# TODO: Find stripped duplicates
def income_increase_under_15(
    person_yrs_of_coverage_for_under_5s,
    person_yrs_of_coverage_for_5_to_14s,
    reduction_in_prob_of_covered_child_being_infected_under_5,
    reduction_in_prob_of_covered_child_being_infected_5_to_14,
    increase_in_income_from_eliminating_prob_of_infection_in_youth,
    replicability_adjustment_for_malaria_vs_income,
    num_yrs_between_program_and_long_term_benefits,
    discount_rate,
    duration_of_long_term_benefits,
    multiplier_for_sharing_win_households,
    value_of_increasing_ln_consumption_per_capita_per_annum,
):
    reduction_in_num_people_infected_at_point_in_time_in_cohort_under_5 = (
        person_yrs_of_coverage_for_under_5s
        * reduction_in_prob_of_covered_child_being_infected_under_5
    )
    reduction_in_num_people_infected_at_point_in_time_in_cohort_5_to_14 = (
        person_yrs_of_coverage_for_5_to_14s
        * reduction_in_prob_of_covered_child_being_infected_5_to_14
    )
    adjusted_increase_in_ln_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_an_invidiual_between_the_ages_of_0_and_14 = (
        (
            log(1 + increase_in_income_from_eliminating_prob_of_infection_in_youth)
            - log(1)
        )
        * replicability_adjustment_for_malaria_vs_income
    )
    benefit_on_one_years_income = (
        adjusted_increase_in_ln_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_an_invidiual_between_the_ages_of_0_and_14
        / (1 + discount_rate) ** num_yrs_between_program_and_long_term_benefits
    )
    present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14 = present_value_of_annuity(
        discount_rate, duration_of_long_term_benefits, benefit_on_one_years_income
    )  # TODO: end of period
    present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14 = (
        present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14
        * multiplier_for_sharing_win_households
    )
    total_increase_in_ln_income_under_5 = (
        present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        * reduction_in_num_people_infected_at_point_in_time_in_cohort_under_5
    )
    total_increase_in_ln_income_5_to_14 = (
        present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        * reduction_in_num_people_infected_at_point_in_time_in_cohort_5_to_14
    )
    value_from_development_benefits_per_million_donated = (
        value_of_increasing_ln_consumption_per_capita_per_annum
        * (total_increase_in_ln_income_5_to_14 + total_increase_in_ln_income_under_5)
    )
    value_from_development_benefits_per_dollar_wout_levfun = (
        value_from_development_benefits_per_million_donated / arbitrary_donation_size
    )
    return {
        "value from development benefits per dollar w/out lev/fun": value_from_development_benefits_per_dollar_wout_levfun
    }


results_givewell = {"expected cost from lev/fun": 0.1568}


def results(
    value_from_development_benefits_per_dollar_wout_levfun,
    value_from_over_5_deaths_averted_per_dollar_wout_levfun,
    value_from_under_5_deaths_averted_per_dollar_wout_levfun,
    expected_cost_from_levfun,
):
    value_per_dollar_wout_levfun = (
        value_from_development_benefits_per_dollar_wout_levfun
        + value_from_over_5_deaths_averted_per_dollar_wout_levfun
        + value_from_under_5_deaths_averted_per_dollar_wout_levfun
    )
    value_per_dollar_w_levfun = value_per_dollar_wout_levfun * (
        1 - expected_cost_from_levfun
    )
    return {"value per dollar w/ lev/fun": value_per_dollar_w_levfun}
