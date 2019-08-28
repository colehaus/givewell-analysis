#!/usr/bin/env python3

from pymc3.math import log

from utility import present_value_of_annuity

smc_givewell = {
    "SMC: cost per child targeted": 4.536588,
    "SMC: direct mortality in high transmission season": 0.7,
    "SMC: internal validity adjustment": 0.95,
    "SMC: external validity adjustment": 1.00,
    "SMC: reduction in untreated pop per reduction in treated pop": 0.43,
    "SMC: adjustment for higher percent covered in trial than ACCESS": 0.31,
}

adherence_adjustment_givewell = {
    "SMC: percent of first doses unobserved": 0.30,
    "SMC: adherence rate to unobserved first doses": 0.84,
    "SMC: adherence rate to second and third doses": 0.84,
    "SMC: efficacy loss from non adherence to 2nd and 3rd doses": 0.50,
}


def adherence_adjustment(
    percent_of_first_doses_unobserved,
    adherence_rate_to_unobserved_first_doses,
    adherence_rate_to_second_and_third_doses,
    efficacy_loss_from_non_adherence_to_2nd_and_3rd_doses,
):
    overall_adherence_adjustment = (
        1
        - (
            percent_of_first_doses_unobserved
            - percent_of_first_doses_unobserved
            * adherence_rate_to_unobserved_first_doses
        )
        - (
            (1 - adherence_rate_to_second_and_third_doses)
            * efficacy_loss_from_non_adherence_to_2nd_and_3rd_doses
        )
    )
    return {"SMC: overall adherence adjustment": overall_adherence_adjustment}


coverage_adjustment_givewell = {
    "SMC: percent of targeted receiving at least 1 round": 0.92,
    "SMC: percent of targeted receiving at least 2 rounds": 0.83,
    "SMC: percent of targeted receiving at least 3 rounds": 0.71,
    "SMC: percent of targeted receiveding all 4 rounds": 0.55,
    "SMC: coverage in trials in meta analysis": 0.90,
}


def coverage_adjustment(
    percent_of_targeted_receiving_at_least_1_round,
    percent_of_targeted_receiving_at_least_2_rounds,
    percent_of_targeted_receiving_at_least_3_rounds,
    percent_of_targeted_receiveding_all_4_rounds,
    coverage_in_trials_in_meta_analysis,
):
    percent_of_children_who_received_exactly_1_round = (
        percent_of_targeted_receiving_at_least_1_round
        - percent_of_targeted_receiving_at_least_2_rounds
    )
    percent_of_children_who_received_exactly_2_round = (
        percent_of_targeted_receiving_at_least_2_rounds
        - percent_of_targeted_receiving_at_least_3_rounds
    )
    percent_of_children_who_received_exactly_3_round = (
        percent_of_targeted_receiving_at_least_3_rounds
        - percent_of_targeted_receiveding_all_4_rounds
    )
    average_number_of_treatments = (
        percent_of_targeted_receiveding_all_4_rounds * 4
        + percent_of_children_who_received_exactly_3_round * 3
        + percent_of_children_who_received_exactly_2_round * 2
        + percent_of_children_who_received_exactly_1_round * 1
    )
    coverage_in_ACCESS_SMC_program = average_number_of_treatments / 4
    coverage_in_ACCESS_vs_in_RCTs = (
        coverage_in_ACCESS_SMC_program / coverage_in_trials_in_meta_analysis
    )
    return {"SMC: coverage in ACCESS vs in RCTs": coverage_in_ACCESS_vs_in_RCTs}


smc_effectiveness_givewell = {"SMC: relative risk for intention to treat": 0.25}


def smc_effectiveness(
    relative_risk_for_intention_to_treat,
    coverage_in_ACCESS_vs_in_RCTs,
    overall_adherence_adjustment,
):
    corresponding_reduction_in_clinical_malaria = (
        1 - relative_risk_for_intention_to_treat
    )
    total_adjustment_to_meta_analysis_finding_due_to_imperfect_adherence_and_coverage = (
        coverage_in_ACCESS_vs_in_RCTs * overall_adherence_adjustment
    )
    reduction_in_malaria_after_adjustments = (
        corresponding_reduction_in_clinical_malaria
        * total_adjustment_to_meta_analysis_finding_due_to_imperfect_adherence_and_coverage
    )
    return {
        "SMC: reduction in malaria after adjustments": reduction_in_malaria_after_adjustments
    }


def costs(cost_per_child_targeted):
    cost_to_cover_cohort = cost_per_child_targeted * hypothetical_cohort_size
    return {"SMC: cost to cover cohort": cost_to_cover_cohort}


hypothetical_cohort_size = 1000000

treated_population_givewell = {
    "SMC: young all-cause mortality per 1000 per annum": 14.2,
    "SMC: percent of young deaths due to malaria": 0.27,
    "SMC: indirect to direct malaria deaths": 0.5,
    "SMC: reduction in mortality per incidence": 1.0,
}


def mortality_reduction_in_treated_population(
    cost_to_cover_cohort,
    reduction_in_malaria_after_adjustments,
    young_all_cause_mortality_per_1000_per_annum,
    percent_of_young_deaths_due_to_malaria,
    indirect_to_direct_malaria_deaths,
    direct_mortality_in_high_transmission_season,
    reduction_in_mortality_per_incidence,
    internal_validity_adjustment,
    external_validity_adjustment,
    value_of_averting_death_of_a_young_child,
):
    estimated_percent_of_young_deaths_attributable_to_malaria = (
        percent_of_young_deaths_due_to_malaria * (1 + indirect_to_direct_malaria_deaths)
    )
    malaria_attributable_deaths_in_ACCESS_SMCs_target_population_per_1000_under_5_person_years = (
        estimated_percent_of_young_deaths_attributable_to_malaria
        * young_all_cause_mortality_per_1000_per_annum
    )
    malaria_attributable_deaths_per_1000_under_5s_during_high_transmission_season = (
        malaria_attributable_deaths_in_ACCESS_SMCs_target_population_per_1000_under_5_person_years
        * direct_mortality_in_high_transmission_season
    )
    unadjusted_deaths_averted_per_1000_under_5s_targeted = (
        malaria_attributable_deaths_per_1000_under_5s_during_high_transmission_season
        * reduction_in_mortality_per_incidence
        * reduction_in_malaria_after_adjustments
    )
    expected_deaths_averted_in_treated_cohort_after_adjustments = (
        hypothetical_cohort_size
        * unadjusted_deaths_averted_per_1000_under_5s_targeted
        * internal_validity_adjustment
        * external_validity_adjustment
        / 1000
    )
    cost_per_young_death_averted_before_accounting_for_leveraging_and_funging = (
        cost_to_cover_cohort
        / expected_deaths_averted_in_treated_cohort_after_adjustments
    )
    value_from_under_5_deaths_averted_per_dollar_wout_levfun = (
        value_of_averting_death_of_a_young_child
        / cost_per_young_death_averted_before_accounting_for_leveraging_and_funging
    )
    return {
        "SMC: unadjusted deaths averted per 1000 under 5s targeted": unadjusted_deaths_averted_per_1000_under_5s_targeted,
        "SMC: value from under 5 deaths averted per dollar w/out lev/fun": value_from_under_5_deaths_averted_per_dollar_wout_levfun,
    }


untreated_population_givewell = {"SMC: deaths at all ages vs deaths in young": 1.34}


def mortality_reduction_in_untreated_population(
    reduction_in_untreated_pop_per_reduction_in_treated_pop,
    deaths_at_all_ages_vs_deaths_in_young,
    unadjusted_deaths_averted_per_1000_under_5s_targeted,
    internal_validity_adjustment,
    external_validity_adjustment,
    adjustment_for_higher_percent_covered_in_trial_than_ACCESS,
    cost_to_cover_cohort,
    value_of_averting_death_of_an_individual_5_or_older,
):
    number_of_malaria_attributable_deaths_prevented_in_untreated_population_per_1000_under_5s_targeted_prior_to_adjustments = (
        unadjusted_deaths_averted_per_1000_under_5s_targeted
        * reduction_in_untreated_pop_per_reduction_in_treated_pop
        * (deaths_at_all_ages_vs_deaths_in_young - 1)
    )
    number_of_malaria_attributable_deaths_prevented_in_community_per_1000_under_5s_targeted_after_adjustments = (
        number_of_malaria_attributable_deaths_prevented_in_untreated_population_per_1000_under_5s_targeted_prior_to_adjustments
        * internal_validity_adjustment
        * external_validity_adjustment
        * adjustment_for_higher_percent_covered_in_trial_than_ACCESS
    )
    expected_deaths_averted_in_untreated_population_after_adjustments = (
        number_of_malaria_attributable_deaths_prevented_in_community_per_1000_under_5s_targeted_after_adjustments
        * hypothetical_cohort_size
        / 1000
    )
    cost_per_death_averted_in_untreated_population_before_accounting_for_leverage_and_funging = (
        cost_to_cover_cohort
        / expected_deaths_averted_in_untreated_population_after_adjustments
    )
    value_from_over_5_deaths_averted_per_dollar_wout_levfun = (
        value_of_averting_death_of_an_individual_5_or_older
        / cost_per_death_averted_in_untreated_population_before_accounting_for_leverage_and_funging
    )

    return {
        "SMC: value from over 5 deaths averted per dollar w/out lev/fun": value_from_over_5_deaths_averted_per_dollar_wout_levfun
    }


income_increase_ages_14_and_under_givewell = {
    "SMC: malaria prevalance young": 0.29,
    "SMC: malaria prevalence old": 0.29,
    "SMC: annual increase in income for eliminating prob of infection in youth": 0.023,
    "SMC: replicability adjustment for malaria vs income": 0.52,
    "SMC: num yrs between program and long term benefits": 10,
    "SMC: duration of long term benefits": 40,
    "SMC: multiplier for sharing w/in households": 2,
}


def income_increases_age_14_and_under(
    malaria_prevalance_young,
    malaria_prevalence_old,
    direct_mortality_in_high_transmission_season,
    reduction_in_malaria_after_adjustments,
    internal_validity_adjustment,
    external_validity_adjustment,
    reduction_in_untreated_pop_per_reduction_in_treated_pop,
    adjustment_for_higher_percent_covered_in_trial_than_ACCESS,
    annual_increase_in_income_for_eliminating_prob_of_infection_in_youth,
    replicability_adjustment_for_malaria_vs_income,
    num_yrs_between_program_and_long_term_benefits,
    discount_rate,
    duration_of_long_term_benefits,
    multiplier_for_sharing_win_households,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    cost_per_child_targeted,
):
    percentage_reduction_in_malaria_prevalance_in_treated_population_after_adherence_and_coverage_adjustments_young = (
        malaria_prevalance_young
        * direct_mortality_in_high_transmission_season
        * reduction_in_malaria_after_adjustments
        * internal_validity_adjustment
        * external_validity_adjustment
    )
    percentage_reduction_in_malaria_prevalance_in_untreated_population_after_adherence_and_coverage_adjustments_old = (
        malaria_prevalence_old
        * direct_mortality_in_high_transmission_season
        * reduction_in_malaria_after_adjustments
        * internal_validity_adjustment
        * external_validity_adjustment
        * reduction_in_untreated_pop_per_reduction_in_treated_pop
        * adjustment_for_higher_percent_covered_in_trial_than_ACCESS
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
            log(
                1 + annual_increase_in_income_for_eliminating_prob_of_infection_in_youth
            )
            - log(1)
        )
        * replicability_adjustment_for_malaria_vs_income
    )
    benefit_on_one_years_income = (
        increase_in_ln_income_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14
        / (1 + discount_rate) ** num_yrs_between_program_and_long_term_benefits
    )
    present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14 = present_value_of_annuity(
        discount_rate, duration_of_long_term_benefits, benefit_on_one_years_income
    )  # TODO: end of period
    present_value_of_benefits_from_reducing_point_in_time_probability_of_malaria_infection_from_100_to_0_for_individual_for_one_year_between_ages_of_0_and_14 = (
        present_value_of_lifetime_benefits_from_reducing_prevalence_from_1_to_0_for_an_individual_for_one_year_between_ages_of_0_and_14
        * multiplier_for_sharing_win_households
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
    value_from_development_per_dollar_wout_levfun = (
        value_from_development_benefits_per_1000_under_5s_targeted
        / (cost_per_child_targeted * 1000)
    )
    return {
        "SMC: value from development per dollar w/out lev/fun": value_from_development_per_dollar_wout_levfun
    }


results_givewell = {"SMC: expected cost from lev/fun": 0.1074}


def results(
    value_from_development_per_dollar_wout_levfun,
    value_from_under_5_deaths_averted_per_dollar_wout_levfun,
    value_from_over_5_deaths_averted_per_dollar_wout_levfun,
    expected_cost_from_levfun,
):
    value_per_dollar_before_accounting_for_leverage_and_funging = (
        value_from_development_per_dollar_wout_levfun
        + value_from_under_5_deaths_averted_per_dollar_wout_levfun
        + value_from_over_5_deaths_averted_per_dollar_wout_levfun
    )
    value_per_dollar_w_levfun = (
        value_per_dollar_before_accounting_for_leverage_and_funging
        * (1 - expected_cost_from_levfun)
    )
    return {"SMC: value per dollar w/ lev/fun": value_per_dollar_w_levfun}
