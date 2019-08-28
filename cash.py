#!/usr/bin/env python3

from pymc3.math import log

from utility import present_value_of_annuity

cash_transfers_givewell = {
    "Cash: average household size": 4.7,
    "Cash: percent of transfers invested": 0.39,
    "Cash: return on investment": 0.1,
    "Cash: baseline consumption per capita": 285.92,
    "Cash: duration of investment benefits": 15,
    "Cash: percent of investment returned when benefits end": 0.2,
    "Cash: discount from negative spillover": 0.05,
    "Cash: transfer as percent of total cost": 0.83,
}


def cash_transfers(
    average_household_size,
    percent_of_transfers_invested,
    return_on_investment,
    baseline_consumption_per_capita,
    discount_rate,
    duration_of_investment_benefits,
    percent_of_investment_returned_when_benefits_end,
    discount_from_negative_spillover,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    transfer_as_percent_of_total_cost,
):

    total_size_of_transfer = 1000

    size_of_transfer_per_person = total_size_of_transfer / average_household_size
    amount_invested = size_of_transfer_per_person * percent_of_transfers_invested
    total_increase_in_consumption_due_to_funds_transferred = (
        1 - percent_of_transfers_invested
    ) * size_of_transfer_per_person
    annual_increase_in_consumption_due_to_investment_returns = (
        amount_invested * return_on_investment
    )
    total_increase_in_ln_consumption_due_to_funds_transferred = log(
        baseline_consumption_per_capita
        + total_increase_in_consumption_due_to_funds_transferred
    ) - log(baseline_consumption_per_capita)
    future_annual_increase_in_ln_consumption_from_return_on_investments = log(
        baseline_consumption_per_capita
        + annual_increase_in_consumption_due_to_investment_returns
    ) - log(baseline_consumption_per_capita)
    present_value_of_future_increases_in_ln_consumption_excluding_final_year = present_value_of_annuity(
        discount_rate,
        duration_of_investment_benefits - 1,
        future_annual_increase_in_ln_consumption_from_return_on_investments,
    )
    _additional_consumption_in_final_year = (
        amount_invested * return_on_investment
        + amount_invested * percent_of_investment_returned_when_benefits_end
    )
    present_value_of_ln_consumption_increase_in_final_year = (
        log(baseline_consumption_per_capita + _additional_consumption_in_final_year)
        - log(baseline_consumption_per_capita)
    ) / (1 + discount_rate) ** duration_of_investment_benefits
    present_value_of_all_future_increases_in_ln_consumption = (
        present_value_of_ln_consumption_increase_in_final_year
        + present_value_of_future_increases_in_ln_consumption_excluding_final_year
    )
    total_present_value_of_cash_transfer = (
        present_value_of_all_future_increases_in_ln_consumption
        + total_increase_in_ln_consumption_due_to_funds_transferred
    )
    total_present_value_of_cash_transfer_accounting_for_spillovers = (
        1 - discount_from_negative_spillover
    ) * total_present_value_of_cash_transfer
    increase_in_ln_consumption_for_each_dollar_donated = (
        total_present_value_of_cash_transfer_accounting_for_spillovers
        * transfer_as_percent_of_total_cost
    ) / size_of_transfer_per_person
    value_per_dollar = (
        increase_in_ln_consumption_for_each_dollar_donated
        * value_of_increasing_ln_consumption_per_capita_per_annum
    )
    return {"Cash: value per dollar": value_per_dollar}
