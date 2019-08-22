#!/usr/bin/env python3

import math
from utility import present_value_of_annuity

staff_aggregates = {
    "average_household_size": 4.7,
    "percentage_of_transfers_invested": 0.39,
    "return_on_investment": 0.1,
    "baseline_annual_consumption_per_capita": 285.92,
    "duration_of_investment_benefts": 15,
    "percent_of_investment_returned_when_benefits_end": 0.2,
    "discount_from_negative_spillover": 0.05,
    "transfer_as_percentage_of_total_cost": 0.83,
}


def cash_transfers(
    average_household_size,
    percentage_of_transfers_invested,
    return_on_investment,
    baseline_annual_consumption_per_capita,
    discount_rate,
    duration_of_investment_benefits,
    percent_of_investment_returned_when_benefits_end,
    discount_from_negative_spillover,
    value_of_increasing_ln_consumption_per_capita_per_annum,
    transfer_as_percentage_of_total_cost,
):

    total_size_of_transfer = 1000

    size_of_transfer_per_person = total_size_of_transfer / average_household_size
    print(size_of_transfer_per_person)
    amount_invested = size_of_transfer_per_person * percentage_of_transfers_invested
    print(amount_invested)
    total_increase_in_consumption_due_to_funds_transferred = (
        1 - percentage_of_transfers_invested
    ) * size_of_transfer_per_person
    print(total_increase_in_consumption_due_to_funds_transferred)
    annual_increase_in_consumption_due_to_investment_returns = (
        amount_invested * return_on_investment
    )
    print(annual_increase_in_consumption_due_to_investment_returns)
    total_increase_in_ln_consumption_due_to_funds_transferred = math.log(
        baseline_annual_consumption_per_capita
        + total_increase_in_consumption_due_to_funds_transferred
    ) - math.log(baseline_annual_consumption_per_capita)
    print(total_increase_in_ln_consumption_due_to_funds_transferred)
    future_annual_increase_in_ln_consumption_from_return_on_investments = math.log(
        baseline_annual_consumption_per_capita
        + annual_increase_in_consumption_due_to_investment_returns
    ) - math.log(baseline_annual_consumption_per_capita)
    print(future_annual_increase_in_ln_consumption_from_return_on_investments)
    present_value_of_future_increases_in_ln_consumption_excluding_final_year = present_value_of_annuity(
        discount_rate,
        duration_of_investment_benefits - 1,
        future_annual_increase_in_ln_consumption_from_return_on_investments,
    )
    print(present_value_of_future_increases_in_ln_consumption_excluding_final_year)
    _additional_consumption_in_final_year = (
        amount_invested * return_on_investment
        + amount_invested * percent_of_investment_returned_when_benefits_end
    )
    present_value_of_ln_consumption_increase_in_final_year = (
        math.log(
            baseline_annual_consumption_per_capita
            + _additional_consumption_in_final_year
        )
        - math.log(baseline_annual_consumption_per_capita)
    ) / (1 + discount_rate) ** duration_of_investment_benefits
    print(present_value_of_ln_consumption_increase_in_final_year)
    present_value_of_all_future_increases_in_ln_consumption = (
        present_value_of_ln_consumption_increase_in_final_year
        + present_value_of_future_increases_in_ln_consumption_excluding_final_year
    )
    print(present_value_of_all_future_increases_in_ln_consumption)
    total_present_value_of_cash_transfer = (
        present_value_of_all_future_increases_in_ln_consumption
        + total_increase_in_ln_consumption_due_to_funds_transferred
    )
    print(total_present_value_of_cash_transfer)
    total_present_value_of_cash_transfer_accounting_for_spillovers = (
        1 - discount_from_negative_spillover
    ) * total_present_value_of_cash_transfer
    print(total_present_value_of_cash_transfer_accounting_for_spillovers)
    increase_in_ln_consumption_for_each_dollar_donated = (
        total_present_value_of_cash_transfer * transfer_as_percentage_of_total_cost
    ) / size_of_transfer_per_person
    print(increase_in_ln_consumption_for_each_dollar_donated)
    value_per_dollar_donated = (
        increase_in_ln_consumption_for_each_dollar_donated
        * value_of_increasing_ln_consumption_per_capita_per_annum
    )
    print(value_per_dollar_donated)
    return value_per_dollar_donated
