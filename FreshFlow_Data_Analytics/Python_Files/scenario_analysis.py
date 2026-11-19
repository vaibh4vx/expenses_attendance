
import numpy as np
from pathlib import Path


# PHASE 9 - RELATIONSHIPS AND SCENARIOS


# Loading the cleaned dataset

project_folder = Path(__file__).resolve().parents[1]

dataset_path = project_folder / "Dataset" / "cleaned_sales.csv"

report_folder = project_folder / "Reports"

report_folder.mkdir(exist_ok=True)

data = np.genfromtxt(
    dataset_path,
    delimiter=",",
    skip_header=1,
    ndmin=2
)

print("Dataset Shape:", data.shape)


# 9.1 - Extracting Required Columns

transaction_id = data[:, 0]
quantity = data[:, 5]
unit_price = data[:, 6]
unit_cost = data[:, 7]
discount_rate = data[:, 8]


# 9.2 - Validating and Aligning Observations

valid_records = (
    np.isfinite(transaction_id) &
    (transaction_id > 0) &
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity)) &
    np.isfinite(unit_price) &
    (unit_price > 0) &
    np.isfinite(unit_cost) &
    (unit_cost > 0) &
    np.isfinite(discount_rate) &
    (discount_rate >= 0) &
    (discount_rate <= 1)
)

valid_quantity = quantity[valid_records]
valid_price = unit_price[valid_records]
valid_cost = unit_cost[valid_records]
valid_discount = discount_rate[valid_records]

print("\n========== VALID OBSERVATIONS ==========")

print("Total Transactions:", len(data))
print("Valid Observations:", np.sum(valid_records))
print("Invalid Observations:", np.sum(~valid_records))


# Calculating financial metrics for valid records

gross_sales = valid_quantity * valid_price

actual_discount_amount = (
    gross_sales * valid_discount
)

actual_net_revenue = (
    gross_sales - actual_discount_amount
)

cogs = (
    valid_quantity * valid_cost
)

actual_gross_profit = (
    actual_net_revenue - cogs
)

print("\nAligned Array Shapes:")
print("Discount:", valid_discount.shape)
print("Revenue:", actual_net_revenue.shape)
print("Profit:", actual_gross_profit.shape)


# 9.3 - CORRELATION ANALYSIS USING np.corrcoef()

print("\n========== CORRELATION ANALYSIS ==========")

relationship_data = np.vstack((
    valid_discount,
    actual_net_revenue,
    actual_gross_profit
))

correlation_matrix = np.corrcoef(relationship_data)

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nDiscount and Net Revenue Correlation:",
      correlation_matrix[0, 1])

print("Discount and Gross Profit Correlation:",
      correlation_matrix[0, 2])

print("Net Revenue and Gross Profit Correlation:",
      correlation_matrix[1, 2])


# 9.4 - COVARIANCE ANALYSIS USING np.cov()

print("\n========== COVARIANCE ANALYSIS ==========")

covariance_matrix = np.cov(relationship_data)

print("\nCovariance Matrix:")
print(covariance_matrix)

print("\nDiscount and Net Revenue Covariance:",
      covariance_matrix[0, 1])

print("Discount and Gross Profit Covariance:",
      covariance_matrix[0, 2])

print("Net Revenue and Gross Profit Covariance:",
      covariance_matrix[1, 2])


# 9.5 - DEFINING ALTERNATIVE DISCOUNT SCENARIOS

print("\n========== DISCOUNT SCENARIOS ==========")

scenario_rates = np.array([
    0.10,
    0.15,
    0.20
])

scenario_rates = np.clip(scenario_rates, 0, 1)

print("Alternative Discount Rates:")
print(scenario_rates * 100)


# 9.6 - MODELLING SCENARIOS USING BROADCASTING

scenario_rates_2d = scenario_rates[:, np.newaxis]

gross_sales_2d = gross_sales[np.newaxis, :]

scenario_discount_amount = (
    scenario_rates_2d * gross_sales_2d
)

scenario_net_revenue = (
    gross_sales_2d - scenario_discount_amount
)

cogs_2d = cogs[np.newaxis, :]

scenario_gross_profit = (
    scenario_net_revenue - cogs_2d
)

print("\nBroadcasting Shapes:")
print("Scenario Rates:", scenario_rates_2d.shape)
print("Gross Sales:", gross_sales_2d.shape)
print("Scenario Discount Amount:", scenario_discount_amount.shape)
print("Scenario Net Revenue:", scenario_net_revenue.shape)
print("Scenario Gross Profit:", scenario_gross_profit.shape)


# 9.7 - CALCULATING ACTUAL BASELINE METRICS

baseline_transactions = len(valid_discount)

baseline_quantity = np.sum(valid_quantity)

baseline_gross_sales = np.sum(gross_sales)

baseline_discount_amount = np.sum(actual_discount_amount)

baseline_net_revenue = np.sum(actual_net_revenue)

baseline_cogs = np.sum(cogs)

baseline_gross_profit = np.sum(actual_gross_profit)

baseline_margin = np.nan

if baseline_net_revenue != 0:
    baseline_margin = (
        baseline_gross_profit / baseline_net_revenue
    ) * 100

baseline_discount_rate = np.nan

if baseline_gross_sales != 0:
    baseline_discount_rate = (
        baseline_discount_amount / baseline_gross_sales
    ) * 100


# 9.8 - CALCULATING SCENARIO TOTALS

scenario_discount_totals = np.sum(
    scenario_discount_amount,
    axis=1
)

scenario_revenue_totals = np.sum(
    scenario_net_revenue,
    axis=1
)

scenario_profit_totals = np.sum(
    scenario_gross_profit,
    axis=1
)

scenario_cogs_totals = np.sum(
    np.broadcast_to(cogs_2d, scenario_gross_profit.shape),
    axis=1
)

scenario_margin = np.divide(
    scenario_profit_totals * 100,
    scenario_revenue_totals,
    out=np.full(len(scenario_rates), np.nan),
    where=scenario_revenue_totals != 0
)


# 9.9 - COMPARING SCENARIOS WITH ACTUAL BASELINE

revenue_change = (
    scenario_revenue_totals - baseline_net_revenue
)

profit_change = (
    scenario_profit_totals - baseline_gross_profit
)

revenue_change_percentage = np.divide(
    revenue_change * 100,
    baseline_net_revenue,
    out=np.full(len(scenario_rates), np.nan),
    where=baseline_net_revenue != 0
)

profit_change_percentage = np.divide(
    profit_change * 100,
    baseline_gross_profit,
    out=np.full(len(scenario_rates), np.nan),
    where=baseline_gross_profit != 0
)


# 9.10 - DISPLAYING SCENARIO COMPARISON

scenario_profit_status = np.where(
    scenario_profit_totals >= 0,
    "Positive",
    "Negative"
)

print("\n========== ACTUAL BASELINE ==========")

print("Valid Transactions:", baseline_transactions)
print("Quantity Sold:", baseline_quantity)
print("Gross Sales:", round(baseline_gross_sales, 2))
print("Discount Amount:", round(baseline_discount_amount, 2))
print("Effective Discount Rate:", round(baseline_discount_rate, 2), "%")
print("Net Revenue:", round(baseline_net_revenue, 2))
print("COGS:", round(baseline_cogs, 2))
print("Gross Profit:", round(baseline_gross_profit, 2))
print("Gross Margin:", round(baseline_margin, 2), "%")

print("\n========== ALTERNATIVE SCENARIOS ==========")

for i in range(len(scenario_rates)):

    print("\nScenario:", int(scenario_rates[i] * 100), "% Discount")

    print("Discount Amount:",
          round(scenario_discount_totals[i], 2))

    print("Net Revenue:",
          round(scenario_revenue_totals[i], 2))

    print("COGS:",
          round(scenario_cogs_totals[i], 2))

    print("Gross Profit:",
          round(scenario_profit_totals[i], 2))

    print("Profit Status:", scenario_profit_status[i])

    print("Gross Margin:",
          round(scenario_margin[i], 2), "%")

    print("Revenue Change:",
          round(revenue_change[i], 2))

    print("Revenue Change Percentage:",
          round(revenue_change_percentage[i], 2), "%")

    print("Profit Change:",
          round(profit_change[i], 2))

    print("Profit Change Percentage:",
          round(profit_change_percentage[i], 2), "%")


# 9.11 - CREATING SCENARIO ANALYSIS REPORT

scenario_names = np.array([
    "Actual Baseline",
    "10% Discount",
    "15% Discount",
    "20% Discount"
])

discount_rates_report = np.append(
    baseline_discount_rate,
    scenario_rates * 100
)

transaction_report = np.full(
    len(scenario_rates) + 1,
    baseline_transactions
)

quantity_report = np.full(
    len(scenario_rates) + 1,
    baseline_quantity
)

gross_sales_report = np.full(
    len(scenario_rates) + 1,
    baseline_gross_sales
)

discount_amount_report = np.concatenate((
    [baseline_discount_amount],
    scenario_discount_totals
))

revenue_report = np.concatenate((
    [baseline_net_revenue],
    scenario_revenue_totals
))

cogs_report = np.concatenate((
    [baseline_cogs],
    scenario_cogs_totals
))

profit_report = np.concatenate((
    [baseline_gross_profit],
    scenario_profit_totals
))

margin_report = np.concatenate((
    [baseline_margin],
    scenario_margin
))

revenue_change_report = np.concatenate((
    [0],
    revenue_change
))

revenue_change_percentage_report = np.concatenate((
    [0],
    revenue_change_percentage
))

profit_change_report = np.concatenate((
    [0],
    profit_change
))

profit_change_percentage_report = np.concatenate((
    [0],
    profit_change_percentage
))

scenario_report = np.column_stack((
    scenario_names,
    discount_rates_report,
    transaction_report,
    quantity_report,
    gross_sales_report,
    discount_amount_report,
    revenue_report,
    cogs_report,
    profit_report,
    margin_report,
    revenue_change_report,
    revenue_change_percentage_report,
    profit_change_report,
    profit_change_percentage_report
))

column_names = (
    "scenario,"
    "discount_rate_percentage,"
    "transaction_count,"
    "quantity_sold,"
    "gross_sales,"
    "discount_amount,"
    "net_revenue,"
    "cogs,"
    "gross_profit,"
    "gross_margin_percentage,"
    "revenue_change,"
    "revenue_change_percentage,"
    "profit_change,"
    "profit_change_percentage"
)

scenario_report_path = report_folder / "scenario_analysis.csv"

np.savetxt(
    scenario_report_path,
    scenario_report,
    delimiter=",",
    header=column_names,
    comments="",
    fmt="%s"
)

print("\nScenario Analysis Report Saved Successfully!")
print("File Location:", scenario_report_path)


# 9.12 - SAVING RELATIONSHIP ANALYSIS REPORT

relationship_report = np.array([
    [
        "Correlation",
        "Discount",
        correlation_matrix[0, 0],
        correlation_matrix[0, 1],
        correlation_matrix[0, 2]
    ],
    [
        "Correlation",
        "Net Revenue",
        correlation_matrix[1, 0],
        correlation_matrix[1, 1],
        correlation_matrix[1, 2]
    ],
    [
        "Correlation",
        "Gross Profit",
        correlation_matrix[2, 0],
        correlation_matrix[2, 1],
        correlation_matrix[2, 2]
    ],
    [
        "Covariance",
        "Discount",
        covariance_matrix[0, 0],
        covariance_matrix[0, 1],
        covariance_matrix[0, 2]
    ],
    [
        "Covariance",
        "Net Revenue",
        covariance_matrix[1, 0],
        covariance_matrix[1, 1],
        covariance_matrix[1, 2]
    ],
    [
        "Covariance",
        "Gross Profit",
        covariance_matrix[2, 0],
        covariance_matrix[2, 1],
        covariance_matrix[2, 2]
    ]
], dtype=object)

relationship_report_path = report_folder / "relationship_analysis.csv"

np.savetxt(
    relationship_report_path,
    relationship_report,
    delimiter=",",
    header="analysis_type,variable,discount,net_revenue,gross_profit",
    comments="",
    fmt="%s"
)

print("\nRelationship Analysis Report Saved Successfully!")
print("File Location:", relationship_report_path)


# 9.13 - ASSUMPTIONS AND LIMITATIONS

print("\n========== ASSUMPTIONS ==========")

print("1. Quantity remains constant in every scenario.")

print("2. Unit selling price remains constant.")

print("3. Unit cost remains constant.")

print("4. Only the discount rate changes.")

print("5. All scenarios use the same valid transactions.")

print("6. Scenarios are hypothetical and are not forecasts.")


print("\n========== CONFOUNDING FACTORS ==========")

print("1. Product and outlet differences may affect revenue and profit.")

print("2. Product prices and unit costs may influence profitability.")

print("3. Customer demand and promotional activities may influence discounts.")

print("4. These factors may affect the observed relationship between discount and performance.")


print("\n========== WHAT THE DATA CANNOT PROVE ==========")

print("1. Correlation does not prove that discounts cause changes in revenue or profit.")

print("2. The analysis cannot prove how customers will react to new discount rates.")

print("3. Constant quantity is an assumption, not a measured demand response.")

print("4. The scenarios cannot guarantee future business performance.")


print("\n==========================================")
print("PHASE 9 COMPLETED")
