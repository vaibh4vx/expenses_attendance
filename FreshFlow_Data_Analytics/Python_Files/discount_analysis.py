
import numpy as np
from pathlib import Path

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

# H.1 - Extracting Required Columns

transaction_id = data[:, 0]
quantity = data[:, 5]
unit_price = data[:, 6]
unit_cost = data[:, 7]
discount_rate = data[:, 8]

# H.2 - Validating Discount Records

valid_transaction = (
    np.isfinite(transaction_id) &
    (transaction_id > 0)
)

valid_quantity = (
    np.isfinite(quantity) &
    (quantity > 0)
)

valid_price = (
    np.isfinite(unit_price) &
    (unit_price > 0)
)

valid_discount = (
    np.isfinite(discount_rate) &
    (discount_rate >= 0) &
    (discount_rate <= 1)
)

valid_cost = (
    np.isfinite(unit_cost) &
    (unit_cost > 0)
)

valid_discount_records = (
    valid_transaction &
    valid_quantity &
    valid_price &
    valid_discount
)

valid_profit_records = (
    valid_discount_records &
    valid_cost
)

print("\nValid Discount Records:", np.sum(valid_discount_records))
print("Valid Profit Records:", np.sum(valid_profit_records))

# H.3 - Calculating Financial Metrics

gross_sales = quantity * unit_price

discount_amount = (
    gross_sales * discount_rate
)

net_revenue = (
    gross_sales - discount_amount
)

gross_profit = (
    net_revenue - (quantity * unit_cost)
)

# H.4 - Creating Discount Bands

discount_band_names = [
    "No Discount",
    "0-5%",
    "5-10%",
    "10-15%",
    "15-20%",
    "Above 20%"
]

discount_band_masks = [
    valid_discount_records & (discount_rate == 0),

    valid_discount_records & (discount_rate > 0) & (discount_rate <= 0.05),

    valid_discount_records & (discount_rate > 0.05) & (discount_rate <= 0.10),

    valid_discount_records & (discount_rate > 0.10) & (discount_rate <= 0.15),

    valid_discount_records & (discount_rate > 0.15) & (discount_rate <= 0.20),

    valid_discount_records & (discount_rate > 0.20)
]

# H.5 - Calculating Transaction Counts

transaction_counts = np.zeros(len(discount_band_masks))

for i in range(len(discount_band_masks)):
    transaction_counts[i] = np.sum(discount_band_masks[i])

print("\nTransaction Counts:")
print(transaction_counts)

# H.6 - Calculating Quantity Sold

quantity_by_band = np.zeros(len(discount_band_masks))

for i in range(len(discount_band_masks)):
    quantity_by_band[i] = np.sum(
        quantity[discount_band_masks[i]]
    )

print("\nQuantity Sold by Discount Band:")
print(quantity_by_band)

# H.7 - Calculating Net Revenue

revenue_by_band = np.zeros(len(discount_band_masks))

for i in range(len(discount_band_masks)):
    revenue_by_band[i] = np.sum(
        net_revenue[discount_band_masks[i]]
    )

print("\nNet Revenue by Discount Band:")
print(revenue_by_band)

# H.8 - Calculating Gross Profit

profit_by_band = np.zeros(len(discount_band_masks))
profit_revenue_by_band = np.zeros(len(discount_band_masks))

for i in range(len(discount_band_masks)):
    profit_mask = (
        discount_band_masks[i] &
        valid_profit_records
    )

    profit_by_band[i] = np.sum(
        gross_profit[profit_mask]
    )

    profit_revenue_by_band[i] = np.sum(
        net_revenue[profit_mask]
    )

print("\nGross Profit by Discount Band:")
print(profit_by_band)

# H.9 - Calculating Gross Margin Percentage

gross_margin = np.divide(
    profit_by_band * 100,
    profit_revenue_by_band,
    out=np.zeros(len(discount_band_masks)),
    where=profit_revenue_by_band != 0
)

print("\nGross Margin Percentage:")
print(gross_margin)

# H.10 - Calculating Average Revenue per Transaction

average_revenue = np.divide(
    revenue_by_band,
    transaction_counts,
    out=np.zeros(len(discount_band_masks)),
    where=transaction_counts != 0
)

print("\nAverage Revenue per Transaction:")
print(average_revenue)

# H.11 - Calculating Overall Totals

total_transactions = np.sum(transaction_counts)
total_quantity = np.sum(quantity_by_band)
total_revenue = np.sum(revenue_by_band)
total_profit = np.sum(profit_by_band)

print("\nTotal Transactions:", total_transactions)
print("Total Quantity Sold:", total_quantity)
print("Total Net Revenue:", total_revenue)
print("Total Gross Profit:", total_profit)

# H.12 - Creating Discount Analysis Report

discount_report = np.column_stack((
    transaction_counts,
    quantity_by_band,
    revenue_by_band,
    profit_by_band,
    gross_margin,
    average_revenue
))

column_names = (
    "transaction_count,"
    "quantity_sold,"
    "net_revenue,"
    "gross_profit,"
    "gross_margin_percentage,"
    "average_revenue_per_transaction"
)

report_path = report_folder / "discount_analysis.csv"

np.savetxt(
    report_path,
    discount_report,
    delimiter=",",
    header=column_names,
    comments="",
    fmt="%.4f"
)

print("\nDiscount Analysis Report Saved Successfully!")
print("File Location:", report_path)



