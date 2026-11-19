
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
    dtype=str,
    ndmin=2
)

print("Dataset Shape:", data.shape)

# 8.1 - Extracting Required Columns

transaction_id = data[:, 0]
day_index = data[:, 1].astype(float)
outlet = data[:, 2].astype(float)
product = data[:, 3].astype(float)
category = data[:, 4].astype(float)

quantity = data[:, 5].astype(float)
unit_price = data[:, 6].astype(float)
unit_cost = data[:, 7].astype(float)
discount_rate = data[:, 8].astype(float)

# 8.2 - Validating Records

valid_day = (
    np.isfinite(day_index) &
    (day_index >= 1) &
    (day_index <= 30) &
    (day_index == np.floor(day_index))
)

valid_outlet = (
    np.isfinite(outlet) &
    np.isin(outlet, [1, 2, 3]) &
    (outlet == np.floor(outlet))
)

valid_product = (
    np.isfinite(product) &
    np.isin(product, [1, 2, 3, 4, 5, 6, 7, 8]) &
    (product == np.floor(product))
)

valid_category = (
    np.isfinite(category) &
    np.isin(category, [1, 2, 3, 4]) &
    (category == np.floor(category))
)

valid_quantity = (
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

valid_price = (
    np.isfinite(unit_price) &
    (unit_price > 0)
)

valid_cost = (
    np.isfinite(unit_cost) &
    (unit_cost > 0)
)

valid_discount = (
    np.isfinite(discount_rate) &
    (discount_rate >= 0) &
    (discount_rate <= 1)
)

valid_records = (
    valid_day &
    valid_outlet &
    valid_product &
    valid_category &
    valid_quantity &
    valid_price &
    valid_cost &
    valid_discount
)

print("\nValid Records:", np.sum(valid_records))
print("Invalid Records:", len(data) - np.sum(valid_records))

# 8.3 - Calculating Financial Metrics

net_unit_price = unit_price * (1 - discount_rate)

revenue = quantity * net_unit_price

gross_profit = quantity * (net_unit_price - unit_cost)

discount_amount = quantity * unit_price * discount_rate

# 8.4 - Calculating IQR Limits

valid_price_values = unit_price[valid_records]
valid_discount_values = discount_rate[valid_records]
valid_quantity_values = quantity[valid_records]
valid_cost_values = unit_cost[valid_records]

q1_price = np.percentile(valid_price_values, 25)
q3_price = np.percentile(valid_price_values, 75)
iqr_price = q3_price - q1_price

q1_discount = np.percentile(valid_discount_values, 25)
q3_discount = np.percentile(valid_discount_values, 75)
iqr_discount = q3_discount - q1_discount

q1_quantity = np.percentile(valid_quantity_values, 25)
q3_quantity = np.percentile(valid_quantity_values, 75)
iqr_quantity = q3_quantity - q1_quantity

q1_cost = np.percentile(valid_cost_values, 25)
q3_cost = np.percentile(valid_cost_values, 75)
iqr_cost = q3_cost - q1_cost

price_upper = q3_price + 1.5 * iqr_price
discount_upper = q3_discount + 1.5 * iqr_discount

quantity_lower = q1_quantity - 1.5 * iqr_quantity
quantity_upper = q3_quantity + 1.5 * iqr_quantity

cost_upper = q3_cost + 1.5 * iqr_cost

# 8.5 - Applying IQR Anomaly Flags

iqr_high_price = valid_records & (unit_price > price_upper)

iqr_high_discount = valid_records & (discount_rate > discount_upper)

iqr_unusual_quantity = valid_records & (
    (quantity < quantity_lower) |
    (quantity > quantity_upper)
)

iqr_high_cost = valid_records & (unit_cost > cost_upper)

print("\nIQR Limits:")
print("Price Upper Limit:", price_upper)
print("Discount Upper Limit:", discount_upper)
print("Quantity Lower Limit:", quantity_lower)
print("Quantity Upper Limit:", quantity_upper)
print("Unit Cost Upper Limit:", cost_upper)

print("\nIQR Anomaly Counts:")
print("High Price:", np.sum(iqr_high_price))
print("High Discount:", np.sum(iqr_high_discount))
print("Unusual Quantity:", np.sum(iqr_unusual_quantity))
print("High Unit Cost:", np.sum(iqr_high_cost))

# 8.6 - Calculating Z-Scores

price_mean = np.mean(valid_price_values)
price_std = np.std(valid_price_values)

discount_mean = np.mean(valid_discount_values)
discount_std = np.std(valid_discount_values)

quantity_mean = np.mean(valid_quantity_values)
quantity_std = np.std(valid_quantity_values)

cost_mean = np.mean(valid_cost_values)
cost_std = np.std(valid_cost_values)

price_z_score = np.divide(
    unit_price - price_mean,
    price_std,
    out=np.zeros(len(data)),
    where=price_std != 0
)

discount_z_score = np.divide(
    discount_rate - discount_mean,
    discount_std,
    out=np.zeros(len(data)),
    where=discount_std != 0
)

quantity_z_score = np.divide(
    quantity - quantity_mean,
    quantity_std,
    out=np.zeros(len(data)),
    where=quantity_std != 0
)

cost_z_score = np.divide(
    unit_cost - cost_mean,
    cost_std,
    out=np.zeros(len(data)),
    where=cost_std != 0
)

# 8.7 - Applying Z-Score Anomaly Flags

z_high_price = valid_records & (price_z_score > 3)

z_high_discount = valid_records & (discount_z_score > 3)

z_unusual_quantity = valid_records & (
    np.abs(quantity_z_score) > 3
)

z_high_cost = valid_records & (cost_z_score > 3)

print("\nZ-Score Anomaly Counts:")
print("High Price:", np.sum(z_high_price))
print("High Discount:", np.sum(z_high_discount))
print("Unusual Quantity:", np.sum(z_unusual_quantity))
print("High Unit Cost:", np.sum(z_high_cost))

# 8.8 - Combining Anomaly Masks

high_price = iqr_high_price | z_high_price

high_discount = iqr_high_discount | z_high_discount

unusual_quantity = iqr_unusual_quantity | z_unusual_quantity

high_unit_cost = iqr_high_cost | z_high_cost

negative_gross_profit = valid_records & (gross_profit < 0)

combined_anomalies = (
    high_price |
    high_discount |
    unusual_quantity |
    high_unit_cost |
    negative_gross_profit
)

anomaly_count = (
    high_price.astype(int) +
    high_discount.astype(int) +
    unusual_quantity.astype(int) +
    high_unit_cost.astype(int) +
    negative_gross_profit.astype(int)
)

print("\nCombined Anomaly Counts:")
print("High Price:", np.sum(high_price))
print("High Discount:", np.sum(high_discount))
print("Unusual Quantity:", np.sum(unusual_quantity))
print("High Unit Cost:", np.sum(high_unit_cost))
print("Negative Gross Profit:", np.sum(negative_gross_profit))
print("Total Flagged Records:", np.sum(combined_anomalies))

# 8.9 - Investigating Flagged Transactions

flagged_indices = np.where(combined_anomalies)[0]

print("\nFlagged Transaction Details:")

for index in flagged_indices:
    print(
        "Transaction:", transaction_id[index],
        "| Day:", day_index[index],
        "| Outlet:", outlet[index],
        "| Product:", product[index],
        "| Price:", unit_price[index],
        "| Discount:", discount_rate[index],
        "| Quantity:", quantity[index],
        "| Gross Profit:", gross_profit[index],
        "| Anomaly Count:", anomaly_count[index]
    )

# 8.10 - Estimating Possible Financial Exposure

discount_exposure = np.where(
    high_discount,
    discount_amount,
    0
)

negative_profit_exposure = np.where(
    negative_gross_profit,
    np.abs(gross_profit),
    0
)

possible_financial_exposure = np.maximum(
    discount_exposure,
    negative_profit_exposure
)

# 8.11 - Creating Prioritised Exception Register

exception_register = np.column_stack((
    transaction_id[combined_anomalies],
    day_index[combined_anomalies],
    outlet[combined_anomalies],
    product[combined_anomalies],
    category[combined_anomalies],
    quantity[combined_anomalies],
    unit_price[combined_anomalies],
    unit_cost[combined_anomalies],
    discount_rate[combined_anomalies],
    revenue[combined_anomalies],
    gross_profit[combined_anomalies],
    high_price[combined_anomalies].astype(int),
    high_discount[combined_anomalies].astype(int),
    unusual_quantity[combined_anomalies].astype(int),
    high_unit_cost[combined_anomalies].astype(int),
    negative_gross_profit[combined_anomalies].astype(int),
    anomaly_count[combined_anomalies],
    discount_exposure[combined_anomalies],
    negative_profit_exposure[combined_anomalies],
    possible_financial_exposure[combined_anomalies]
))

if len(exception_register) > 0:
    priority_order = np.argsort(
        exception_register[:, -1].astype(float)
    )[::-1]

    exception_register = exception_register[priority_order]

print("\nPrioritised Exception Register:")
print(exception_register)

print("\nTotal Flagged Transactions:", len(exception_register))

print(
    "Total Discount Exposure:",
    np.sum(discount_exposure[combined_anomalies])
)

print(
    "Total Negative Profit Exposure:",
    np.sum(negative_profit_exposure[combined_anomalies])
)

print(
    "Total Possible Financial Exposure:",
    np.sum(possible_financial_exposure[combined_anomalies])
)

print(
    "\nNote: These amounts are review indicators, "
    "not confirmed financial losses."
)

# 8.12 - Investigating Stock Anomalies

print("\nStock Anomalies:")
print(
    "Not assessed because the companion operations file "
    "was not available. Transaction data alone cannot "
    "establish stock-flow or waste anomalies."
)

# 8.13 - Saving Exception Register

column_names = (
    "Transaction_ID,Day_Index,Outlet,Product,Category,"
    "Quantity,Unit_Price,Unit_Cost,Discount_Rate,"
    "Revenue,Gross_Profit,High_Price,High_Discount,"
    "Unusual_Quantity,High_Unit_Cost,Negative_Gross_Profit,"
    "Anomaly_Count,Discount_Exposure,Negative_Profit_Exposure,"
    "Possible_Financial_Exposure"
)

output_file = report_folder / "exception_register.csv"

np.savetxt(
    output_file,
    exception_register,
    delimiter=",",
    fmt="%s",
    header=column_names,
    comments=""
)

print("\nException Register Saved Successfully!")
print("File Location:", output_file)
