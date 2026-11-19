
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

# 7.1 - Extracting Required Columns

day_index = data[:, 1]
outlet = data[:, 2]
product = data[:, 3]
category = data[:, 4]
quantity = data[:, 5]
unit_price = data[:, 6]
unit_cost = data[:, 7]
discount_rate = data[:, 8]

# 7.2 - Validating Daily Records

valid_day = (
    np.isfinite(day_index) &
    (day_index >= 1) &
    (day_index <= 30) &
    (day_index == np.floor(day_index))
)

valid_outlet_code = (
    np.isfinite(outlet) &
    np.isin(outlet, [1, 2, 3]) &
    (outlet == np.floor(outlet))
)

valid_product_code = (
    np.isfinite(product) &
    np.isin(product, [1, 2, 3, 4, 5, 6, 7, 8]) &
    (product == np.floor(product))
)

valid_category_code = (
    np.isfinite(category) &
    np.isin(category, [1, 2, 3, 4]) &
    (category == np.floor(category))
)

valid_group_records = (
    valid_outlet_code &
    valid_product_code &
    valid_category_code
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

valid_discount = (
    np.isfinite(discount_rate) &
    (discount_rate >= 0) &
    (discount_rate <= 1)
)

valid_daily_quantity_records = (
    valid_day &
    valid_group_records &
    valid_quantity
)

valid_daily_revenue_records = (
    valid_daily_quantity_records &
    valid_price &
    valid_discount
)

# 7.3 - Calculating Net Revenue

net_revenue = quantity * unit_price * (1 - discount_rate)

valid_daily_profit_records = (
    valid_daily_revenue_records &
    np.isfinite(unit_cost) &
    (unit_cost > 0)
)

gross_profit = (
    quantity * (unit_price * (1 - discount_rate) - unit_cost)
)

# 7.4 - Creating Daily Totals

days = np.arange(1, 31)

daily_record_count = np.bincount(
    day_index[valid_daily_quantity_records].astype(int),
    minlength=31
)[1:31]

daily_units_sold = np.bincount(
    day_index[valid_daily_quantity_records].astype(int),
    weights=quantity[valid_daily_quantity_records],
    minlength=31
)[1:31]

daily_net_revenue = np.bincount(
    day_index[valid_daily_revenue_records].astype(int),
    weights=net_revenue[valid_daily_revenue_records],
    minlength=31
)[1:31]

daily_gross_profit = np.bincount(
    day_index[valid_daily_profit_records].astype(int),
    weights=gross_profit[valid_daily_profit_records],
    minlength=31
)[1:31]

print("\nDaily Units Sold:")
print(daily_units_sold)

print("\nDaily Net Revenue:")
print(daily_net_revenue)

print("\nDaily Gross Profit:")
print(daily_gross_profit)

# 7.5 - Calculating Daily Revenue Changes

daily_revenue_change = np.full(30, np.nan)

daily_revenue_change[1:] = np.diff(daily_net_revenue)

print("\nDaily Revenue Changes:")
print(daily_revenue_change)

# 7.6 - Calculating Percentage Changes

daily_revenue_percentage_change = np.full(30, np.nan)

previous_day_revenue = daily_net_revenue[:-1]
current_day_revenue = daily_net_revenue[1:]

daily_revenue_percentage_change[1:] = np.divide(
    (current_day_revenue - previous_day_revenue) * 100,
    previous_day_revenue,
    out=np.full(29, np.nan),
    where=previous_day_revenue != 0
)

print("\nDaily Revenue Percentage Changes:")
print(daily_revenue_percentage_change)

# 7.7 - Calculating Cumulative Revenue

cumulative_revenue = np.cumsum(daily_net_revenue)

print("\nCumulative Revenue:")
print(cumulative_revenue)

# 7.8 - Calculating 7-Day Moving Average

window_size = 7

moving_average_revenue = np.full(30, np.nan)

cumulative_for_rolling = np.cumsum(daily_net_revenue)

moving_average_revenue[window_size - 1:] = (
    cumulative_for_rolling[window_size - 1:]
    - np.concatenate((
        [0],
        cumulative_for_rolling[:-window_size]
    ))
) / window_size

print("\n7-Day Moving Average Revenue:")
print(moving_average_revenue)

# 7.9 - Identifying Unusual Revenue Changes

unusual_change_threshold = 20

unusual_days = (
    np.isfinite(daily_revenue_percentage_change) &
    (np.abs(daily_revenue_percentage_change) >= unusual_change_threshold)
)

print("\nUnusual Revenue Changes:")

if np.any(unusual_days):
    for i in range(30):
        if unusual_days[i]:
            print(
                "Day", days[i],
                "Revenue Change:",
                round(daily_revenue_percentage_change[i], 2),
                "%"
            )
else:
    print("No unusual revenue changes found.")

# 7.10 - Creating Daily Trend Table

daily_trend_table = np.column_stack((
    days,
    daily_record_count,
    daily_units_sold,
    daily_net_revenue,
    daily_gross_profit,
    daily_revenue_change,
    daily_revenue_percentage_change,
    cumulative_revenue,
    moving_average_revenue
))

column_names = (
    "day_index,"
    "transaction_count,"
    "units_sold,"
    "net_revenue,"
    "gross_profit,"
    "revenue_change,"
    "revenue_percentage_change,"
    "cumulative_revenue,"
    "7_day_moving_average"
)

# 7.11 - Saving Daily Trend Report

report_path = report_folder / "daily_trend.csv"

np.savetxt(
    report_path,
    daily_trend_table,
    delimiter=",",
    header=column_names,
    comments="",
    fmt="%.4f"
)

print("\nDaily Trend Table:")
print(daily_trend_table)

print("\nTotal Units Sold:", np.sum(daily_units_sold))
print("Total Net Revenue:", np.sum(daily_net_revenue))
print("Total Gross Profit:", np.sum(daily_gross_profit))

print("\nDaily Trend Report Saved Successfully!")
print("File Location:", report_path)
