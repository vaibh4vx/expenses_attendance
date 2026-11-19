
import numpy as np
import os


# ==============================
# PHASE 4 - CALCULATE METRICS
# ==============================


# Loading the cleaned dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(base_path, "Dataset")

data = np.genfromtxt(
    os.path.join(dataset_path, "cleaned_sales.csv"),
    delimiter=",",
    skip_header=1,
    dtype=float,
    filling_values=np.nan,
    ndmin=2
)

print("Dataset Shape:", data.shape)


# Extracting required columns

quantity = data[:, 5]
price = data[:, 6]
cost = data[:, 7]
discount = data[:, 8]


# ==============================
# 4.1 - VECTOR OPERATIONS AND BROADCASTING
# ==============================

print("\n========== VECTORIZATION AND BROADCASTING ==========")

# Vectorized arithmetic example

sample_prices = np.array([100, 200, 300])

updated_prices = sample_prices + 10

print("Original Prices:", sample_prices)
print("Updated Prices:", updated_prices)


# Broadcasting example

sample_quantity = np.array([[2], [3], [4]])

sample_price = np.array([10, 20])

sample_sales = sample_quantity * sample_price

print("\nQuantity Array Shape:", sample_quantity.shape)
print("Price Array Shape:", sample_price.shape)

print("Broadcasted Sales:")
print(sample_sales)


# ==============================
# 4.2 - CALCULATING GROSS SALES
# ==============================

valid_gross_sales = (
    np.isfinite(quantity) &
    np.isfinite(price) &
    (quantity > 0) &
    (quantity == np.floor(quantity)) &
    (price > 0)
)

gross_sales = np.full(len(data), np.nan)

gross_sales[valid_gross_sales] = (
    quantity[valid_gross_sales] *
    price[valid_gross_sales]
)

print("\n========== GROSS SALES ==========")

print("Valid Gross Sales Records:", np.sum(valid_gross_sales))
print("Invalid Gross Sales Records:", np.sum(~valid_gross_sales))

print("First 10 Gross Sales:")
print(gross_sales[:10])

print("Total Gross Sales:", np.round(np.nansum(gross_sales), 2))


# ==============================
# 4.3 - CALCULATING DISCOUNT AMOUNT
# ==============================

valid_discount_amount = (
    valid_gross_sales &
    np.isfinite(discount) &
    (discount >= 0) &
    (discount <= 1)
)

discount_amount = np.full(len(data), np.nan)

discount_amount[valid_discount_amount] = (
    gross_sales[valid_discount_amount] *
    discount[valid_discount_amount]
)

print("\n========== DISCOUNT AMOUNT ==========")

print("Valid Discount Records:", np.sum(valid_discount_amount))
print("Invalid Discount Records:", np.sum(~valid_discount_amount))

print("First 10 Discount Amounts:")
print(discount_amount[:10])

print("Total Discount Amount:",
      np.round(np.nansum(discount_amount), 2))


# ==============================
# 4.4 - CALCULATING NET REVENUE
# ==============================

valid_net_revenue = valid_discount_amount

net_revenue = np.full(len(data), np.nan)

net_revenue[valid_net_revenue] = (
    gross_sales[valid_net_revenue] -
    discount_amount[valid_net_revenue]
)

print("\n========== NET REVENUE ==========")

print("Valid Net Revenue Records:", np.sum(valid_net_revenue))
print("Invalid Net Revenue Records:", np.sum(~valid_net_revenue))

print("First 10 Net Revenue Values:")
print(net_revenue[:10])

print("Total Net Revenue:",
      np.round(np.nansum(net_revenue), 2))


# ==============================
# 4.5 - CALCULATING COST OF GOODS SOLD
# ==============================

valid_cogs = (
    np.isfinite(quantity) &
    np.isfinite(cost) &
    (quantity > 0) &
    (quantity == np.floor(quantity)) &
    (cost > 0)
)

cogs = np.full(len(data), np.nan)

cogs[valid_cogs] = (
    quantity[valid_cogs] *
    cost[valid_cogs]
)

print("\n========== COST OF GOODS SOLD ==========")

print("Valid COGS Records:", np.sum(valid_cogs))
print("Invalid COGS Records:", np.sum(~valid_cogs))

print("First 10 COGS Values:")
print(cogs[:10])

print("Total COGS:", np.round(np.nansum(cogs), 2))


# ==============================
# 4.6 - CALCULATING GROSS PROFIT
# ==============================

valid_gross_profit = (
    valid_net_revenue &
    valid_cogs
)

gross_profit = np.full(len(data), np.nan)

gross_profit[valid_gross_profit] = (
    net_revenue[valid_gross_profit] -
    cogs[valid_gross_profit]
)

print("\n========== GROSS PROFIT ==========")

print("Valid Gross Profit Records:",
      np.sum(valid_gross_profit))

print("Invalid Gross Profit Records:",
      np.sum(~valid_gross_profit))

print("First 10 Gross Profit Values:")
print(gross_profit[:10])

print("Total Gross Profit:",
      np.round(np.nansum(gross_profit), 2))


# ==============================
# 4.7 - CALCULATING MARGIN PERCENTAGE
# ==============================

valid_margin = (
    valid_gross_profit &
    (net_revenue != 0)
)

margin_percentage = np.full(len(data), np.nan)

np.divide(
    gross_profit,
    net_revenue,
    out=margin_percentage,
    where=valid_margin
)

margin_percentage[valid_margin] = (
    margin_percentage[valid_margin] * 100
)

print("\n========== MARGIN PERCENTAGE ==========")

print("Valid Margin Records:", np.sum(valid_margin))
print("Invalid Margin Records:", np.sum(~valid_margin))

print("First 10 Margin Percentages:")
print(margin_percentage[:10])

print("Average Transaction Margin:",
      np.round(np.nanmean(margin_percentage), 2))

print("Minimum Margin:",
      np.round(np.nanmin(margin_percentage), 2))

print("Maximum Margin:",
      np.round(np.nanmax(margin_percentage), 2))


# ==============================
# 4.8 - CALCULATING AVERAGE SELLING PRICE
# ==============================

valid_asp = valid_net_revenue

total_net_revenue = np.sum(net_revenue[valid_asp])

total_units_sold = np.sum(quantity[valid_asp])

average_selling_price = np.nan

if total_units_sold > 0:
    average_selling_price = np.divide(
        total_net_revenue,
        total_units_sold
    )

print("\n========== AVERAGE SELLING PRICE ==========")

print("Valid ASP Records:", np.sum(valid_asp))

print("Total Net Revenue for ASP:",
      np.round(total_net_revenue, 2))

print("Total Units Sold:", total_units_sold)

print("Average Selling Price:",
      np.round(average_selling_price, 2))


# ==============================
# 4.9 - USING NUMPY FUNCTIONS
# ==============================

print("\n========== NUMPY FUNCTIONS ==========")

# np.abs() - Checking difference between gross sales and net revenue
discount_check = np.full(len(data), np.nan)

discount_check[valid_net_revenue] = np.abs(
    gross_sales[valid_net_revenue] -
    net_revenue[valid_net_revenue]
)

print("Discount Reconciliation Difference:",
      np.round(
          np.nanmax(
              np.abs(
                  discount_check[valid_net_revenue] -
                  discount_amount[valid_net_revenue]
              )
          ), 6
      ))


# np.maximum() - Separating positive gross profit
positive_profit = np.maximum(gross_profit, 0)

print("Total Positive Gross Profit:",
      np.round(np.nansum(positive_profit), 2))


# np.minimum() - Separating negative gross profit
negative_profit = np.minimum(gross_profit, 0)

print("Total Negative Gross Profit:",
      np.round(np.nansum(negative_profit), 2))


# np.clip() is not used because clipping invalid
# discount rates would hide data quality problems.


# ==============================
# 4.10 - FINAL METRIC SUMMARY
# ==============================

# Using the same valid records for financial comparison

summary_net_revenue = np.sum(
    net_revenue[valid_gross_profit]
)

summary_cogs = np.sum(
    cogs[valid_gross_profit]
)

summary_gross_profit = np.sum(
    gross_profit[valid_gross_profit]
)


# Calculating overall business margin

overall_margin = np.nan

if summary_net_revenue != 0:
    overall_margin = np.divide(
        summary_gross_profit,
        summary_net_revenue
    ) * 100


# Calculating transaction margin statistics

average_margin = np.nanmean(margin_percentage)

minimum_margin = np.nanmin(margin_percentage)

maximum_margin = np.nanmax(margin_percentage)


# Displaying final summary

print("\n========== FINAL METRIC SUMMARY ==========")

print("Total Transactions:", len(data))

print("\nGross Sales:",
      np.round(np.nansum(gross_sales), 2))

print("Discount Amount:",
      np.round(np.nansum(discount_amount), 2))

print("Total Net Revenue:",
      np.round(np.nansum(net_revenue), 2))


print("\nCOGS (All Valid Records):",
      np.round(np.nansum(cogs), 2))

print("COGS (Common Valid Records):",
      np.round(summary_cogs, 2))

print("Gross Profit:",
      np.round(summary_gross_profit, 2))


print("\nOverall Business Margin:",
      np.round(overall_margin, 2), "%")

print("Average Transaction Margin:",
      np.round(average_margin, 2), "%")

print("Minimum Transaction Margin:",
      np.round(minimum_margin, 2), "%")

print("Maximum Transaction Margin:",
      np.round(maximum_margin, 2), "%")


print("\nTotal Units Sold for ASP:", total_units_sold)

print("Average Selling Price:",
      np.round(average_selling_price, 2))


print("\nValid Gross Sales Records:",
      np.sum(valid_gross_sales))

print("Valid Net Revenue Records:",
      np.sum(valid_net_revenue))

print("Valid COGS Records:",
      np.sum(valid_cogs))

print("Valid Gross Profit Records:",
      np.sum(valid_gross_profit))

print("Valid Margin Records:",
      np.sum(valid_margin))

print("Valid ASP Records:",
      np.sum(valid_asp))


print("\n==========================================")
print("PHASE 4 COMPLETED")
