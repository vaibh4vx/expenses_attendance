
import numpy as np
import os


# PHASE 5 - DESCRIBE DISTRIBUTIONS

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

quantity = data[:, 5]

# 5.1 - Mean

valid_quantity = (
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

quantity_values = quantity[valid_quantity]

mean_quantity = np.mean(quantity_values)

# 5.2 - Median

median_quantity = np.median(quantity_values)

# 5.3 - Average

average_quantity = np.average(quantity_values)

# 5.4 - Minimum and Maximum

minimum_quantity = np.min(quantity_values)
maximum_quantity = np.max(quantity_values)

# 5.5 - Peak-to-Peak

quantity_ptp = np.ptp(quantity_values)

# 5.6 - Variance

quantity_variance = np.var(quantity_values)

# 5.7 - Standard Deviation

quantity_std = np.std(quantity_values)

# 5.8 - Percentiles

quantity_p25 = np.percentile(quantity_values, 25)
quantity_p50 = np.percentile(quantity_values, 50)
quantity_p75 = np.percentile(quantity_values, 75)
quantity_p90 = np.percentile(quantity_values, 90)

# 5.9 - Quantiles and Quartiles

quantity_quantile_10 = np.quantile(quantity_values, 0.10)
quantity_quantile_50 = np.quantile(quantity_values, 0.50)
quantity_quantile_90 = np.quantile(quantity_values, 0.90)

quantity_q1 = np.quantile(quantity_values, 0.25)
quantity_q2 = np.quantile(quantity_values, 0.50)
quantity_q3 = np.quantile(quantity_values, 0.75)

quantity_iqr = quantity_q3 - quantity_q1

# 5.10 - Histogram

quantity_counts, quantity_bins = np.histogram(quantity_values, bins=5)

# 5.11 - Sample vs Population Statistics

population_variance = np.var(quantity_values, ddof=0)
population_std = np.std(quantity_values, ddof=0)

sample_variance = np.var(quantity_values, ddof=1)
sample_std = np.std(quantity_values, ddof=1)

# 5.12 - Weighted vs Unweighted Averages

unit_price = data[:, 6]

valid_price_quantity = (
    np.isfinite(unit_price) &
    np.isfinite(quantity) &
    (unit_price > 0) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

price_values = unit_price[valid_price_quantity]
quantity_weights = quantity[valid_price_quantity]

unweighted_average = np.mean(price_values)

weighted_average = np.average(
    price_values,
    weights=quantity_weights
)

# Outlier Interpretation

lower_bound = quantity_q1 - (1.5 * quantity_iqr)
upper_bound = quantity_q3 + (1.5 * quantity_iqr)

outlier_mask = (
    (quantity_values < lower_bound) |
    (quantity_values > upper_bound)
)

outlier_values = quantity_values[outlier_mask]
outlier_count = len(outlier_values)

# PHASE 5 - FINAL SUMMARY

print("\n========== PHASE 5 - DESCRIPTIVE STATISTICS SUMMARY ==========")

print("\n--- Dataset Information ---")
print("Total Transactions:", len(data))
print("Valid Quantity Records:", len(quantity_values))

print("\n--- Central Tendency ---")
print("Mean:", mean_quantity)
print("Median:", median_quantity)
print("Average:", average_quantity)

print("\n--- Minimum, Maximum and Range ---")
print("Minimum:", minimum_quantity)
print("Maximum:", maximum_quantity)
print("Peak-to-Peak:", quantity_ptp)

print("\n--- Variance and Standard Deviation ---")
print("Variance:", quantity_variance)
print("Standard Deviation:", quantity_std)

print("\n--- Percentiles ---")
print("25th Percentile:", quantity_p25)
print("50th Percentile:", quantity_p50)
print("75th Percentile:", quantity_p75)
print("90th Percentile:", quantity_p90)

print("\n--- Quantiles ---")
print("10th Quantile:", quantity_quantile_10)
print("50th Quantile:", quantity_quantile_50)
print("90th Quantile:", quantity_quantile_90)

print("\n--- Quartiles ---")
print("First Quartile (Q1):", quantity_q1)
print("Second Quartile (Q2):", quantity_q2)
print("Third Quartile (Q3):", quantity_q3)
print("Interquartile Range (IQR):", quantity_iqr)

print("\n--- Histogram ---")
for i in range(len(quantity_counts)):
    print(
        "Range:", quantity_bins[i], "to", quantity_bins[i + 1],
        "| Frequency:", quantity_counts[i]
    )

print("\n--- Sample vs Population ---")
print("Population Variance:", population_variance)
print("Population Standard Deviation:", population_std)
print("Sample Variance:", sample_variance)
print("Sample Standard Deviation:", sample_std)

print("\n--- Weighted vs Unweighted Average ---")
print("Unweighted Average Price:", unweighted_average)
print("Weighted Average Price:", weighted_average)

print("\n--- Outlier Interpretation ---")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Outliers:", outlier_count)
print("Outlier Values:", outlier_values)

if outlier_count == 0:
    print("Conclusion: No quantity outliers detected using the IQR method.")
else:
    print("Conclusion: Potential quantity outliers detected.")

print("\n==============================================================")
print("PHASE 5 COMPLETED")
