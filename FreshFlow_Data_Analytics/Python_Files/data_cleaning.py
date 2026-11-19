
import numpy as np
import os

# Phase 3

# Loading the dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(base_path, "Dataset")
report_path = os.path.join(base_path, "Reports")

os.makedirs(report_path, exist_ok=True)

data = np.genfromtxt(
    os.path.join(dataset_path, "freshflow_sales.csv"),
    delimiter=",",
    skip_header=1,
    dtype=float,
    filling_values=np.nan,
    ndmin=2
)

print("Original dataset shape:", data.shape)


# Column positions

transaction_id = data[:, 0]
day = data[:, 1]
outlet = data[:, 2]
product = data[:, 3]
category = data[:, 4]
quantity = data[:, 5]
price = data[:, 6]
cost = data[:, 7]
discount = data[:, 8]
stock = data[:, 9]
payment = data[:, 10]
rating = data[:, 11]


# 1. Checking missing values

missing_values = np.isnan(data)
missing_count = np.sum(missing_values)

print("\nMissing values:", missing_count)

column_names = [
    "transaction_id", "day_index", "outlet_id", "product_code",
    "category_code", "quantity", "unit_price", "unit_cost",
    "discount_rate", "stock_before_units", "payment_code",
    "customer_rating"
]

for i in range(data.shape[1]):
    print(column_names[i], ":", np.sum(missing_values[:, i]))

required_data = data[:, :11]

missing_required_flag = np.any(np.isnan(required_data), axis=1)


# 2. Checking infinite and non-finite values

infinite_values = np.isinf(data)

print("\nInfinite values:", np.sum(infinite_values))

invalid_rating_infinite = np.isinf(rating)

non_finite_required_flag = np.any(
    ~np.isfinite(required_data),
    axis=1
)


# 3. Checking integer values

transaction_id_flag = np.isfinite(transaction_id) & (
    transaction_id != np.floor(transaction_id)
)

day_integer_flag = np.isfinite(day) & (
    day != np.floor(day)
)

outlet_integer_flag = np.isfinite(outlet) & (
    outlet != np.floor(outlet)
)

product_integer_flag = np.isfinite(product) & (
    product != np.floor(product)
)

category_integer_flag = np.isfinite(category) & (
    category != np.floor(category)
)

payment_integer_flag = np.isfinite(payment) & (
    payment != np.floor(payment)
)

quantity_integer_flag = np.isfinite(quantity) & (
    quantity != np.floor(quantity)
)

stock_integer_flag = np.isfinite(stock) & (
    stock != np.floor(stock)
)

invalid_integer_value_flag = (
    transaction_id_flag |
    day_integer_flag |
    outlet_integer_flag |
    product_integer_flag |
    category_integer_flag |
    payment_integer_flag |
    quantity_integer_flag |
    stock_integer_flag
)

print("\nInvalid integer values:", np.sum(invalid_integer_value_flag))


# 4. Checking valid code values

invalid_outlet = np.isfinite(outlet) & (
    ~np.isin(outlet, [1, 2, 3])
)

invalid_product = np.isfinite(product) & (
    ~np.isin(product, [1, 2, 3, 4, 5, 6, 7, 8])
)

invalid_category = np.isfinite(category) & (
    ~np.isin(category, [1, 2, 3, 4])
)

invalid_payment = np.isfinite(payment) & (
    ~np.isin(payment, [1, 2, 3, 4])
)

invalid_code_flag = (
    invalid_outlet |
    invalid_product |
    invalid_category |
    invalid_payment
)

print("\nInvalid outlet IDs:", np.sum(invalid_outlet))
print("Invalid product codes:", np.sum(invalid_product))
print("Invalid category codes:", np.sum(invalid_category))
print("Invalid payment codes:", np.sum(invalid_payment))


# 5. Checking values outside allowed ranges

invalid_transaction_id = np.isfinite(transaction_id) & (
    transaction_id <= 0
)

invalid_day = np.isfinite(day) & (
    (day < 1) |
    (day > 30) |
    (day != np.floor(day))
)

invalid_quantity = np.isfinite(quantity) & (
    (quantity <= 0) |
    (quantity != np.floor(quantity))
)

invalid_price = np.isfinite(price) & (price <= 0)

invalid_cost = np.isfinite(cost) & (cost <= 0)

invalid_discount = np.isfinite(discount) & (
    (discount < 0) | (discount > 1)
)

invalid_stock = np.isfinite(stock) & (
    (stock < 0) |
    (stock != np.floor(stock))
)

invalid_rating = (
    (np.isfinite(rating) & ((rating < 1) | (rating > 5))) |
    invalid_rating_infinite
)

print("\nInvalid transaction IDs:", np.sum(invalid_transaction_id))
print("Invalid days:", np.sum(invalid_day))
print("Invalid quantities:", np.sum(invalid_quantity))
print("Invalid prices:", np.sum(invalid_price))
print("Invalid costs:", np.sum(invalid_cost))
print("Invalid discounts:", np.sum(invalid_discount))
print("Invalid stock:", np.sum(invalid_stock))
print("Invalid ratings:", np.sum(invalid_rating))


# 6. Checking product and category combinations

print("\nProduct and category combinations:")

valid_mapping_rows = (
    np.isfinite(product) &
    np.isfinite(category) &
    ~invalid_product &
    ~invalid_category &
    ~product_integer_flag &
    ~category_integer_flag
)

product_category = data[valid_mapping_rows][:, [3, 4]]

combinations, counts = np.unique(
    product_category,
    axis=0,
    return_counts=True
)

for i in range(len(combinations)):
    print(
        "Product:", int(combinations[i, 0]),
        "Category:", int(combinations[i, 1]),
        "Count:", counts[i]
    )

expected_category = np.zeros(len(data))

expected_category[np.isin(product, [1, 2, 3, 4])] = 1
expected_category[np.isin(product, [5, 6])] = 2
expected_category[product == 7] = 3
expected_category[product == 8] = 4

mapping_check = (
    np.isfinite(product) &
    np.isfinite(category) &
    ~invalid_product &
    ~invalid_category &
    ~product_integer_flag &
    ~category_integer_flag
)

invalid_mapping = np.zeros(len(data), dtype=bool)

invalid_mapping[mapping_check] = (
    category[mapping_check] != expected_category[mapping_check]
)

print("\nInvalid product-category combinations:", np.sum(invalid_mapping))


# 7. Checking duplicate transaction IDs

valid_ids = transaction_id[np.isfinite(transaction_id)]

unique_ids, id_counts = np.unique(
    valid_ids,
    return_counts=True
)

duplicate_ids = unique_ids[id_counts > 1]

duplicate_flag = np.isin(transaction_id, duplicate_ids)

print("\nDuplicate transaction IDs:", duplicate_ids)
print("Rows with duplicate transaction IDs:", np.sum(duplicate_flag))


# 8. Copies and views

view_data = data[:, 5:8]
copy_data = data[:, 5:8].copy()

print("\nOriginal data shape:", data.shape)
print("View shape:", view_data.shape)
print("Copy shape:", copy_data.shape)


# 9. Safe division practice

average_price_flag = (
    np.isfinite(quantity) &
    np.isfinite(price) &
    (quantity > 0) &
    (price > 0) &
    (quantity == np.floor(quantity))
)

valid_quantity = quantity[average_price_flag]
valid_price = price[average_price_flag]

total_quantity = np.sum(valid_quantity)
total_sales_value = np.sum(valid_quantity * valid_price)

if total_quantity != 0:
    average_price = total_sales_value / total_quantity
else:
    average_price = np.nan

print("\nWeighted average selling price:", average_price)


# 10. Creating quality flags

suspicious_flag = (
    non_finite_required_flag |
    invalid_rating |
    invalid_transaction_id |
    invalid_day |
    invalid_quantity |
    invalid_price |
    invalid_cost |
    invalid_discount |
    invalid_stock |
    invalid_integer_value_flag |
    invalid_code_flag |
    invalid_mapping |
    duplicate_flag
)

valid_flag = ~suspicious_flag

print("\nTotal records:", len(data))
print("Valid records:", np.sum(valid_flag))
print("Suspicious records:", np.sum(suspicious_flag))


# 11. Checking suspicious records

suspicious_records = data[suspicious_flag]

print("\nFirst 10 suspicious records:")
print(suspicious_records[:10])


# 12. Creating a cleaned copy

cleaned_data = data.copy()

cleaned_data[invalid_rating, 11] = np.nan

cleaned_file = os.path.join(dataset_path, "cleaned_sales.csv")

np.savetxt(
    cleaned_file,
    cleaned_data,
    delimiter=",",
    fmt="%.4f",
    header="transaction_id,day_index,outlet_id,product_code,category_code,quantity,unit_price,unit_cost,discount_rate,stock_before_units,payment_code,customer_rating",
    comments=""
)

print("\nCleaned dataset saved successfully.")
print("Invalid ratings changed to NaN:", np.sum(invalid_rating))
print("Suspicious rows retained for investigation:", np.sum(suspicious_flag))


# 13. Creating quality report

rule_names = [
    "Missing cells",
    "Rows with missing required values",
    "Infinite values",
    "Rows with non-finite required values",
    "Invalid transaction IDs",
    "Invalid days",
    "Invalid quantities",
    "Invalid prices",
    "Invalid costs",
    "Invalid discounts",
    "Invalid stock",
    "Invalid ratings",
    "Invalid integer values",
    "Invalid outlet IDs",
    "Invalid product codes",
    "Invalid category codes",
    "Invalid payment codes",
    "Invalid product-category combinations",
    "Rows with duplicate transaction IDs",
    "Valid records",
    "Suspicious records",
    "Invalid ratings changed to NaN",
    "Suspicious rows retained for investigation",
    "Rows in cleaned dataset"
]

rule_counts = [
    missing_count,
    np.sum(missing_required_flag),
    np.sum(infinite_values),
    np.sum(non_finite_required_flag),
    np.sum(invalid_transaction_id),
    np.sum(invalid_day),
    np.sum(invalid_quantity),
    np.sum(invalid_price),
    np.sum(invalid_cost),
    np.sum(invalid_discount),
    np.sum(invalid_stock),
    np.sum(invalid_rating),
    np.sum(invalid_integer_value_flag),
    np.sum(invalid_outlet),
    np.sum(invalid_product),
    np.sum(invalid_category),
    np.sum(invalid_payment),
    np.sum(invalid_mapping),
    np.sum(duplicate_flag),
    np.sum(valid_flag),
    np.sum(suspicious_flag),
    np.sum(invalid_rating),
    np.sum(suspicious_flag),
    len(cleaned_data)
]

quality_report = np.column_stack((
    np.array(rule_names),
    np.array(rule_counts, dtype=str)
))

np.savetxt(
    os.path.join(report_path, "data_quality_report.csv"),
    quality_report,
    delimiter=",",
    fmt="%s",
    header="Rule,Count",
    comments=""
)

print("Quality report saved successfully.")


# 14. Checking the saved dataset

check_data = np.genfromtxt(
    cleaned_file,
    delimiter=",",
    skip_header=1,
    dtype=float,
    ndmin=2
)

print("\nOriginal shape:", data.shape)
print("Cleaned shape:", check_data.shape)
print("Original and cleaned shapes match:", data.shape == check_data.shape)

print("\nPhase 3 completed.")
