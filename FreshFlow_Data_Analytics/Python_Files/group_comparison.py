import numpy as np
import os

# 6.1 - Loading the Cleaned Dataset

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

outlet = data[:, 2]
product = data[:, 3]
category = data[:, 4]
quantity = data[:, 5]
price = data[:, 6]
cost = data[:, 7]
discount = data[:, 8]


# 6.2 - Validating Group Codes

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

expected_category = np.zeros(len(data))

expected_category[np.isin(product, [1, 2, 3, 4])] = 1
expected_category[np.isin(product, [5, 6])] = 2
expected_category[product == 7] = 3
expected_category[product == 8] = 4

valid_product_category = (
    valid_product_code &
    valid_category_code &
    (category == expected_category)
)

invalid_product_category = (
    valid_product_code &
    valid_category_code &
    ~valid_product_category
)

print(
    "Invalid product-category combinations:",
    np.sum(invalid_product_category)
)

valid_group_records = (
    valid_outlet_code &
    valid_product_category
)


# 6.3 - Identifying Unique Outlets

valid_outlet = valid_group_records

unique_outlets, outlet_counts = np.unique(
    outlet[valid_outlet],
    return_counts=True
)

print("\nUnique Outlets:", unique_outlets)
print("Outlet Record Counts:", outlet_counts)


# 6.4 - Calculating Total Units Sold by Outlet

valid_outlet_quantity = (
    valid_outlet &
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

outlet_units = np.bincount(
    outlet[valid_outlet_quantity].astype(int),
    weights=quantity[valid_outlet_quantity],
    minlength=4
)

print("\nTotal Units Sold by Outlet:")

for i in range(1, 4):
    print("Outlet", i, ":", outlet_units[i])


# 6.5 - Calculating Outlet-wise Revenue

valid_revenue = (
    valid_outlet_quantity &
    np.isfinite(price) &
    np.isfinite(discount) &
    (price > 0) &
    (discount >= 0) &
    (discount <= 1)
)

net_revenue = (
    quantity[valid_revenue] *
    price[valid_revenue] *
    (1 - discount[valid_revenue])
)

outlet_revenue = np.bincount(
    outlet[valid_revenue].astype(int),
    weights=net_revenue,
    minlength=4
)

print("\nTotal Net Revenue by Outlet:")

for i in range(1, 4):
    print("Outlet", i, ":", round(outlet_revenue[i], 2))


# 6.6 - Calculating Outlet-wise Gross Profit

valid_profit = (
    valid_revenue &
    np.isfinite(cost) &
    (cost > 0)
)

outlet_cogs = (
    quantity[valid_profit] *
    cost[valid_profit]
)

outlet_net_revenue = (
    quantity[valid_profit] *
    price[valid_profit] *
    (1 - discount[valid_profit])
)

gross_profit = outlet_net_revenue - outlet_cogs

outlet_profit = np.bincount(
    outlet[valid_profit].astype(int),
    weights=gross_profit,
    minlength=4
)

print("\nTotal Gross Profit by Outlet:")

for i in range(1, 4):
    print("Outlet", i, ":", round(outlet_profit[i], 2))


# 6.7 - Identifying Unique Products

valid_product = valid_group_records

unique_products, product_counts = np.unique(
    product[valid_product],
    return_counts=True
)

print("\nUnique Products:", unique_products)
print("Product Record Counts:", product_counts)


# 6.8 - Calculating Total Units Sold by Product

valid_product_quantity = (
    valid_product &
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

product_units = np.bincount(
    product[valid_product_quantity].astype(int),
    weights=quantity[valid_product_quantity],
    minlength=9
)

print("\nTotal Units Sold by Product:")

for i in range(1, 9):
    print("Product", i, ":", product_units[i])


# 6.9 - Calculating Product-wise Revenue

valid_product_revenue = (
    valid_product_quantity &
    np.isfinite(price) &
    np.isfinite(discount) &
    (price > 0) &
    (discount >= 0) &
    (discount <= 1)
)

product_net_revenue = (
    quantity[valid_product_revenue] *
    price[valid_product_revenue] *
    (1 - discount[valid_product_revenue])
)

product_revenue = np.bincount(
    product[valid_product_revenue].astype(int),
    weights=product_net_revenue,
    minlength=9
)

print("\nTotal Net Revenue by Product:")

for i in range(1, 9):
    print("Product", i, ":", round(product_revenue[i], 2))


# 6.10 - Calculating Product-wise Gross Profit

valid_product_profit = (
    valid_product_revenue &
    np.isfinite(cost) &
    (cost > 0)
)

product_cogs = (
    quantity[valid_product_profit] *
    cost[valid_product_profit]
)

product_net_revenue_valid = (
    quantity[valid_product_profit] *
    price[valid_product_profit] *
    (1 - discount[valid_product_profit])
)

product_gross_profit = (
    product_net_revenue_valid - product_cogs
)

product_profit = np.bincount(
    product[valid_product_profit].astype(int),
    weights=product_gross_profit,
    minlength=9
)

print("\nTotal Gross Profit by Product:")

for i in range(1, 9):
    print("Product", i, ":", round(product_profit[i], 2))


# 6.11 - Identifying Unique Categories

valid_category = valid_group_records

unique_categories, category_counts = np.unique(
    category[valid_category],
    return_counts=True
)

print("\nUnique Categories:", unique_categories)
print("Category Record Counts:", category_counts)


# 6.12 - Calculating Total Units Sold by Category

valid_category_quantity = (
    valid_category &
    np.isfinite(quantity) &
    (quantity > 0) &
    (quantity == np.floor(quantity))
)

category_units = np.bincount(
    category[valid_category_quantity].astype(int),
    weights=quantity[valid_category_quantity],
    minlength=5
)

print("\nTotal Units Sold by Category:")

for i in range(1, 5):
    print("Category", i, ":", category_units[i])


# 6.13 - Calculating Category-wise Revenue

valid_category_revenue = (
    valid_category_quantity &
    np.isfinite(price) &
    np.isfinite(discount) &
    (price > 0) &
    (discount >= 0) &
    (discount <= 1)
)

category_net_revenue = (
    quantity[valid_category_revenue] *
    price[valid_category_revenue] *
    (1 - discount[valid_category_revenue])
)

category_revenue = np.bincount(
    category[valid_category_revenue].astype(int),
    weights=category_net_revenue,
    minlength=5
)

print("\nTotal Net Revenue by Category:")

for i in range(1, 5):
    print("Category", i, ":", round(category_revenue[i], 2))


# 6.14 - Calculating Category-wise Gross Profit

valid_category_profit = (
    valid_category_revenue &
    np.isfinite(cost) &
    (cost > 0)
)

category_cogs = (
    quantity[valid_category_profit] *
    cost[valid_category_profit]
)

category_net_revenue_valid = (
    quantity[valid_category_profit] *
    price[valid_category_profit] *
    (1 - discount[valid_category_profit])
)

category_gross_profit = (
    category_net_revenue_valid - category_cogs
)

category_profit = np.bincount(
    category[valid_category_profit].astype(int),
    weights=category_gross_profit,
    minlength=5
)

print("\nTotal Gross Profit by Category:")

for i in range(1, 5):
    print("Category", i, ":", round(category_profit[i], 2))


outlet_codes = np.array([1, 2, 3])
product_codes = np.array([1, 2, 3, 4, 5, 6, 7, 8])


# 6.15 - Ranking Outlets by Units Sold

outlet_units_values = outlet_units[1:]
outlet_order = np.argsort(outlet_units_values)[::-1]

print("\nOutlet Ranking by Units Sold:")

for i in outlet_order:
    print(
        "Outlet", outlet_codes[i],
        ":", outlet_units_values[i], "units"
    )


# 6.16 - Ranking Outlets by Net Revenue

outlet_revenue_values = outlet_revenue[1:]
outlet_revenue_order = np.argsort(outlet_revenue_values)[::-1]

print("\nOutlet Ranking by Net Revenue:")

for i in outlet_revenue_order:
    print(
        "Outlet", outlet_codes[i],
        ": ₹", round(outlet_revenue_values[i], 2)
    )


# 6.17 - Ranking Outlets by Gross Profit

outlet_profit_values = outlet_profit[1:]
outlet_profit_order = np.argsort(outlet_profit_values)[::-1]

print("\nOutlet Ranking by Gross Profit:")

for i in outlet_profit_order:
    print(
        "Outlet", outlet_codes[i],
        ": ₹", round(outlet_profit_values[i], 2)
    )


# 6.18 - Ranking Products by Units Sold

product_units_values = product_units[1:]
product_units_order = np.argsort(product_units_values)[::-1]

print("\nProduct Ranking by Units Sold:")

for i in product_units_order:
    print(
        "Product", product_codes[i],
        ":", product_units_values[i], "units"
    )


# 6.19 - Ranking Products by Net Revenue

product_revenue_values = product_revenue[1:]
product_revenue_order = np.argsort(product_revenue_values)[::-1]

print("\nProduct Ranking by Net Revenue:")

for i in product_revenue_order:
    print(
        "Product", product_codes[i],
        ": ₹", round(product_revenue_values[i], 2)
    )


# 6.20 - Ranking Products by Gross Profit

product_profit_values = product_profit[1:]
product_profit_order = np.argsort(product_profit_values)[::-1]

print("\nProduct Ranking by Gross Profit:")

for i in product_profit_order:
    print(
        "Product", product_codes[i],
        ": ₹", round(product_profit_values[i], 2)
    )


# 6.21 - Using Sort, Argmin and Argmax

print("\nOutlet Revenue Sorted in Descending Order:")
print(np.sort(outlet_revenue_values)[::-1])

outlet_min_revenue_index = np.argmin(outlet_revenue_values)
outlet_max_revenue_index = np.argmax(outlet_revenue_values)

print(
    "Outlet with Minimum Revenue:",
    outlet_codes[outlet_min_revenue_index]
)

print(
    "Outlet with Maximum Revenue:",
    outlet_codes[outlet_max_revenue_index]
)

print("\nProduct Revenue Sorted in Descending Order:")
print(np.sort(product_revenue_values)[::-1])

product_min_revenue_index = np.argmin(product_revenue_values)
product_max_revenue_index = np.argmax(product_revenue_values)

print(
    "Product with Minimum Revenue:",
    product_codes[product_min_revenue_index]
)

print(
    "Product with Maximum Revenue:",
    product_codes[product_max_revenue_index]
)


# 6.22 - Creating Outlet-wise Aligned Matrix

outlet_comparison = np.column_stack((
    outlet_units_values,
    outlet_revenue_values,
    outlet_profit_values
))

print("\nOutlet Comparison Matrix:")
print(outlet_comparison)

print("\nOutlet Matrix Shape:", outlet_comparison.shape)
print("Columns: Units Sold, Net Revenue, Gross Profit")


# 6.23 - Creating Product-wise Aligned Matrix

product_comparison = np.column_stack((
    product_units_values,
    product_revenue_values,
    product_profit_values
))

print("\nProduct Comparison Matrix:")
print(product_comparison)

print("\nProduct Matrix Shape:", product_comparison.shape)
print("Columns: Units Sold, Net Revenue, Gross Profit")


# 6.24 - Creating Outlet-Category Units Matrix

outlet_category_units = np.zeros((3, 4))

for i in range(3):
    for j in range(4):

        valid_outlet_category = (
            valid_outlet_quantity &
            (outlet == i + 1) &
            (category == j + 1)
        )

        outlet_category_units[i, j] = np.sum(
            quantity[valid_outlet_category]
        )

print("\nOutlet-Category Units Matrix:")
print(outlet_category_units)


# 6.25 - Performing Axis Operations

outlet_total_units = np.sum(
    outlet_category_units,
    axis=1
)

category_total_units = np.sum(
    outlet_category_units,
    axis=0
)

print("\nTotal Units Sold by Each Outlet using Axis 1:")

for i in range(3):
    print("Outlet", i + 1, ":", outlet_total_units[i])

print("\nTotal Units Sold in Each Category using Axis 0:")

for i in range(4):
    print("Category", i + 1, ":", category_total_units[i])