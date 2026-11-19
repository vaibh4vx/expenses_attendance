
import numpy as np


# ==============================
# PHASE 1 - REPRESENT AND INSPECT DATA
# ==============================

# Loading the CSV file
data = np.genfromtxt(
    "../Dataset/freshflow_sales.csv",
    delimiter=",",
    skip_header=1,
    dtype=float,
    filling_values=np.nan,
    ndmin=2
)

print("Complete Dataset:")
print(data)


# Understanding array dimensions
print("\n========== ARRAY INFORMATION ==========")

print("Dataset Shape:", np.shape(data))
print("Total Elements:", np.size(data))
print("Number of Dimensions:", np.ndim(data))


# Understanding data types
print("Data Type:", data.dtype)


# Reading column headers
with open("../Dataset/freshflow_sales.csv", "r") as file:
    headers = file.readline()

    print("\nColumn Headers:")
    print(headers)


# Extracting numeric columns
print("\n========== NUMERIC COLUMNS ==========")

quantity = data[:, 5]
unit_price = data[:, 6]
unit_cost = data[:, 7]

print("Quantity Column:")
print(quantity[:10])

print("Unit Price Column:")
print(unit_price[:10])

print("Unit Cost Column:")
print(unit_cost[:10])


# Understanding Python List and NumPy Array
price_list = [50, 60, 70, 80]

price_array = np.array(price_list)

print("\n========== LIST VS NUMPY ARRAY ==========")

print("Python List:", price_list)
print("NumPy Array:", price_array)


# Adding 10 to every price
new_price = price_array + 10

print("Updated Prices:", new_price)


# NumPy Array Creation Functions
print("\n========== ARRAY CREATION ==========")

a = np.array([10, 20, 30, 40])

b = np.arange(1, 11)

c = np.zeros(5)

d = np.ones(5)

e = np.empty(5)

f = np.full(5, 100)

print("Array:", a)
print("Range:", b)
print("Zeros:", c)
print("Ones:", d)
print("Empty:", e)
print("Full:", f)


# Creating a 2D array
g = np.zeros((3, 4))

print("\n2D Zeros Array:")
print(g)


# Understanding astype()
print("\n========== ASTYPE() ==========")

price_float = np.array([50.5, 60.5, 70.5])

price_int = price_float.astype(int)

print("Original Array:", price_float)
print("Converted Array:", price_int)
print("Original Data Type:", price_float.dtype)
print("Converted Data Type:", price_int.dtype)


# Understanding itemsize and nbytes
print("\n========== ARRAY MEMORY ==========")

print("Memory per Element:", data.itemsize, "bytes")
print("Total Array Memory:", data.nbytes, "bytes")


# ==============================
# PHASE 2 - SELECT AND FILTER RECORDS
# ==============================


# 2.1 - Positive and Negative Indexing
print("\n========== INDEXING ==========")

print("First Transaction:", data[0])
print("Last Transaction:", data[-1])
print("First Transaction ID:", data[0, 0])


# 2.2 - Slicing
print("\n========== SLICING ==========")

print("First 5 Transactions:")
print(data[0:5])

print("Transactions from Index 5 to 9:")
print(data[5:10])

print("Every Second Transaction:")
print(data[0:10:2])

print("Last 5 Transactions:")
print(data[-5:])


# 2.3 - Row and Column Extraction
print("\n========== ROW AND COLUMN EXTRACTION ==========")

print("First Transaction:")
print(data[0, :])

print("Transaction IDs:")
print(data[:, 0])

print("Quantities:")
print(data[:, 5])

print("Unit Prices:")
print(data[:, 6])

print("Selected Columns:")
print(data[:, [0, 5, 6]])


# 2.4 - Boolean Masking
print("\n========== BOOLEAN MASKING ==========")

quantity = data[:, 5]

mask = quantity > 5

print("Boolean Mask:")
print(mask[:10])

filtered_data = data[mask]

print("Transactions with Quantity Greater than 5:")
print(filtered_data)

print("Total Selected Transactions:", len(filtered_data))


# 2.5 - Compound Conditions
print("\n========== COMPOUND CONDITIONS ==========")

quantity = data[:, 5]
discount = data[:, 8]

# AND condition
mask_and = (quantity > 5) & (discount > 0.10)

filtered_data = data[mask_and]

print("Transactions with Quantity > 5 AND Discount > 10%:")
print(filtered_data)

print("Total Selected Transactions:", len(filtered_data))


# OR condition
mask_or = (quantity > 5) | (discount > 0.10)

filtered_data = data[mask_or]

print("\nTransactions with Quantity > 5 OR Discount > 10%:")
print(filtered_data)

print("Total Selected Transactions:", len(filtered_data))


# NOT condition
mask_not = ~(quantity > 5)

filtered_data = data[mask_not]

print("\nTransactions where Quantity is NOT Greater than 5:")
print(filtered_data)

print("Total Selected Transactions:", len(filtered_data))


# 2.6 - Fancy Indexing
print("\n========== FANCY INDEXING ==========")

selected_transactions = data[[0, 5, 10, 15]]

print("Selected Transactions:")
print(selected_transactions)


selected_columns = data[:, [0, 5, 6, 8]]

print("Selected Columns:")
print(selected_columns[:5])


selected_data = data[[0, 5, 10], :][:, [0, 5, 6, 8]]

print("Selected Rows and Columns:")
print(selected_data)


# 2.7 - np.where()
print("\n========== NP.WHERE() ==========")

quantity = data[:, 5]

result = np.where(quantity > 5)

print("Indexes of Transactions with Quantity > 5:")
print(result)

selected_transactions = data[result]

print("Selected Transactions:")
print(selected_transactions)

print("Total Matching Transactions:", len(result[0]))


# 2.8 - np.nonzero()
print("\n========== NP.NONZERO() ==========")

result = np.nonzero(quantity > 5)

print("Indexes of Transactions with Quantity > 5:")
print(result)

selected_transactions = data[result]

print("Selected Transactions:")
print(selected_transactions)

print("Total Matching Transactions:", len(result[0]))


# 2.9 - np.select()
print("\n========== NP.SELECT() ==========")

conditions = [
    quantity <= 2,
    (quantity > 2) & (quantity <= 5),
    quantity > 5
]

choices = [
    "Low Quantity",
    "Medium Quantity",
    "High Quantity"
]

quantity_category = np.select(
    conditions,
    choices,
    default="Unknown"
)

print("Quantity Categories:")
print(quantity_category[:20])

print("Total Low Quantity Transactions:",
      np.sum(quantity_category == "Low Quantity"))

print("Total Medium Quantity Transactions:",
      np.sum(quantity_category == "Medium Quantity"))

print("Total High Quantity Transactions:",
      np.sum(quantity_category == "High Quantity"))
