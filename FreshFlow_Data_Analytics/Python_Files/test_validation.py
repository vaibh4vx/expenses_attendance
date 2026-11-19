
import numpy as np
import time
from pathlib import Path


# Dataset Path

def get_dataset_path():

    project_folder = Path(__file__).resolve().parent.parent

    return project_folder / "Dataset" / "cleaned_sales.csv"


# 11.1 Shape Mismatch

def validate_array_shapes(array1, array2):

    if array1.shape != array2.shape:
        raise ValueError("Both arrays must have the same shape")

    return True


def calculate_net_sales(sales, discount):

    validate_array_shapes(sales, discount)

    return sales - discount


def test_shape_mismatch():

    print("\n========== 11.1 SHAPE MISMATCH ==========")

    sales = np.array([100, 200, 300, 400, 500])
    discount = np.array([10, 20, 30, 40, 50])

    print("Sales Shape:", sales.shape)
    print("Discount Shape:", discount.shape)

    result = calculate_net_sales(sales, discount)

    expected = np.array([90, 180, 270, 360, 450])

    np.testing.assert_array_equal(result, expected)

    print("Matching shape test passed")

    wrong_discount = np.array([10, 20, 30, 40])

    try:
        calculate_net_sales(sales, wrong_discount)

    except ValueError as error:
        print("Shape mismatch detected:", error)

    else:
        raise AssertionError("Shape mismatch was not detected")


# 11.2 Broadcasting

def test_broadcasting():

    print("\n========== 11.2 BROADCASTING ==========")

    array1 = np.array([[10], [20], [30]])
    array2 = np.array([1, 2, 3, 4])

    print("First Array Shape:", array1.shape)
    print("Second Array Shape:", array2.shape)

    result = array1 + array2

    print("Broadcasting Result:")
    print(result)

    expected = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34]
    ])

    np.testing.assert_array_equal(result, expected)

    print("Compatible broadcasting test passed")

    try:
        array3 = np.array([10, 20, 30])
        array4 = np.array([1, 2])

        result = array3 + array4

    except ValueError:
        print("Incompatible broadcasting test passed")

    else:
        raise AssertionError("Incompatible shapes were not detected")


# 11.3 Dtype, Overflow and NaN

def test_dtype_conversion():

    print("\n========== DTYPE CONVERSION ==========")

    values = np.array([10.8, 20.5, 30.9])

    print("Original Array:", values)
    print("Original Dtype:", values.dtype)

    integer_values = values.astype(np.int64)

    print("Converted Array:", integer_values)
    print("Converted Dtype:", integer_values.dtype)

    expected = np.array([10, 20, 30])

    np.testing.assert_array_equal(integer_values, expected)

    print("Dtype conversion test passed")


def test_integer_overflow():

    print("\n========== INTEGER OVERFLOW ==========")

    small_array = np.array([120], dtype=np.int8)
    second_array = np.array([20], dtype=np.int8)

    result = small_array + second_array

    print("Original Dtype:", small_array.dtype)
    print("Result with int8:", result)

    np.testing.assert_array_equal(
        result,
        np.array([-116], dtype=np.int8)
    )

    large_array = small_array.astype(np.int64)
    second_array = second_array.astype(np.int64)

    corrected_result = large_array + second_array

    print("Corrected Result:", corrected_result)

    expected = np.array([140], dtype=np.int64)

    np.testing.assert_array_equal(corrected_result, expected)

    print("Integer overflow handling test passed")


def test_nan_propagation():

    print("\n========== NAN PROPAGATION ==========")

    values = np.array([10.0, 20.0, np.nan, 40.0])

    normal_mean = np.mean(values)
    safe_mean = np.nanmean(values)

    print("Original Array:", values)
    print("Mean with NaN:", normal_mean)
    print("Mean excluding NaN:", safe_mean)

    if not np.isnan(normal_mean):
        raise AssertionError("Expected NaN was not produced")

    np.testing.assert_allclose(safe_mean, 70.0 / 3.0)

    print("NaN propagation test passed")


def safe_divide(numerator, denominator):

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full(numerator.shape, np.nan)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0
    )

    return result


def test_zero_denominator():

    print("\n========== ZERO DENOMINATOR ==========")

    revenue = np.array([100.0, 200.0, 300.0])
    quantity = np.array([10.0, 0.0, 15.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        unsafe_result = revenue / quantity

    print("Unsafe Division Result:", unsafe_result)

    safe_result = safe_divide(revenue, quantity)

    print("Safe Division Result:", safe_result)

    expected = np.array([10.0, np.nan, 20.0])

    np.testing.assert_allclose(
        safe_result,
        expected,
        equal_nan=True
    )

    print("Zero denominator test passed")


# 11.4 Views, Copies and Runtime

def test_views_and_copies():

    print("\n========== VIEWS AND COPIES ==========")

    original = np.array([10, 20, 30, 40, 50])

    view_array = original[1:4]
    copy_array = original[1:4].copy()

    print("Original Array:", original)
    print("View Array:", view_array)
    print("Copy Array:", copy_array)

    print("View Shares Memory:", np.shares_memory(original, view_array))
    print("Copy Shares Memory:", np.shares_memory(original, copy_array))

    view_array[0] = 200

    print("Original After View Modification:", original)

    copy_array[1] = 300

    print("Original After Copy Modification:", original)

    expected = np.array([10, 200, 30, 40, 50])

    np.testing.assert_array_equal(original, expected)

    print("Views and copies test passed")


def test_runtime():

    print("\n========== RUNTIME COMPARISON ==========")

    values = np.arange(1, 100001, dtype=np.int64)

    start_time = time.perf_counter()

    numpy_total = np.sum(values)

    numpy_time = time.perf_counter() - start_time

    start_time = time.perf_counter()

    loop_total = 0

    for value in values:
        loop_total += value

    loop_time = time.perf_counter() - start_time

    print("NumPy Total:", numpy_total)
    print("Loop Total:", loop_total)

    print("NumPy Runtime:", numpy_time)
    print("Loop Runtime:", loop_time)

    np.testing.assert_equal(numpy_total, loop_total)

    print("Runtime comparison test passed")


# 11.5 Testing and Validation

def test_validation_assertions():

    print("\n========== VALIDATION ASSERTIONS ==========")

    actual_array = np.array([10, 20, 30, 40])
    expected_array = np.array([10, 20, 30, 40])

    np.testing.assert_array_equal(
        actual_array,
        expected_array
    )

    print("Array equality assertion passed")

    actual_values = np.array([10.001, 20.002, 30.003])
    expected_values = np.array([10.0, 20.0, 30.0])

    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=0.001,
        atol=0.01
    )

    print("Approximate equality assertion passed")


def test_dataset():

    print("\n========== FRESHFLOW DATASET VALIDATION ==========")

    dataset_path = get_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError("Cleaned FreshFlow dataset not found")

    data = np.genfromtxt(
        dataset_path,
        delimiter=",",
        skip_header=1,
        dtype=float,
        encoding="utf-8",
        ndmin=2
    )

    print("Dataset Shape:", data.shape)
    print("Dataset Dtype:", data.dtype)

    assert data.ndim == 2, "Dataset must be two-dimensional"
    assert data.shape[0] > 0, "Dataset must contain records"
    assert data.shape[1] == 12, "Dataset must contain 12 columns"

    print("FreshFlow dataset validation passed")


# 12.1 Manual Calculation Verification

def test_manual_calculations():

    print("\n========== 12.1 MANUAL CALCULATION VERIFICATION ==========")

    dataset_path = get_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError("Cleaned FreshFlow dataset not found")

    data = np.genfromtxt(
        dataset_path,
        delimiter=",",
        skip_header=1,
        dtype=float,
        encoding="utf-8",
        ndmin=2
    )

    sample_data = data[:5]

    transaction_id = sample_data[:, 0]
    quantity = sample_data[:, 5]
    unit_price = sample_data[:, 6]
    unit_cost = sample_data[:, 7]
    discount_rate = sample_data[:, 8]

    gross_sales = quantity * unit_price
    discount_amount = gross_sales * discount_rate
    net_revenue = gross_sales - discount_amount
    cogs = quantity * unit_cost
    gross_profit = net_revenue - cogs

    expected_gross_sales = np.array([
        83.96, 150.54, 157.17, 101.42, 243.30
    ])

    expected_discount = np.array([
        4.198, 30.108, 31.434, 15.213, 36.495
    ])

    expected_net_revenue = np.array([
        79.762, 120.432, 125.736, 86.207, 206.805
    ])

    expected_cogs = np.array([
        31.96, 85.86, 61.00, -17.98, 136.64
    ])

    expected_gross_profit = np.array([
        47.802, 34.572, 64.736, 104.187, 70.165
    ])

    print("Transaction IDs:", transaction_id)

    np.testing.assert_allclose(
        gross_sales,
        expected_gross_sales
    )

    np.testing.assert_allclose(
        discount_amount,
        expected_discount
    )

    np.testing.assert_allclose(
        net_revenue,
        expected_net_revenue
    )

    np.testing.assert_allclose(
        cogs,
        expected_cogs
    )

    np.testing.assert_allclose(
        gross_profit,
        expected_gross_profit
    )

    print("\nManual and NumPy Calculations:")

    for i in range(len(sample_data)):

        print("\nTransaction ID:", int(transaction_id[i]))
        print("Gross Sales:", gross_sales[i])
        print("Discount Amount:", discount_amount[i])
        print("Net Revenue:", net_revenue[i])
        print("COGS:", cogs[i])
        print("Gross Profit:", gross_profit[i])

        if unit_cost[i] <= 0:
            print("Profit Status: Invalid unit cost - exclude from valid profit analysis")
        else:
            print("Profit Status: Valid for profit calculation")

    print("\nAll five sample calculations matched successfully")


# Main Function

def main():

    print("==========================================")
    print("PHASE 11 - DEBUG AND OPTIMISE")
    print("==========================================")

    test_shape_mismatch()
    test_broadcasting()
    test_dtype_conversion()
    test_integer_overflow()
    test_nan_propagation()
    test_zero_denominator()
    test_views_and_copies()
    test_runtime()
    test_validation_assertions()
    test_dataset()

    print("\n==========================================")
    print("PHASE 12 - FINAL INDEPENDENT ASSIGNMENT")
    print("==========================================")

    test_manual_calculations()

    print("\n==========================================")
    print("PHASE 12.1 VERIFICATION COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()
