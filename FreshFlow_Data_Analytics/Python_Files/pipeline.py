
import numpy as np
import subprocess
import sys
from pathlib import Path


# PHASE 10 - BUILD A PIPELINE


# Project paths

project_folder = Path(__file__).resolve().parents[1]

python_folder = project_folder / "Python_Files"


# 10.1 - NUMPY ARRAY OPERATIONS

def demonstrate_array_operations():

    print("\n========== NUMPY ARRAY OPERATIONS ==========")

    first_array = np.array([10, 20, 30])
    second_array = np.array([40, 50, 60])

    # np.concatenate()

    combined_array = np.concatenate((
        first_array,
        second_array
    ))

    print("\nConcatenate:")
    print(combined_array)

    # np.stack()

    stacked_array = np.stack((
        first_array,
        second_array
    ))

    print("\nStack:")
    print(stacked_array)

    # np.vstack()

    vertical_array = np.vstack((
        first_array,
        second_array
    ))

    print("\nVertical Stack:")
    print(vertical_array)

    # np.hstack()

    horizontal_array = np.hstack((
        first_array,
        second_array
    ))

    print("\nHorizontal Stack:")
    print(horizontal_array)

    # np.split()

    split_array = np.array([10, 20, 30, 40, 50, 60])

    equal_parts = np.split(split_array, 3)

    print("\nSplit:")
    print(equal_parts)

    # np.array_split()

    unequal_parts = np.array_split(split_array, 4)

    print("\nArray Split:")
    print(unequal_parts)

    # np.cumsum()

    cumulative_sum = np.cumsum(first_array)

    print("\nCumulative Sum:")
    print(cumulative_sum)

    # np.cumprod()

    cumulative_product = np.cumprod(first_array)

    print("\nCumulative Product:")
    print(cumulative_product)


# 10.2 - CREATING REUSABLE FUNCTIONS

def calculate_total(array):

    total = np.sum(array)

    return total


def calculate_average(array):

    if len(array) == 0:
        return np.nan

    average = np.mean(array)

    return average


def calculate_cumulative_revenue(revenue_array):

    cumulative_revenue = np.cumsum(revenue_array)

    return cumulative_revenue


def combine_arrays(array_one, array_two):

    combined = np.concatenate((
        array_one,
        array_two
    ))

    return combined


def split_array(array, number_of_parts):

    divided_array = np.array_split(
        array,
        number_of_parts
    )

    return divided_array


# 10.3 - TESTING REUSABLE FUNCTIONS

def test_reusable_functions():

    print("\n========== REUSABLE FUNCTIONS ==========")

    sample_revenue = np.array([
        1000,
        1500,
        2000,
        2500,
        3000
    ])

    total_revenue = calculate_total(sample_revenue)

    average_revenue = calculate_average(sample_revenue)

    cumulative_revenue = calculate_cumulative_revenue(
        sample_revenue
    )

    print("\nSample Revenue:")
    print(sample_revenue)

    print("\nTotal Revenue:")
    print(total_revenue)

    print("\nAverage Revenue:")
    print(average_revenue)

    print("\nCumulative Revenue:")
    print(cumulative_revenue)

    first_part = np.array([10, 20, 30])

    second_part = np.array([40, 50, 60])

    combined = combine_arrays(
        first_part,
        second_part
    )

    print("\nCombined Arrays:")
    print(combined)

    divided = split_array(
        combined,
        3
    )

    print("\nDivided Arrays:")
    print(divided)


# 10.4 - DEFINING PIPELINE MODULES

pipeline_modules = [
    "data_analysis.py",
    "data_cleaning.py",
    "metrics_analysis.py",
    "descriptive_analysis.py",
    "group_comparison.py",
    "daily_trend.py",
    "discount_analysis.py",
    "anomaly_analysis.py",
    "scenario_analysis.py"
]


# 10.5 - CREATING A REUSABLE MODULE RUNNER

def run_module(module_name):

    module_path = python_folder / module_name

    if not module_path.exists():

        print("\nModule Not Found:", module_name)

        return False

    print("\n==========================================")
    print("Running Module:", module_name)
    print("==========================================")

    try:

        subprocess.run(
            [
                sys.executable,
                str(module_path)
            ],
            cwd=python_folder,
            check=True
        )

        print("\nCompleted Successfully:", module_name)

        return True

    except subprocess.CalledProcessError:

        print("\nModule Failed:", module_name)

        return False


# 10.6 - BUILDING THE COMPLETE PIPELINE

def run_pipeline():

    print("\n==========================================")
    print("FRESHFLOW DATA ANALYSIS PIPELINE")
    print("==========================================")

    successful_modules = []
    failed_modules = []

    for module_name in pipeline_modules:

        result = run_module(module_name)

        if result:

            successful_modules.append(module_name)

        else:

            failed_modules.append(module_name)

            print("\nPipeline stopped due to module failure.")

            break

    print("\n========== PIPELINE SUMMARY ==========")

    print("Successful Modules:", len(successful_modules))

    print("Failed Modules:", len(failed_modules))

    print("\nSuccessful Module List:")

    for module in successful_modules:

        print(module)

    if failed_modules:

        print("\nFailed Module List:")

        for module in failed_modules:

            print(module)

        print("\nPipeline Execution Failed.")

        return False

    print("\nAll Modules Executed Successfully.")

    return True


# 10.7 - MAIN EXECUTION

def main():

    demonstrate_array_operations()

    test_reusable_functions()

    pipeline_success = run_pipeline()

    print("\n==========================================")

    if pipeline_success:

        print("PHASE 10 PIPELINE COMPLETED")

    else:

        print("PHASE 10 PIPELINE NEEDS CORRECTION")

    print("==========================================")


if __name__ == "__main__":

    main()
