import pandas as pd
import matplotlib.pyplot as plt

def assess_data_quality(input_file, output_file):
    """Checks for missing values and other data quality issues."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    print("Assessing data quality...")
    missing_values = data.isnull().sum()
    data_types = data.dtypes
    quality_report = pd.DataFrame({
        "Missing Values": missing_values,
        "Data Type": data_types
    })

    print(f"Saving quality report to {output_file}...")
    quality_report.to_csv(output_file)
    print("Data quality assessment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Assessment Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_file", type=str, help="Path to save the quality report (.csv)")
    args = parser.parse_args()

    assess_data_quality(args.input_file, args.output_file)
