import pandas as pd
import matplotlib.pyplot as plt

def detect_outliers(input_file, output_dir):
    """Detects outliers in numerical data."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

    print("Generating box plots for outlier detection...")
    for col in numeric_cols:
        plt.figure()
        data.boxplot(column=col)
        plt.title(f"Outliers in {col}")
        plt.savefig(f"{output_dir}/{col}_boxplot.png")
        plt.close()

    print("Outlier detection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier Detection Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()

    detect_outliers(args.input_file, args.output_dir)
