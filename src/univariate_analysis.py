import pandas as pd
import matplotlib.pyplot as plt

def univariate_analysis(input_file, output_dir):
    """Generates univariate analysis visualizations."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    print("Generating histograms for numerical columns...")
    for col in numeric_cols:
        plt.figure()
        data[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/{col}_histogram.png")
        plt.close()

    print("Generating bar charts for categorical columns...")
    for col in categorical_cols:
        plt.figure()
        data[col].value_counts().plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(f"{output_dir}/{col}_bar_chart.png")
        plt.close()

    print("Univariate analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Univariate Analysis Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()

    univariate_analysis(args.input_file, args.output_dir)
