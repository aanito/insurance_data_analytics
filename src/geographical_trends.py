import pandas as pd
import matplotlib.pyplot as plt

def geographical_trends(input_file, output_dir):
    """Analyzes geographical trends in the dataset."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    if "Province" not in data.columns:
        print("Error: 'Province' column not found in data.")
        return

    print("Analyzing geographical trends...")
    province_summary = data.groupby("Province")[["TotalPremium", "TotalClaims"]].mean()
    province_summary.to_csv(f"{output_dir}/province_summary.csv")

    province_summary.plot(kind="bar", figsize=(10, 6))
    plt.title("Average Premiums and Claims by Province")
    plt.xlabel("Province")
    plt.ylabel("Value")
    plt.savefig(f"{output_dir}/province_trends.png")
    plt.close()

    print("Geographical trends analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geographical Trends Analysis Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    args = parser.parse_args()

    geographical_trends(args.input_file, args.output_dir)
