import pandas as pd
import matplotlib.pyplot as plt

def bivariate_analysis(input_file, output_dir):
    """Performs bivariate analysis and generates visualizations."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    print("Calculating correlations...")
    correlation_matrix = data.corr()
    correlation_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")

    print("Generating scatter plots for key relationships...")
    plt.scatter(data['TotalPremium'], data['TotalClaims'])
    plt.title("TotalPremium vs TotalClaims")
    plt.xlabel("TotalPremium")
    plt.ylabel("TotalClaims")
    plt.savefig(f"{output_dir}/totalpremium_vs_totalclaims_scatter.png")
    plt.close()

    print("Bivariate analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bivariate Analysis Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    args = parser.parse_args()

    bivariate_analysis(args.input_file, args.output_dir)
