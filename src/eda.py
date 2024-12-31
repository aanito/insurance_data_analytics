import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def perform_eda(input_file):
    """Performs exploratory data analysis on the given file."""
    print(f"Loading processed data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # Descriptive statistics
    print("Generating descriptive statistics...")
    desc_stats = data.describe()
    print(desc_stats)
    
    # Visualizations
    print("Generating visualizations...")
    sns.histplot(data['TotalPremium'], bins=20, kde=True)
    plt.title("Histogram of TotalPremium")
    plt.savefig("reports/total_premium_histogram.png")
    print("Saved histogram of TotalPremium.")

    sns.scatterplot(data=data, x='TotalPremium', y='TotalClaims', hue='PostalCode')
    plt.title("TotalPremium vs TotalClaims by PostalCode")
    plt.savefig("reports/total_premium_vs_total_claims.png")
    print("Saved scatter plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Script")
    parser.add_argument("input_file", type=str, help="Path to processed data file (.csv)")
    args = parser.parse_args()

    perform_eda(args.input_file)
