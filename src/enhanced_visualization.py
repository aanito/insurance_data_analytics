import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def generate_visualizations(input_file, output_dir):
    """Generates advanced visualizations for EDA insights."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    print("Creating visualizations...")

    # 1. Premiums by Province
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Province', y='TotalPremium', data=data)
    plt.title('Distribution of Premiums by Province')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/premiums_by_province.png")
    plt.close()

    # 2. Claims vs. Premiums
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='Province', data=data)
    plt.title('Claims vs Premiums by Province')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/claims_vs_premiums.png")
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation = data[['TotalPremium', 'TotalClaims', 'SumInsured']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    print("Visualizations saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization Script")
    parser.add_argument("input_file", type=str, help="Path to preprocessed data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()

    generate_visualizations(args.input_file, args.output_dir)
