import pandas as pd
import matplotlib.pyplot as plt

def creative_visualizations(input_file, output_dir):
    """Generates creative visualizations of key insights."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    print("Generating visualizations...")
    plt.figure(figsize=(10, 6))
    data.groupby("Province")["TotalPremium"].mean().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Premium Distribution by Province")
    plt.savefig(f"{output_dir}/premium_distribution_pie.png")
    plt.close()

    print("Creative visualizations complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative Visualization Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()

    creative_visualizations(args.input_file, args.output_dir)
