import pandas as pd
import argparse

def summarize_data(input_file, output_file):
    """Summarizes the dataset with descriptive statistics."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    print("Generating descriptive statistics...")
    summary = data.describe(include="all").transpose()
    
    print(f"Saving summary to {output_file}...")
    summary.to_csv(output_file)
    print("Data summarization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Summarization Script")
    parser.add_argument("input_file", type=str, help="Path to the input data file (.txt)")
    parser.add_argument("output_file", type=str, help="Path to save the summary (.csv)")
    args = parser.parse_args()

    summarize_data(args.input_file, args.output_file)
