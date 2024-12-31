import pandas as pd
import argparse

def preprocess_data(input_file, output_file):
    """Preprocesses raw data and saves a clean CSV file."""
    # Load the data
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    if data.empty:
        print("The input file is empty or could not be read. Please check the file.")
        return

    
    # Handle missing values
    print("Handling missing values...")
    data.fillna(method='ffill', inplace=True)
    
    # Convert dates
    print("Converting dates...")
    data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
    
    # Save processed data
    print(f"Saving cleaned data to {output_file}...")
    data.to_csv(output_file, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("input_file", type=str, help="Path to raw data file (.txt)")
    parser.add_argument("output_file", type=str, help="Path to save processed data (.csv)")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file)
