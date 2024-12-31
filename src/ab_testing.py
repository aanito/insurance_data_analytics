import pandas as pd
import scipy.stats as stats
import argparse

def perform_ab_testing(input_file, output_file):
    """Performs A/B testing to evaluate risk and margin differences."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    results = []

    # 1. Risk differences across provinces
    print("Performing A/B test: Risk differences across provinces...")
    provinces = data.groupby('Province')['TotalClaims']
    f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in provinces])
    results.append({"Test": "Risk differences across provinces", "p-value": p_val})

    # 2. Risk differences between zip codes
    print("Performing A/B test: Risk differences between zip codes...")
    zipcodes = data.groupby('PostalCode')['TotalClaims']
    f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in zipcodes])
    results.append({"Test": "Risk differences between zip codes", "p-value": p_val})

    # 3. Margin differences between zip codes
    print("Performing A/B test: Margin differences between zip codes...")
    zipcodes = data.groupby('PostalCode')['TotalPremium']
    f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in zipcodes])
    results.append({"Test": "Margin differences between zip codes", "p-value": p_val})

    # 4. Risk differences between genders
    print("Performing A/B test: Risk differences between genders...")
    genders = data.groupby('Gender')['TotalClaims']
    f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in genders])
    results.append({"Test": "Risk differences between genders", "p-value": p_val})

    # Save results
    print(f"Saving test results to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print("A/B Testing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Testing Script")
    parser.add_argument("input_file", type=str, help="Path to preprocessed data file (.txt)")
    parser.add_argument("output_file", type=str, help="Path to save test results (.csv)")
    args = parser.parse_args()

    perform_ab_testing(args.input_file, args.output_file)
