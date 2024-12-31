import pandas as pd
import scipy.stats as stats
import argparse
import numpy as np

def select_metrics(data):
    """
    Select key metrics for hypothesis testing.
    """
    metrics = ['TotalClaims', 'TotalPremium']
    print(f"Selected metrics for analysis: {metrics}")
    return metrics

def segment_data(data, group_col, feature_value_a, feature_value_b):
    """
    Segment data into control (Group A) and test (Group B) groups.
    """
    print(f"Segmenting data on {group_col}...")
    group_a = data[data[group_col] == feature_value_a]
    group_b = data[data[group_col] == feature_value_b]
    return group_a, group_b

def perform_statistical_tests(group_a, group_b, metrics):
    """
    Perform statistical tests to evaluate null hypotheses.
    """
    results = {}
    for metric in metrics:
        print(f"Performing t-test for {metric}...")
        stat, p_value = stats.ttest_ind(group_a[metric].dropna(), group_b[metric].dropna(), equal_var=False)
        results[metric] = {'t_statistic': stat, 'p_value': p_value}
    return results

def analyze_and_report(results, alpha=0.05):
    """
    Analyze test results and report conclusions.
    """
    report = {}
    for metric, result in results.items():
        print(f"Analyzing {metric}...")
        p_value = result['p_value']
        reject_null = p_value < alpha
        report[metric] = {
            "p_value": p_value,
            "reject_null": reject_null,
            "conclusion": "Significant difference" if reject_null else "No significant difference"
        }
    return report

def main(input_file, output_file, group_col, feature_value_a, feature_value_b):
    # Load the data
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # Select metrics
    metrics = select_metrics(data)
    
    # Segment data
    group_a, group_b = segment_data(data, group_col, feature_value_a, feature_value_b)
    
    # Perform statistical tests
    test_results = perform_statistical_tests(group_a, group_b, metrics)
    
    # Analyze and report
    report = analyze_and_report(test_results)
    
    # Save report
    print(f"Saving report to {output_file}...")
    with open(output_file, 'w') as f:
        for metric, result in report.items():
            f.write(f"Metric: {metric}\n")
            f.write(f"P-value: {result['p_value']}\n")
            f.write(f"Reject Null: {result['reject_null']}\n")
            f.write(f"Conclusion: {result['conclusion']}\n")
            f.write("-" * 40 + "\n")
    print("Report generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Hypothesis Testing Script")
    parser.add_argument("input_file", type=str, help="Path to preprocessed data file (.csv)")
    parser.add_argument("output_file", type=str, help="Path to save the report")
    parser.add_argument("group_col", type=str, help="Column to group data on (e.g., Province, ZipCode, Gender)")
    parser.add_argument("feature_value_a", type=str, help="Value representing Group A")
    parser.add_argument("feature_value_b", type=str, help="Value representing Group B")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.group_col, args.feature_value_a, args.feature_value_b)


# import pandas as pd
# import scipy.stats as stats
# import argparse

# def perform_ab_testing(input_file, output_file):
#     """Performs A/B testing to evaluate risk and margin differences."""
#     print(f"Loading data from {input_file}...")
#     data = pd.read_csv(input_file, delimiter="|")

#     results = []

#     # 1. Risk differences across provinces
#     print("Performing A/B test: Risk differences across provinces...")
#     provinces = data.groupby('Province')['TotalClaims']
#     f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in provinces])
#     results.append({"Test": "Risk differences across provinces", "p-value": p_val})

#     # 2. Risk differences between zip codes
#     print("Performing A/B test: Risk differences between zip codes...")
#     zipcodes = data.groupby('PostalCode')['TotalClaims']
#     f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in zipcodes])
#     results.append({"Test": "Risk differences between zip codes", "p-value": p_val})

#     # 3. Margin differences between zip codes
#     print("Performing A/B test: Margin differences between zip codes...")
#     zipcodes = data.groupby('PostalCode')['TotalPremium']
#     f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in zipcodes])
#     results.append({"Test": "Margin differences between zip codes", "p-value": p_val})

#     # 4. Risk differences between genders
#     print("Performing A/B test: Risk differences between genders...")
#     genders = data.groupby('Gender')['TotalClaims']
#     f_stat, p_val = stats.f_oneway(*[group.dropna() for name, group in genders])
#     results.append({"Test": "Risk differences between genders", "p-value": p_val})

#     # Save results
#     print(f"Saving test results to {output_file}...")
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_file, index=False)
#     print("A/B Testing complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="A/B Testing Script")
#     parser.add_argument("input_file", type=str, help="Path to preprocessed data file (.txt)")
#     parser.add_argument("output_file", type=str, help="Path to save test results (.csv)")
#     args = parser.parse_args()

#     perform_ab_testing(args.input_file, args.output_file)
