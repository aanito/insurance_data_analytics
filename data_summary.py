import pandas as pd

# Load Data
data = pd.read_csv('insurance_data.csv', delimiter='|')

# Descriptive Statistics
numerical_features = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
desc_stats = data[numerical_features].describe()
print("Descriptive Statistics:\n", desc_stats)

# Data Structure
print("Data Types:\n", data.dtypes)

# Confirm date formatting
data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
print("Sample Data after Date Conversion:\n", data[['TransactionMonth']].head())
