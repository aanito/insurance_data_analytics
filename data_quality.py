# Check Missing Values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Remove or Impute Missing Values
data.fillna(method='ffill', inplace=True)
print("Data Quality Check: Missing Values Addressed.")
