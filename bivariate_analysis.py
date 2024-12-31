import numpy as np

# Correlation Matrix for Numerical Features
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter Plot: TotalPremium vs. TotalClaims by ZipCode
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='TotalPremium', y='TotalClaims', hue='PostalCode', palette='viridis', alpha=0.6)
plt.title('Scatter Plot of TotalPremium vs TotalClaims by ZipCode')
plt.show()
