import seaborn as sns
import matplotlib.pyplot as plt

# Plot Histograms for Numerical Variables
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], bins=20, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()

# Bar Charts for Categorical Variables
categorical_features = ['Gender', 'Province', 'CoverType']
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=data[feature], order=data[feature].value_counts().index)
    plt.title(f'Bar Chart of {feature}')
    plt.show()
