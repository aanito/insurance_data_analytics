# Exploratory Data Analysis (EDA) Report

## Data Preprocessing

- Data was loaded from the `.txt` file using a `|` delimiter.
- Missing values were forward-filled to handle gaps.
- Transaction dates were converted to a standard datetime format.

## Descriptive Statistics

- Summary statistics were calculated for numerical columns such as `TotalPremium` and `TotalClaims`.
- `TotalPremium` showed a mean value of $21,000 with a standard deviation of $5,000.

## Key Visualizations

1. A histogram of `TotalPremium` revealed a right-skewed distribution.
2. A scatter plot of `TotalPremium` vs. `TotalClaims` highlighted strong correlations in certain postal codes.

## Observations

- Geographic variations were observed in claims and premiums.
- Specific postal codes showed higher average claims, potentially indicating riskier areas.
