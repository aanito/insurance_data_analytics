import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import argparse

def tune_hyperparameters(input_file, output_file):
    """Tunes hyperparameters for predicting premiums."""
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, delimiter="|")

    # Prepare features and target
    X = data[['Cylinders', 'cubiccapacity', 'kilowatts', 'SumInsured']].fillna(0)
    y = data['TotalPremium'].fillna(0)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Initializing Random Forest model...")
    model = RandomForestRegressor(random_state=42)

    print("Defining parameter grid...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    print("Performing grid search...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R2 score: {grid_search.best_score_}")

    # Save best parameters
    with open(output_file, 'w') as f:
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best R2 Score: {grid_search.best_score_}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
    parser.add_argument("input_file", type=str, help="Path to preprocessed data file (.txt)")
    parser.add_argument("output_file", type=str, help="Path to save best parameters (.txt)")
    args = parser.parse_args()

    tune_hyperparameters(args.input_file, args.output_file)
