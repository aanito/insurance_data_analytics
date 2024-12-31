import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import argparse

def train_model(input_file):
    """Trains a Random Forest model to predict TotalPremium."""
    print(f"Loading processed data from {input_file}...")
    data = pd.read_csv(input_file)

    # Feature Selection
    print("Selecting features and target...")
    features = ['Cylinders', 'cubiccapacity', 'kilowatts', 'SumInsured']
    target = 'TotalPremium'
    
    X = data[features]
    y = data[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save Model
    print("Saving model...")
    pd.to_pickle(model, "models/random_forest_model.pkl")
    print("Model training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("input_file", type=str, help="Path to processed data file (.csv)")
    args = parser.parse_args()

    train_model(args.input_file)
