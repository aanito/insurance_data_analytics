import os
import subprocess

def setup_environment():
    """Sets up the Python environment for the project."""
    print("Installing required packages...")
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "scikit-learn", "xgboost", "pytest", "argparse"
    ]
    subprocess.run(["pip", "install"] + packages)
    print("Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
