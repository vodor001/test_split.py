"""Script for odtp component to create train/test split.

Author: Jan Aarts

Usage: python train_test_split.py data.csv instructions.json
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

def load_data(file_path):
    if Path(file_path).suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError("The provided file is not a CSV file.")

def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def main():

    file_path = sys.argv[1]
    instructions = sys.argv[2]
    try:
        df = load_data(file_path)
        train_df, test_df = split_data(df)
        
        train_output_path = "train.csv"
        test_output_path = "test.csv"
        
        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)
        
        print(f"Train and test datasets saved as {train_output_path} and {test_output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()