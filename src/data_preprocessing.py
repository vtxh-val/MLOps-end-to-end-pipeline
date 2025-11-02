import os
import pandas as pd

def data_preprocessing(input_path="data/raw/raw.csv", output_path="data/processed/cleaned.csv"):
    try:
        # Ensure model directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load data
        print(f"==> Loading data from {input_path}...")
        df = pd.read_csv(input_path, header=[0,1], index_col=0, parse_dates=True)
        print(f"Data shape: {df.shape}")

        # Flatten multi-level columns
        df.columns = [col[0] if col[1]=='' else col[0] for col in df.columns]
        df = df.apply(pd.to_numeric, errors='coerce') 

        print("\n==> Checking for missing values and duplicates...")
        df = df.dropna().drop_duplicates()
        df = df.sort_index()
        df.to_csv(output_path)
        print(f"Cleaned data saved: {output_path}")
        return df

    except Exception as e:
        print(f"Error in data_preprocessing: {e}")
        raise e

if __name__ == "__main__":
    data_preprocessing()