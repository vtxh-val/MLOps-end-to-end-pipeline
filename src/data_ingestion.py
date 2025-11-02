import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def data_ingestion(ticker="AAPL", start="2024-01-01", data_dir="data/raw"):
    try:
        os.makedirs(data_dir, exist_ok=True)
        end = datetime.today().strftime("%Y-%m-%d")
        
        print(f"==> Downloading {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.index = pd.to_datetime(df.index)
        
        raw_path = os.path.join(data_dir, "raw.csv")
        df.to_csv(raw_path)
        print(f"Raw data saved: {raw_path}")

        return df
    
    except Exception as e:
        print(f"Error in data_ingestion: {e}")
        raise e

if __name__ == "__main__":
    data_ingestion()