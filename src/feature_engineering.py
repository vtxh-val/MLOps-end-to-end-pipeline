import os
import numpy as np
import pandas as pd

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def feature_engineering(input_path="data/processed/cleaned.csv", output_path="data/features/engineered.csv"):
    try:
        # Ensure model directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load data
        print(f"==> Loading data from {input_path}...")
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"Data shape: {df.shape}")

        # 1. Lag features
        for lag in [1, 2, 3]:
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # 2. Returns
        df['Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # 3. Rolling windows
        for window in [3, 7, 14]:
            df[f"MA{window}"] = df['Close'].rolling(window).mean()
            df[f"Volatility{window}"] = df['Close'].rolling(window).std()
            df[f"HighMax{window}"] = df['High'].rolling(window).max()
            df[f"LowMin{window}"] = df['Low'].rolling(window).min()

        # 4. Date features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['IsFriday'] = (df.index.dayofweek == 4).astype(int)

        # 5. RSI
        df['RSI14'] = compute_RSI(df['Close'])

        # Targets
        df['Target_Return'] = df['Return'].shift(-1)
        df['Target_Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        df = df.dropna()
        df.to_csv(output_path)
        print(f"Feature-engineered data saved: {output_path}")
        return df

    except Exception as e:
        print(f"Error in feature_engineering: {e}")
        raise e

if __name__ == "__main__":
    feature_engineering()