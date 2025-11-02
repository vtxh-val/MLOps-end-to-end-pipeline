import os
import pickle
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

def model_building(input_path="data/features/engineered.csv", output_dir="data/input", model_dir="models"):
    try:
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        print(f"==> Loading data from {input_path}...")
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"Data shape: {df.shape}")

        # Time-based split
        train_size = int(0.7 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # ---------------- Regression Model ----------------
        X_train_reg = train_df.drop(columns=['Target_Return', 'Target_Trend'])
        y_train_reg = train_df['Target_Return']
        X_test_reg = test_df.drop(columns=['Target_Return', 'Target_Trend'])
        y_test_reg = test_df['Target_Return']


        reg_model = XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=5,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
        reg_model.fit(X_train_reg, y_train_reg)

        # Save regression model
        reg_model_path = os.path.join(model_dir, "xgb_regressor.pkl")
        with open(reg_model_path, "wb") as f:
            pickle.dump(reg_model, f)
        print(f"Regression model saved: {reg_model_path}")

        # Save data
        X_train_reg.to_csv(os.path.join(output_dir, "X_train_reg.csv"))
        y_train_reg.to_csv(os.path.join(output_dir, "y_train_reg.csv"))
        X_test_reg.to_csv(os.path.join(output_dir, "X_test_reg.csv"))
        y_test_reg.to_csv(os.path.join(output_dir, "y_test_reg.csv"))

        # ---------------- Classification Model ----------------
        X_train_clf = train_df.drop(columns=['Target_Return', 'Target_Trend'])
        y_train_clf = train_df['Target_Trend']
        X_test_clf = test_df.drop(columns=['Target_Return', 'Target_Trend'])
        y_test_clf = test_df['Target_Trend']

        clf_model = XGBClassifier(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=5,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
        clf_model.fit(X_train_clf, y_train_clf)

        # Save classification model
        clf_model_path = os.path.join(model_dir, "xgb_classifier.pkl")
        with open(clf_model_path, "wb") as f:
            pickle.dump(clf_model, f)
        print(f"Classification model saved: {clf_model_path}")

        # Save data
        X_train_clf.to_csv(os.path.join(output_dir, "X_train_clf.csv"))
        y_train_clf.to_csv(os.path.join(output_dir, "y_train_clf.csv"))
        X_test_clf.to_csv(os.path.join(output_dir, "X_test_clf.csv"))
        y_test_clf.to_csv(os.path.join(output_dir, "y_test_clf.csv"))

        print("\nAll models and datasets successfully saved!")
        print(f"Data directory: {output_dir}")
        print(f"Models directory: {model_dir}")

        return reg_model_path, clf_model_path

    except Exception as e:
        print(f"Error in model_building: {e}")
        raise e

if __name__ == "__main__":
    model_building()