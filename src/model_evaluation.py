import os
import pickle
import json
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from xgboost import plot_importance

load_dotenv()  
repo_onwer = os.getenv("REPO_ONWER") 
repo_name = os.getenv("REPO_NAME")
tracking_url = os.getenv("TRACKING_URL") 

def model_evaluation(input_dir="data/input", model_dir="models", output_dir="data/metrics"):
    os.makedirs(output_dir, exist_ok=True)

    # --- MLflow setup ---
    dagshub.init(repo_owner=repo_onwer, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment("Stock_Trend_Prediction")

    with mlflow.start_run(run_name="xgboost_reg_clf_evaluation"):

        # Load data
        X_test_reg = pd.read_csv(os.path.join(input_dir, "X_test_reg.csv"), index_col=0, parse_dates=True)
        y_test_reg = pd.read_csv(os.path.join(input_dir, "y_test_reg.csv"), index_col=0, parse_dates=True).squeeze()
        X_test_clf = pd.read_csv(os.path.join(input_dir, "X_test_clf.csv"), index_col=0, parse_dates=True)
        y_test_clf = pd.read_csv(os.path.join(input_dir, "y_test_clf.csv"), index_col=0, parse_dates=True).squeeze()

        # Load regression model
        reg_model_path = os.path.join(model_dir, "xgb_regressor.pkl")
        with open(reg_model_path, "rb") as f:
            reg_model = pickle.load(f)

        # Evaluate regression
        y_pred_reg = reg_model.predict(X_test_reg)
        rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Log regression model
        # mlflow.xgboost.log_model(reg_model, "regression_model")
        # mlflow.xgboost.log_model(xgb_model=reg_model, name="reg_model_json", model_format="json")

        # Load classification model
        clf_model_path = os.path.join(model_dir, "xgb_classifier.pkl")
        with open(clf_model_path, "rb") as f:
            clf_model = pickle.load(f)

        # Evaluate classification
        y_pred_clf = clf_model.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, y_pred_clf)
        report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_up", report['1']['precision'])
        mlflow.log_metric("recall_up", report['1']['recall'])
        mlflow.log_metric("f1_up", report['1']['f1-score'])

        # Log classification model
        # mlflow.xgboost.log_model(clf_model, "classification_model")
        # mlflow.xgboost.log_model(xgb_model=clf_model, name="clf_model_json", model_format="json")

        # Save metrics locally too
        metrics_dict = {
            'regression': {'rmse': rmse},
            'classification': {
                'accuracy': acc,
                'precision_up': report['1']['precision'],
                'recall_up': report['1']['recall'],
                'f1_up': report['1']['f1-score']
            }
        }

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact(metrics_path)

        print(f"Metrics saved to {metrics_path}")
        print("Experiment logged to MLflow!")

if __name__ == "__main__":
    model_evaluation()