import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import plot_importance

def model_evaluation(input_dir="data/input", model_dir="models", output_dir="data/metrics"):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # ---------------- Load Pre-Saved Data ----------------
        print("==> Loading pre-saved train/test data...")
        
        # Regression data
        X_test_reg = pd.read_csv(os.path.join(input_dir, "X_test_reg.csv"), index_col=0, parse_dates=True)
        y_test_reg = pd.read_csv(os.path.join(input_dir, "y_test_reg.csv"), index_col=0, parse_dates=True).squeeze()

        # Classification data
        X_test_clf = pd.read_csv(os.path.join(input_dir, "X_test_clf.csv"), index_col=0, parse_dates=True)
        y_test_clf = pd.read_csv(os.path.join(input_dir, "y_test_clf.csv"), index_col=0, parse_dates=True).squeeze()

        # ---------------- Regression Evaluation ----------------
        reg_model_path = os.path.join(model_dir, "xgb_regressor.pkl")
        with open(reg_model_path, "rb") as f:
            reg_model = pickle.load(f)
        print(f"Loaded regression model: {reg_model_path}")

        y_pred_reg = reg_model.predict(X_test_reg)
        rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        print(f"Regression RMSE: {rmse:.4f}")

        # Regression feature importance
        plt.figure(figsize=(10,6))
        importance = reg_model.get_booster().get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Gain': list(importance.values())
        }).sort_values(by='Gain', ascending=False).head(20)
        plt.barh(importance_df['Feature'], importance_df['Gain'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel('Gain')
        plt.title('Top 20 Feature Importances (Regression)')
        plt.tight_layout()
        plt.show()

        # Regression prediction plot
        plt.figure(figsize=(12,4))
        plt.plot(y_test_reg.index, y_test_reg, label='Actual Return', linewidth=2)
        plt.plot(y_test_reg.index, y_pred_reg, label='Predicted Return', linestyle='--', color='orange')
        plt.title('Next-Day Return Prediction')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # ---------------- Classification Evaluation ----------------
        clf_model_path = os.path.join(model_dir, "xgb_classifier.pkl")
        with open(clf_model_path, "rb") as f:
            clf_model = pickle.load(f)
        print(f"Loaded classification model: {clf_model_path}")

        y_pred_clf = clf_model.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, y_pred_clf)
        report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
        print(f"Classification Accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test_clf, y_pred_clf)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down','Up'])
        disp.plot(cmap='Blues')
        plt.title('Trend Prediction Confusion Matrix')
        plt.show()

        # Classification feature importance
        fig, ax = plt.subplots(figsize=(10,6))
        plot_importance(clf_model, importance_type='gain', max_num_features=20, height=0.5, ax=ax)
        ax.set_title('Top 20 Feature Importances (Classification)')
        plt.tight_layout()
        plt.show()

        # ---------------- Save Metrics ----------------
        metrics_dict = {
            'regression': {'rmse': rmse},
            'classification': {
                'accuracy': report['accuracy'],
                'precision_up': report['1']['precision'],
                'recall_up': report['1']['recall'],
                'f1_up': report['1']['f1-score'],
                'precision_down': report['0']['precision'],
                'recall_down': report['0']['recall'],
                'f1_down': report['0']['f1-score']
            }
        }

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved: {metrics_path}")

        return metrics_dict

    except Exception as e:
        print(f"Error in model_evaluation: {e}")
        raise e

if __name__ == "__main__":
    model_evaluation()