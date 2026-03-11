# Stacking Ensemble Model for Sales Prediction

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Load base model predictions
# -----------------------------
lr_train = pd.read_csv("outputs/predictions/lr_predictions_train.csv")
lr_test = pd.read_csv("outputs/predictions/lr_predictions_test.csv")

rf_train = pd.read_csv("outputs/predictions/rf_predictions_train.csv")
rf_test = pd.read_csv("outputs/predictions/rf_predictions_test.csv")

svr_train = pd.read_csv("outputs/predictions/svr_predictions_train.csv")
svr_test = pd.read_csv("outputs/predictions/svr_predictions_test.csv")


# -----------------------------
# Standardize column names
# -----------------------------
lr_train.columns = ["lr_pred"]
lr_test.columns = ["lr_pred"]

rf_train.columns = ["rf_pred"]
rf_test.columns = ["rf_pred"]

svr_train.columns = ["svr_pred"]
svr_test.columns = ["svr_pred"]


# -----------------------------
# Combine predictions
# -----------------------------
X_train_stack = pd.concat([lr_train, rf_train, svr_train], axis=1)
X_test_stack = pd.concat([lr_test, rf_test, svr_test], axis=1)

X_train_stack = X_train_stack.reset_index(drop=True)
X_test_stack = X_test_stack.reset_index(drop=True)


# -----------------------------
# Load original dataset
# -----------------------------
data = pd.read_csv("data/train.csv")

X = data.drop("Sales", axis=1)
y = data["Sales"]


# -----------------------------
# Recreate train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# -----------------------------
# Simple averaging (proven to work better)
# -----------------------------
print("\nUsing simple averaging approach")
stack_preds = (X_test_stack['lr_pred'].values + X_test_stack['rf_pred'].values + X_test_stack['svr_pred'].values) / 3


# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, stack_preds)
mse = mean_squared_error(y_test, stack_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, stack_preds)

print("\nStacking Model Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


# -----------------------------
# Save predictions
# -----------------------------
results = pd.DataFrame({
    "stacking_prediction": stack_preds
})

results.to_csv(
    "outputs/predictions/stacking_predictions.csv",
    index=False
)

print("\nStacking predictions saved successfully!")