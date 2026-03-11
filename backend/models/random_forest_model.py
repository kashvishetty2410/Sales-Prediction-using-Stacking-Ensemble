# Random Forest Model for Sales Prediction

# -----------------------------
# Import libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree


# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("data/train.csv")

print("First 5 rows of dataset:")
print(data.head())


# -----------------------------
# Basic preprocessing
# -----------------------------

# Drop unnecessary columns (IDs and high-cardinality columns that add noise)
data = data.drop(columns=[
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name",
    "Country",        # Only one value - United States
    "Postal Code",    # High cardinality - adds noise
    "City",           # High cardinality - adds noise
    "State"           # High cardinality - adds noise
])

print("\nColumns after dropping unnecessary ones:")
print(data.columns)


# Convert date columns
data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)
data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True)


# Extract useful features
# Days to ship - important signal for sales prediction
data["days_to_ship"] = (data["Ship Date"] - data["Order Date"]).dt.days

data["Order_Year"] = data["Order Date"].dt.year
data["Order_Month"] = data["Order Date"].dt.month

data["Ship_Year"] = data["Ship Date"].dt.year
data["Ship_Month"] = data["Ship Date"].dt.month


# Drop original date columns
data = data.drop(columns=["Order Date", "Ship Date"])

print("\nDataset after preprocessing:")
print(data.head())


# -----------------------------
# Encode categorical columns using one-hot encoding (consistent with LR)
# -----------------------------

# One-hot encode categorical columns
cat_cols = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
cat_cols = [c for c in cat_cols if c in data.columns]

print(f"[INFO] One-hot encoding columns: {cat_cols}")
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

print("\nDataset after encoding:")
print(data.head())


# -----------------------------
# Separate features and target
# -----------------------------
X = data.drop("Sales", axis=1)
y = data["Sales"]

print("\nFeature columns:")
print(X.columns)

print("\nTarget variable: Sales")


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# -----------------------------
# Train Random Forest Model
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# -----------------------------
# Predictions
# -----------------------------
y_pred = rf_model.predict(X_test)


# -----------------------------
# Evaluation Metrics
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# -----------------------------
# GRAPH 1: Actual vs Predicted
# -----------------------------
plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales (Random Forest)")

plt.show()


# -----------------------------
# GRAPH 2: Feature Importance
# -----------------------------
importance = rf_model.feature_importances_

feature_importance = pd.Series(importance, index=X.columns)

plt.figure(figsize=(8,5))

feature_importance.sort_values().plot(kind="barh")

plt.title("Feature Importance - Random Forest")

plt.xlabel("Importance Score")

plt.show()


# -----------------------------
# GRAPH 3: One Decision Tree
# -----------------------------
plt.figure(figsize=(20,10))

plot_tree(
    rf_model.estimators_[0],
    feature_names=X.columns,
    filled=True,
    max_depth=3
)

plt.title("One Decision Tree from Random Forest")

plt.show()


# -----------------------------
# Generate predictions
# -----------------------------
train_preds = rf_model.predict(X_train)
test_preds = rf_model.predict(X_test)


# -----------------------------
# Save predictions for stacking
# -----------------------------
import os

os.makedirs("outputs/predictions", exist_ok=True)

pd.DataFrame({"rf_pred": train_preds}).to_csv(
    "outputs/predictions/rf_predictions_train.csv", index=False
)

pd.DataFrame({"rf_pred": test_preds}).to_csv(
    "outputs/predictions/rf_predictions_test.csv", index=False
)

print("\nRandom Forest predictions saved for stacking.")