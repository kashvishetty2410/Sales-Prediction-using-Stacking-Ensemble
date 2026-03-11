import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import os

OUTPUTS_DIR = 'outputs/predictions'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def main():
    data = pd.read_csv('data/train.csv')

    # Drop unnecessary columns (IDs and high-cardinality columns that add noise)
    data = data.drop(columns=[
        "Row ID", "Order ID", "Customer ID", "Customer Name",
        "Product ID", "Product Name", "Country", "Postal Code",
        "City", "State"  # High cardinality - adds noise
    ])

    data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)
    data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True)

    # Days to ship - important signal for sales prediction
    data["days_to_ship"] = (data["Ship Date"] - data["Order Date"]).dt.days

    data["Order_Year"] = data["Order Date"].dt.year
    data["Order_Month"] = data["Order Date"].dt.month
    data["Ship_Year"] = data["Ship Date"].dt.year
    data["Ship_Month"] = data["Ship Date"].dt.month

    data = data.drop(columns=["Order Date", "Ship Date"])

    # One-hot encode categorical columns (consistent with LR)
    categorical_cols = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Handle any remaining NaN values
    data = data.fillna(0)

    X = data.drop("Sales", axis=1)
    y = data["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr_model = SVR(kernel="rbf")
    svr_model.fit(X_train_scaled, y_train)

    train_predictions = svr_model.predict(X_train_scaled)
    test_predictions = svr_model.predict(X_test_scaled)

    pd.DataFrame({'svr_pred': train_predictions}).to_csv(
        os.path.join(OUTPUTS_DIR, 'svr_predictions_train.csv'), index=False
    )
    pd.DataFrame({'svr_pred': test_predictions}).to_csv(
        os.path.join(OUTPUTS_DIR, 'svr_predictions_test.csv'), index=False
    )

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(((y_test - test_predictions) ** 2).mean())
    r2 = r2_score(y_test, test_predictions)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()
