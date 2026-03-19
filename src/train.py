from preprocessing import load_and_clean_data
from feature_engineering import add_features

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# Load data
df = load_and_clean_data("data/train.csv")
df = add_features(df)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save BOTH model + columns
joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("Model + columns saved!")