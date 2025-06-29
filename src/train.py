import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    df = pd.read_csv("data/iris.csv")

    # Features and target
    X = df.drop(columns=["species"])
    y = df["species"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    print(" Model trained and saved to models/model.joblib")

if __name__ == "__main__":
    train_model()
