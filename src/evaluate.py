import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load the trained model
model = joblib.load("models/model.joblib")

# Load the dataset
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

# Use same split as training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict
preds = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
report = classification_report(y_test, preds)

# Print to terminal
print("Accuracy:", acc)
print("F1 Score:", f1)
print("Classification Report:\n", report)

# Save metrics for CML report
with open("metrics.txt", "w") as f:
    f.write(f"## Model Evaluation Metrics\n\n")
    f.write(f"**Accuracy**: {acc:.4f}\n\n")
    f.write(f"**F1 Score**: {f1:.4f}\n\n")
    f.write("**Classification Report:**\n\n")
    f.write("```\n" + report + "\n```")

