import pandas as pd
import sys

def validate_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # 1. Expected columns
        expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
        if set(df.columns) != expected_columns:
            print("❌ Column mismatch!")
            print("Found:", set(df.columns))
            print("Expected:", expected_columns)
            return False

        # 2. Null check
        if df.isnull().values.any():
            print("❌ Null values found in data")
            return False

        # 3. Range checks (based on IRIS dataset knowledge)
        if not ((df["sepal_length"] > 0) & (df["sepal_width"] > 0) &
                (df["petal_length"] > 0) & (df["petal_width"] > 0)).all():
            print("❌ Some feature values are non-positive")
            return False

        print("✅ Data validation passed!")
        return True

    except Exception as e:
        print(f"❌ Validation failed due to error: {e}")
        return False

if __name__ == "__main__":
    success = validate_data("data/iris.csv")
    sys.exit(0 if success else 1)

