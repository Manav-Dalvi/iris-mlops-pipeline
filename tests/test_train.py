import os

def test_model_file_created():
    assert os.path.exists("models/model.joblib"), "Trained model not found!"
