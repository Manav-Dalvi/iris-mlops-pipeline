from joblib import load

def test_model_can_predict():
    model = load("models/model.joblib")
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    assert prediction is not None and len(prediction) == 1
