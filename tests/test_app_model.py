import os

def test_if_model_file_exists():
    assert os.path.exists("models/revenue_model.pkl")