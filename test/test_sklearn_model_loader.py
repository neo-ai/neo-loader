import pytest
import sys
from unittest.mock import MagicMock
sys.modules["joblib"] = MagicMock()
sys.modules["sagemaker_sklearn_extension"] = MagicMock()
from neo_loader.sklearn_model_loader import SklearnModelLoader

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.sklearn_model_loader.relay", mock_relay)
    return mock_relay

def test_sklearn_invalid_num_dims():
    model_artifacts = ["model.joblib"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = SklearnModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: InputShape for Sklearn model must have two dimensions, but got 4.' in str(err)

def test_sklearn_invalid_num_cols():
    model_artifacts = ["model.joblib"]
    data_shape = {"input": [-1, -1]}
    loader = SklearnModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: InputShape for Sklearn model must have a static value for the second dimension, equal to the number of input columns or features.' in str(err)

def test_sklearn_inverse_transform_valid_num_cols(patch_relay):
    patch_relay.frontend.from_auto_ml.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["model.joblib"]
    data_shape = {"input": [-1, -1]}
    loader = SklearnModelLoader(model_artifacts, data_shape)
    loader.update_func_name("inverse_transform")
    loader.load_model()
