import pytest
import pickle
from pathlib import Path

from unittest.mock import MagicMock
from neo_loader.xgboost_model_loader import XGBoostModelLoader, RestrictedUnpickler

@pytest.fixture
def patch_treelite(monkeypatch):
    mock_treelite = MagicMock()
    monkeypatch.setattr("neo_loader.xgboost_model_loader.treelite", mock_treelite)
    return mock_treelite


@pytest.fixture
def patch_xgboost_core(monkeypatch):
    mock_xgboost_core = MagicMock()
    monkeypatch.setattr("neo_loader.xgboost_model_loader.xgboost.core", mock_xgboost_core)
    return mock_xgboost_core


@pytest.fixture
def patch_restricted_unpickler(monkeypatch):
    mock_restricted_unpickler = MagicMock()
    monkeypatch.setattr(RestrictedUnpickler, 'load', mock_restricted_unpickler)
    return mock_restricted_unpickler


@pytest.fixture
def patch_file_open(monkeypatch):
    mock_open = MagicMock()
    monkeypatch.setattr("builtins.open", mock_open)
    return mock_open

def test_xgboost(patch_treelite, patch_restricted_unpickler, patch_xgboost_core, patch_file_open):
    model_artifacts = ["xgboost_model"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = XGBoostModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_treelite.Model.from_xgboost.assert_called()


def test_xgboost_unpickling_error(patch_treelite, patch_restricted_unpickler, patch_xgboost_core, patch_file_open):
    patch_restricted_unpickler.side_effect = pickle.UnpicklingError
    model_artifacts = ["xgboost_model"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = XGBoostModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_treelite.Model.load.assert_called()


def test_xgboost_no_model_file_error(patch_treelite, patch_xgboost_core):
    model_artifacts = []
    data_shape = {"input": [1, 3, 224, 224]}
    with pytest.raises(RuntimeError) as err:
        loader = XGBoostModelLoader(model_artifacts, data_shape)
    assert 'InputConfiguration: No model file found.' in str(err)


def test_xgboost_multiple_model_file_error(patch_treelite, patch_xgboost_core):
    model_artifacts = ["model_a", "model_b"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = XGBoostModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Invalid XGBoost model: only one XGBoost model file is allowed.' in str(err)


def test_xgboost_treelite_exception(patch_treelite, patch_xgboost_core):
    patch_treelite.Model.load.side_effect = Exception("Dummy Exception")
    model_artifacts = ["model"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = XGBoostModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Treelite failed to convert XGBoost model.' in str(err)


def test_xgboost_support_cross_validation(patch_treelite, patch_restricted_unpickler, patch_xgboost_core, patch_file_open):
    model_artifacts = ["xgboost-model-0", "xgboost-model-1"]
    data_shape = {"data": [-1, -1]}
    loader = XGBoostModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_file_open.assert_called_with(Path("xgboost-model-0"), "rb")
    patch_treelite.Model.from_xgboost.assert_called()