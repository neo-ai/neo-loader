import sys
import pytest
from unittest.mock import MagicMock
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()
sys.modules["tensorflow.keras.layers"] = MagicMock()
sys.modules["tvm"] = MagicMock()
from neo_loader.keras_model_loader import KerasModelLoader


@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.keras_model_loader.relay", mock_relay)
    return mock_relay


@pytest.fixture
def patch_keras_load_model(monkeypatch):
    mock_keras_load_model = MagicMock()
    monkeypatch.setattr("neo_loader.keras_model_loader.load_model", mock_keras_load_model)
    monkeypatch.setattr("neo_loader.keras_model_loader.InputLayer", MagicMock)
    return mock_keras_load_model


@pytest.fixture
def patch_relay_keras_frontend(patch_relay):
    def mock_relay_frontend_from_keras(model, shape=[]):
        # mod, params
        return MagicMock(), MagicMock()
    patch_relay.frontend.from_keras = mock_relay_frontend_from_keras
    return patch_relay


def test_keras(patch_relay_keras_frontend, patch_keras_load_model):
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    loader.load_model()
    model_objects = loader.model_objects
    assert len(model_objects) == 2
    assert isinstance(model_objects[0], MagicMock)
    assert isinstance(model_objects[1], MagicMock)


def test_keras_no_model_file_error(patch_relay_keras_frontend, patch_keras_load_model):
    model_artifacts = ["test.blah"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: No h5 file provided" in str(err)


def test_keras_multiple_model_files_error(patch_relay_keras_frontend, patch_keras_load_model):
    model_artifacts = ["test1.h5", "test2.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert f'InputConfiguration: Multiple h5 files provided, {model_artifacts}, when only one is allowed.' in str(err)


def test_keras_load_model_exception(patch_relay_keras_frontend, patch_keras_load_model):
    patch_keras_load_model.side_effect = Exception
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(Exception) as err:
        loader.load_model()
    assert 'InputConfiguration: Unable to load provided Keras model.' in str(err)


def test_keras_invalid_input_key(patch_relay_keras_frontend, patch_keras_load_model):
    layer = MagicMock()
    layer.name = "bad_name"
    patch_keras_load_model.return_value.layers.__iter__.return_value = [layer]
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Model contains input layer (bad_name) not specified in data_shape.' in str(err)


def test_keras_invalid_input_batch_size(patch_relay_keras_frontend, patch_keras_load_model):
    layer = MagicMock()
    layer.name = "input"
    layer.input_shape = [(2, 3, 224, 224)]
    patch_keras_load_model.return_value.layers.__iter__.return_value = [layer]
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Input input has wrong batch size in Input Shape dictionary.' in str(err)


def test_keras_invalid_input_shape(patch_relay_keras_frontend, patch_keras_load_model):
    layer = MagicMock()
    layer.name = "input"
    layer.input_shape = [(1, 3, 224, 224)]
    patch_keras_load_model.return_value.layers.__iter__.return_value = [layer]
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 224, 224, 3]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Input input has wrong shape in Input Shape dictionary.' in str(err)


def test_keras_wrong_input_shape(patch_relay_keras_frontend, patch_keras_load_model):
    layer = MagicMock()
    layer.name = "input"
    layer.input_shape = [(1, 3, 224)]
    patch_keras_load_model.return_value.layers.__iter__.return_value = [layer]
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 224, 224, 3]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Input input has wrong shape in Input Shape dictionary.' in str(err)


def test_keras_relay_runtime_error(patch_relay, patch_keras_load_model):
    patch_relay.frontend.from_keras.side_effect = RuntimeError("Dummy error.")
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'Dummy error.' in str(err)


def test_keras_relay_exception(patch_relay, patch_keras_load_model):
    patch_relay.frontend.from_keras.side_effect = Exception("Dummy Exception.")
    model_artifacts = ["test.h5"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = KerasModelLoader(model_artifacts, data_shape)
    with pytest.raises(Exception) as err:
        loader.load_model()
    assert "InputConfiguration: TVM cannot convert the Keras model." in str(err)
