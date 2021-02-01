import pytest
from unittest.mock import MagicMock
from neo_loader.darknet_model_loader import DarkNetModelLoader


class MockedOpError(Exception):
    pass

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.darknet_model_loader.relay", mock_relay)
    return mock_relay

@pytest.fixture
def patch_darknet(monkeypatch):
    mock_darknet = MagicMock()
    mock_darknet.__darknetffi__ = MagicMock()
    monkeypatch.setattr("neo_loader.darknet_model_loader.darknet", mock_darknet)
    return mock_darknet

@pytest.fixture
def patch_op_error(monkeypatch):
    monkeypatch.setattr("neo_loader.darknet_model_loader.OpError", MockedOpError)
    return MockedOpError

def test_darknet(patch_relay, patch_darknet):
    model_artifacts = ["test.cfg", "test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_darknet.__darknetffi__.dlopen.assert_called()
    patch_relay.frontend.from_darknet.assert_called()


def test_darknet_no_config_file_error():
    model_artifacts = ["test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: No .cfg file found for DarkNet model." in str(err)

def test_darknet_no_weight_file_error():
    model_artifacts = ["test.cfg"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: No .weights file found for DarkNet model." in str(err)

def test_darknet_multiple_config_file_errors():
    model_artifacts = ["test1.cfg", "test2.cfg", "test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .cfg file is allowed for DarkNet models.' in str(err)

def test_darknet_multiple_weight_file_errors():
    model_artifacts = ["test.cfg", "test1.weights", "test2.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .weights file is allowed for DarkNet models.' in str(err)

def test_darknet_load_model_exception(patch_relay, patch_darknet):
    patch_darknet.__darknetffi__.dlopen.side_effect = Exception("Model load error.")
    model_artifacts = ["test.cfg", "test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Unable to load DarkNet model.' in str(err)

def test_darknet_relay_frontend_op_error(patch_darknet, patch_relay, patch_op_error):
    patch_relay.frontend.from_darknet.side_effect = patch_op_error("Dummy OpError")
    model_artifacts = ["test.cfg", "test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(patch_op_error) as err:
        loader.load_model()
    assert 'Dummy OpError' in str(err)


def test_darknet_relay_frontend_exception(patch_darknet, patch_relay, patch_op_error):
    patch_relay.frontend.from_darknet.side_effect = Exception("Dummy Error")
    model_artifacts = ["test.cfg", "test.weights"]
    data_shape = {"data": [1, 3, 224, 224]}
    loader = DarkNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: TVM cannot convert DarkNet model.' in str(err)
