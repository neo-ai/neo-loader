import pytest
from unittest.mock import MagicMock
from neo_loader.onnx_model_loader import ONNXModelLoader


class MockedOpError(Exception):
    pass

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.onnx_model_loader.relay", mock_relay)
    return mock_relay


@pytest.fixture
def patch_onnx(monkeypatch):
    mock_torch = MagicMock()
    monkeypatch.setattr("neo_loader.onnx_model_loader.onnx", mock_torch)
    return mock_torch

@pytest.fixture
def patch_op_error(monkeypatch):
    monkeypatch.setattr("neo_loader.onnx_model_loader.OpError", MockedOpError)


def test_onnx(patch_relay, patch_onnx):
    patch_relay.frontend.from_onnx.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["test.onnx"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_onnx.load.assert_called()
    patch_relay.frontend.from_onnx.assert_called()


def test_onnx_no_onnx_file_error(patch_relay, patch_onnx):
    model_artifacts = ["test.blah"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: No .onnx file found for ONNX model." in str(err)


def test_onnx_multiple_onnx_file_errors(patch_relay, patch_onnx):
    model_artifacts = ["test1.onnx", "test2.onnx"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .onnx file is allowed for ONNX models.' in str(err)

def test_onnx_load_model_exception(patch_relay, patch_onnx):
    patch_onnx.load.side_effect = Exception("Model load error.")
    model_artifacts = ["test.onnx"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load ONNX model.' in str(err)


def test_onnx_relay_frontend_op_error(patch_relay, patch_onnx, patch_op_error):
    patch_relay.frontend.from_onnx.side_effect = MockedOpError("Dummy Error")
    model_artifacts = ["test.onnx"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    with pytest.raises(MockedOpError) as err:
        loader.load_model()
    assert 'Dummy Error' in str(err)


def test_onnx_relay_frontend_exception(patch_relay, patch_onnx, patch_op_error):
    patch_relay.frontend.from_onnx.side_effect = Exception("Dummy Error")
    model_artifacts = ["test.onnx"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = ONNXModelLoader(model_artifacts, data_shape)
    with pytest.raises(Exception) as err:
        loader.load_model()
    assert 'InputConfiguration: TVM cannot convert ONNX model.' in str(err)
