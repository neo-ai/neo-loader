import pytest
from unittest.mock import MagicMock
from neo_loader.pytorch_model_loader import PyTorchModelLoader


class MockedOpError(Exception):
    pass


@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.pytorch_model_loader.relay", mock_relay)
    return mock_relay


@pytest.fixture
def patch_torch(monkeypatch):
    mock_torch = MagicMock()
    monkeypatch.setattr("neo_loader.pytorch_model_loader.torch", mock_torch)
    return mock_torch


def test_pytorch(patch_relay, patch_torch):
    patch_relay.frontend.from_pytorch.return_value = MagicMock(), MagicMock()
    model_artifacts = ["test.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_relay.frontend.from_pytorch.assert_called()
    patch_torch.jit.load.return_value.float.return_value.eval.assert_called()
    patch_torch.jit.trace.return_value.float.return_value.eval.assert_called()


def test_pytorch_no_pth_file_error(patch_relay, patch_torch):
    model_artifacts = ["test.blah"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: No pth file found for PyTorch model." in str(err)


def test_pytorch_multiple_pth_file_errors(patch_relay, patch_torch):
    model_artifacts = ["test1.pth", "test2.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .pth file is allowed for PyTorch models.' in str(err)


def test_pytorch_jit_load_exception(patch_relay, patch_torch):
    patch_relay.frontend.from_pytorch.return_value = MagicMock(), MagicMock()
    patch_torch.jit.load.side_effect = RuntimeError
    model_artifacts = ["test.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_torch.load.assert_called()
    patch_torch.jit.trace.return_value.float.return_value.eval.assert_called()


def test_pytorch_load_exception(patch_relay, patch_torch):
    patch_torch.jit.load.side_effect = RuntimeError
    patch_torch.load.side_effect = RuntimeError
    model_artifacts = ["test.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load PyTorch model' in str(err)


def test_pytorch_jit_trace_exception(patch_relay, patch_torch):
    patch_torch.jit.trace.side_effect = RuntimeError
    model_artifacts = ["test.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load PyTorch model' in str(err)
    patch_torch.jit.load.return_value.float.return_value.eval.assert_called()


def test_pytorch_relay_exception(patch_relay, patch_torch):
    patch_relay.frontend.from_pytorch.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.pth"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = PyTorchModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: TVM cannot convert the PyTorch model.' in str(err)
