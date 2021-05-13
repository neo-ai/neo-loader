import pytest
from unittest.mock import MagicMock
from neo_loader.mxnet_model_loader import MxNetModelLoader

class MockedOpError(Exception):
    pass

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.mxnet_model_loader.relay", mock_relay)
    return mock_relay

@pytest.fixture
def patch_mxnet(monkeypatch):
    mock_mxnet = MagicMock()
    monkeypatch.setattr("neo_loader.mxnet_model_loader.mx", mock_mxnet)
    return mock_mxnet


@pytest.fixture
def patch_tvm_error(monkeypatch):
    monkeypatch.setattr("neo_loader.mxnet_model_loader.OpError", MockedOpError)


def test_mxnet(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called()
    patch_mxnet.ndarray.load.assert_called()
    patch_relay.frontend.from_mxnet.assert_called()


def test_mxnet_with_multiple_keys_in_dict(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["test.params", "test.json"]
    data_shape = {"data": [1, 3, 224, 224], "softmax_label": [1]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called()
    patch_mxnet.ndarray.load.assert_called()
    patch_relay.frontend.from_mxnet.assert_called()


def test_mxnet_load_model_op_error(patch_relay, patch_mxnet, patch_tvm_error):
    patch_relay.frontend.from_mxnet.side_effect = MockedOpError

    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(MockedOpError):
        loader.load_model()


def test_mxnet_load_model_exception(patch_relay, patch_mxnet, patch_tvm_error):
    patch_relay.frontend.from_mxnet.side_effect = Exception("Some TVM Error")
    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError, match="InputConfiguration: TVM can't convert the MXNet model."):
        loader.load_model()


def test_mxnet_ndarray_invalid_params_exception(patch_mxnet):
    def mock_mxnet_ndarray_load(param_file):
        return {"arg:foo": param_file, "bad:aux": param_file}

    patch_mxnet.ndarray.load = mock_mxnet_ndarray_load

    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    msg = "InputConfiguration: Framework can't load the MXNet model: Please use HybridBlock.export()"
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert msg in str(errinfo.value)


def test_mxnet_ndarray_exception(patch_mxnet):
    patch_mxnet.ndarray.load.side_effect = Exception("Bad model params.")

    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: Framework can't load the MXNet model: Bad model params." in str(errinfo.value)


def test_mxnet_symbol_exception(patch_mxnet):
    patch_mxnet.symbol.load.side_effect = Exception("Bad model json.")

    model_artifacts = ["test.params", "test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: Framework can't load the MXNet model: Bad model json." in str(errinfo.value)


def test_mxnet_invalid_model_artifact_without_json_file():
    model_artifacts = ["test.params"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: No symbol file found for MXNet model." in str(errinfo.value)


def test_mxnet_invalid_model_artifact_with_multiple_json_files():
    model_artifacts = ["test.params", "test1-symbol.json", "test2-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: Only one symbol file is allowed for MXNet model." in str(errinfo.value)


def test_mxnet_invalid_model_artifact_without_params_file():
    model_artifacts = ["test-symbol.json"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: No parameter file found for MXNet model." in str(errinfo.value)
    

def test_mxnet_mismatched_symbol_and_params_prefix():
    model_artifacts = ["model1-symbol.json", "model2-0010.params"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as errinfo:
        loader.load_model()
    assert "InputConfiguration: No parameter file found for MXNet model: model1-symbol.json" in str(errinfo.value)
    assert "Please make sure the prefix of params file match the prefix of symbol file." in str(errinfo.value)


def test_mxnet_model_artifact_with_multiple_params_files(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["resnet-18-symbol.json", "resnet-18-0000.params", "resnet-18-0042.params"]
    data_shape = {"data": [1, 3, 224, 224]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called()
    patch_mxnet.ndarray.load.assert_called()

    patch_relay.frontend.from_mxnet.assert_called()


def test_mxnet_support_cross_validation(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["model0-0001.params", "model0-symbol.json", "model1-0002.params", "model1-symbol.json"]
    data_shape = {"data": [1, -1]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called_with("model0-symbol.json")
    patch_mxnet.ndarray.load.assert_called_with("model0-0001.params")
    patch_relay.frontend.from_mxnet.assert_called()


def test_mxnet_validate_cross_validation_params(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["model0-symbol.json", "model1-0002.params", "model1-symbol.json"]
    data_shape = {"data": [1, -1]}

    with pytest.raises(RuntimeError) as errinfo:
        loader = MxNetModelLoader(model_artifacts, data_shape)
        loader.load_model()
    assert "InputConfiguration: No parameter file found for MXNet model: model0-symbol.json" in str(errinfo.value)
    
    
def test_mxnet_support_multi_cross_validation_params(patch_relay, patch_mxnet):
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["model0-symbol.json", "model0-0001.params", "model0-00042.params"]
    data_shape = {"data": [1, -1]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called_with("model0-symbol.json")
    patch_mxnet.ndarray.load.assert_called_with("model0-00042.params")
    patch_relay.frontend.from_mxnet.assert_called()


def test_mxnet_support_cross_validation_edge_case(patch_relay, patch_mxnet):
    # allow this since this is potential AP cross-validation
    patch_relay.frontend.from_mxnet.return_value.__iter__.return_value = MagicMock(), MagicMock()
    model_artifacts = ["model0-0001.params", "model0-symbol.json", "resnet-18-0001.params", "resnet-18-symbol.json"]
    data_shape = {"data": [1, -1]}

    loader = MxNetModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_mxnet.symbol.load.assert_called_with("model0-symbol.json")
    patch_mxnet.ndarray.load.assert_called_with("model0-0001.params")
    patch_relay.frontend.from_mxnet.assert_called()