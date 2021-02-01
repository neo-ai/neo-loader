import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from neo_loader.tflite_model_loader import TFLiteModelLoader


class MockedOpError(Exception):
    pass

@pytest.fixture
def patch_tvm(monkeypatch):
    mock_tvm = MagicMock()
    monkeypatch.setattr("neo_loader.tflite_model_loader.tvm", mock_tvm)
    return mock_tvm

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.tflite_model_loader.tvm.relay", mock_relay)
    return mock_relay


@pytest.fixture
def patch_op_error(monkeypatch):
    monkeypatch.setattr("neo_loader.tflite_model_loader.OpError", MockedOpError)
    return MockedOpError


@pytest.fixture
def patch_tflite_model_helper(monkeypatch):
    mock_tflite_model_helper = MagicMock()
    # input_dtypes_dict = PropertyMock(return_value = {'test': 'float32'})
    # type(mock_tflite_model_helper).input_dtypes_dict = input_dtypes_dict

    def mock_input_dtypes_dict(instance):
        return {'test': 'float32'}

    monkeypatch.setattr("neo_loader.tflite_model_loader.TFLiteModelHelper", mock_tflite_model_helper)
    # monkeypatch.setattr("neo_loader.tflite_model_loader.TFLiteModelHelper", "input_dtypes_dict", mock_input_dtypes_dict)

    return mock_tflite_model_helper


def test_tflite_without_model_file():
    model_artifacts = ["test.blah"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = TFLiteModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: No valid TFLite model found in input file.' in str(err)


def test_tflite_with_multiple_model_files():
    model_artifacts = ["test1.tflite", "test2.tflite"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = TFLiteModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .tflite file is allowed for TFLite models.' in str(err)


def test_tflite_model_helper_load_model_exception(patch_tflite_model_helper):
    patch_tflite_model_helper.return_value.load_model.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.tflite"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = TFLiteModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load TFLite model: Dummy Exception' in str(err)


def test_tflite_model_helper_extract_input_and_output_tensors_exception(patch_tflite_model_helper):
    patch_tflite_model_helper.return_value.extract_input_and_output_tensors.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.tflite"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = TFLiteModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'Unable to infer tensor data type for all inputs/outputs: Dummy Exception' in str(err)


def test_tflite_relay_exception(patch_tvm, patch_relay, patch_tflite_model_helper, patch_op_error):
    patch_relay.frontend.from_tflite.side_effect = Exception("Dummy Error")
    model_artifacts = ["test.tflite"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = TFLiteModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert "InputConfiguration: TVM cannot convert TFLite model." in str(err)
