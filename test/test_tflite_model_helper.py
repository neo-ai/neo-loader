import os
import sys
import pytest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock, MagicMock, mock_open

sys.modules['tflite'] = Mock()
sys.modules['tflite.Tensor'] = Mock()
sys.modules['tflite.Model'] = Mock()
sys.modules['tflite.TensorType'] = Mock()

from neo_loader.helpers.tflite_model_helper import TFLiteModelHelper

@pytest.fixture
def patch_tflite_model_helper(monkeypatch):
    def mock_get_supported_tflite_input_tensor_type(instance):
        return [0,3]

    monkeypatch.setattr(TFLiteModelHelper, "get_supported_tflite_input_tensor_type", mock_get_supported_tflite_input_tensor_type)
    monkeypatch.setattr(TFLiteModelHelper, "TFLITE_TENSOR_TYPE_TO_DTYPE", {0 : "float32", 3: "uint8"})


def test_tflite_model_helper_float32(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    sys.modules['tflite.Model'] = Mock()

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        with patch("tflite.Model.Model.GetRootAsModel") as mocked_model:
                instance = mocked_model.return_value
                test_tflite_model_helper.load_model()

                mock_subgraph = Mock()
                mock_subgraph.InputsAsNumpy.return_value = [1]
                mock_subgraph.OutputsAsNumpy.return_value = [2]

                mock_tensors = Mock()
                mock_subgraph.Tensors.return_value = mock_tensors
                mock_tensors.Name.side_effect = [b"test_input", b"test_output", b"test_input", b"test_output"]
                mock_tensors.Type.return_value = 0
                mock_tensors.ShapeAsNumpy.side_effect = [np.array([1, 3, 224, 224]), np.array([1, 10])]
                instance.Subgraphs.return_value = mock_subgraph

                test_tflite_model_helper.extract_input_and_output_tensors(user_shape_dict={"test_input" : [1,3,224,224]})
                out = test_tflite_model_helper.get_metadata()

                assert len(out["Inputs"]) == 1
                assert len(out["Outputs"]) == 1
                assert out["Inputs"][0]['name'] == 'test_input'
                assert out["Outputs"][0]['name'] == 'test_output'
                assert out["Inputs"][0]['dtype'] == 'float32'
                assert out["Outputs"][0]['dtype'] == 'float32'
                assert out["Inputs"][0]['shape'] == [1, 3, 224, 224]
                assert out["Outputs"][0]['shape'] == [1, 10]

    mock_file.assert_called_with(Path("test.tflite").resolve(), "rb")

def test_tflite_model_helper_unsupported_input_data_type(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    sys.modules['tflite.Model'] = Mock()

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        with patch("tflite.Model.Model.GetRootAsModel") as mocked_model:
                instance = mocked_model.return_value
                test_tflite_model_helper.load_model()

                mock_subgraph = Mock()
                mock_subgraph.InputsAsNumpy.return_value = [1]
                mock_subgraph.OutputsAsNumpy.return_value = [2]

                mock_tensors = Mock()
                mock_subgraph.Tensors.return_value = mock_tensors
                mock_tensors.Name.side_effect = [b"test_input", b"test_output", b"test_input", b"test_output"]
                mock_tensors.Type.return_value = 1
                mock_tensors.ShapeAsNumpy.side_effect = [np.array([1, 3, 224, 224]), np.array([1, 10])]
                instance.Subgraphs.return_value = mock_subgraph
                try:
                    test_tflite_model_helper.extract_input_and_output_tensors(user_shape_dict={"test_input" : [1,3,224,224]})
                except Exception as ex:
                    assert str(ex) == "Unsupported input data type for input test_input with tflite tensor type 1"


    mock_file.assert_called_with(Path("test.tflite").resolve(), "rb")


def test_tflite_model_helper_wrong_user_input_names(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    sys.modules['tflite.Model'] = Mock()

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        with patch("tflite.Model.Model.GetRootAsModel") as mocked_model:
                instance = mocked_model.return_value
                test_tflite_model_helper.load_model()

                mock_subgraph = Mock()
                mock_subgraph.InputsAsNumpy.return_value = [1]
                mock_subgraph.OutputsAsNumpy.return_value = [2]

                mock_tensors = Mock()
                mock_subgraph.Tensors.return_value = mock_tensors
                mock_tensors.Name.side_effect = [b"test_input", b"test_output", b"test_input", b"test_output"]
                mock_tensors.Type.return_value = 0
                mock_tensors.ShapeAsNumpy.side_effect = [np.array([1, 3, 224, 224]), np.array([1, 10])]
                instance.Subgraphs.return_value = mock_subgraph
                try:
                    test_tflite_model_helper.extract_input_and_output_tensors(user_shape_dict={"test_blah" : [1,3,224,224]})
                except Exception as ex:
                    assert str(ex) == "Please specify all input layers in data_shape."


    mock_file.assert_called_with(Path("test.tflite").resolve(), "rb")

def test_tflite_model_helper_no_user_input(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    sys.modules['tflite.Model'] = Mock()

    try:
        test_tflite_model_helper.extract_input_and_output_tensors()
    except Exception as ex:
        assert str(ex) == "Model input names and shapes must be provided"

def test_tflite_model_helper_load_model(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        try:
            test_tflite_model_helper.load_model()
        except Exception as ex:
            assert str(ex) == "The tflite package must be installed"

def test_tflite_model_helper_uint8(patch_tflite_model_helper):
    test_tflite_model_helper = TFLiteModelHelper("test.tflite")
    sys.modules['tflite'] = Mock()
    sys.modules['tflite.Model'] = Mock()

    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        with patch("tflite.Model.Model.GetRootAsModel") as mocked_model:
                instance = mocked_model.return_value
                test_tflite_model_helper.load_model()

                mock_subgraph = Mock()
                mock_subgraph.InputsAsNumpy.return_value = [1]
                mock_subgraph.OutputsAsNumpy.return_value = [2]

                mock_tensors = Mock()
                mock_subgraph.Tensors.return_value = mock_tensors
                mock_tensors.Name.side_effect = [b"test_input", b"test_output", b"test_input", b"test_output"]
                mock_tensors.Type.return_value = 3
                mock_tensors.ShapeAsNumpy.side_effect = [np.array([1, 224, 224, 3]), np.array([1, 1001])]
                instance.Subgraphs.return_value = mock_subgraph

                test_tflite_model_helper.extract_input_and_output_tensors(user_shape_dict={"test_input" : [1,3,224,224]})
                out = test_tflite_model_helper.get_metadata()

                assert len(out["Inputs"]) == 1
                assert len(out["Outputs"]) == 1
                assert out["Inputs"][0]['name'] == 'test_input'
                assert out["Outputs"][0]['name'] == 'test_output'
                assert out["Inputs"][0]['dtype'] == 'uint8'
                assert out["Outputs"][0]['dtype'] == 'uint8'
                assert out["Inputs"][0]['shape'] == [1, 224, 224, 3]
                assert out["Outputs"][0]['shape'] == [1, 1001]

    mock_file.assert_called_with(Path("test.tflite").resolve(), "rb")

