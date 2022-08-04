import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from neo_loader.tensorflow2_model_loader import Tensorflow2ModelLoader


class MockedOpError(Exception):
    pass

@pytest.fixture
def patch_tvm(monkeypatch):
    mock_tvm = MagicMock()
    monkeypatch.setattr("neo_loader.tensorflow2_model_loader.tvm", mock_tvm)
    return mock_tvm

@pytest.fixture
def patch_relay(monkeypatch):
    mock_relay = MagicMock()
    monkeypatch.setattr("neo_loader.tensorflow2_model_loader.relay", mock_relay)
    return mock_relay


@pytest.fixture
def patch_op_error(monkeypatch):
    monkeypatch.setattr("neo_loader.tensorflow2_model_loader.OpError", MockedOpError)
    return MockedOpError


@pytest.fixture
def patch_tf2(monkeypatch):
    patch_tf2 = MagicMock()
    monkeypatch.setattr("neo_loader.tensorflow2_model_loader.tf", patch_tf2)
    return patch_tf2


@pytest.fixture
def patch_tf2_model_helper(monkeypatch):
    mock_tf2_model_helper = MagicMock()
    monkeypatch.setattr("neo_loader.tensorflow2_model_loader.TF2ModelHelper", mock_tf2_model_helper)
    return mock_tf2_model_helper


def test_tensorflow_with_pb_file(patch_tvm, patch_relay, patch_tf2_model_helper, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.return_value.__iter__.return_value = ["module", "params"]
    patch_tf2.__version__ = "2.4.2"
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_relay.frontend.tensorflow2.from_tensorflow.assert_called()


def test_tensorflow_with_model_dir(patch_tvm, patch_relay, patch_tf2_model_helper, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.return_value.__iter__.return_value = ["module", "params"]
    patch_tf2.__version__ = "2.4.2"
    model_dir = Path(tempfile.mkdtemp())
    model_dir.joinpath("variables").mkdir(exist_ok=True)
    model_artifacts = [model_dir.as_posix()]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_relay.frontend.tensorflow2.from_tensorflow.assert_called()


def test_tensorflow_multiple_pb_file(patch_tf2):
    model_artifacts = ["test.pb", "test.pbtxt"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one .pb or .pbtxt file is allowed for TensorFlow models.' in str(err)


def test_tensorflow_multiple_saved_model_directory_file(patch_tf2):
    model_dir1 = Path(tempfile.mkdtemp())
    model_dir1.joinpath("variables").mkdir(exist_ok=True)
    model_dir2 = Path(tempfile.mkdtemp())
    model_dir2.joinpath("variables").mkdir(exist_ok=True)
    model_artifacts = [model_dir1.as_posix(), model_dir2.as_posix()]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Exactly one saved model is allowed for TensorFlow models.' in str(err)


def test_tensorflow_with_both_saved_model_and_pb_file(patch_tvm, patch_relay, patch_tf2_model_helper, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.return_value.__iter__.return_value = ["module", "params"]
    model_dir = Path(tempfile.mkdtemp())
    model_dir.joinpath("variables").mkdir(exist_ok=True)
    model_artifacts = ["test.pb", model_dir.as_posix()]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    loader.load_model()
    patch_relay.frontend.tensorflow2.from_tensorflow.assert_called()


def test_tensorflow_without_any_model(patch_tf2):
    model_artifacts = ["test.blah"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: No valid TensorFlow model found in input files.' in str(err)


def test_tensorflow_tf_model_helper_exception(patch_tf2_model_helper, patch_tf2):
    patch_tf2_model_helper.return_value.extract_input_and_output_tensors.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load Tensorflow model' in str(err)


def test_tensorflow_tf_model_helper_get_metadata_exception(patch_tf2_model_helper, patch_tf2):
    patch_tf2_model_helper.return_value.get_metadata.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: Framework cannot load Tensorflow model' in str(err)


def test_tensorflow_op_error(patch_tf2_model_helper, patch_relay, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.side_effect = patch_op_error("Dummy OpError")
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(patch_op_error) as err:
        loader.load_model()
    assert 'Dummy OpError' in str(err)


def test_tensorflow_relay_exception(patch_tf2_model_helper, patch_relay, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.side_effect = Exception("Dummy Exception")
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    patch_tf2.__version__ = "2.4.2"
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    assert 'InputConfiguration: TVM cannot convert Tensorflow model' in str(err)

def test_model_version_message(patch_tvm, patch_relay, patch_tf2_model_helper, patch_op_error, patch_tf2):
    patch_relay.frontend.tensorflow2.from_tensorflow.side_effect = Exception("Dummy Exception")
    patch_tf2_model_helper().get_tensorflow_version.return_value="1.5"
    patch_tf2.__version__ = "2.4.2"
    model_artifacts = ["test.pb"]
    data_shape = {"input": [1, 3, 224, 224]}
    loader = Tensorflow2ModelLoader(model_artifacts, data_shape)
    with pytest.raises(RuntimeError) as err:
        loader.load_model()
    patch_tf2_model_helper().get_tensorflow_version.assert_called()
    assert 'Tensorflow version selected: 2.x' in str(err.value)
    assert 'Model version founded: 1.5' in str(err.value)
