import importlib
import json
import logging
import os
import tarfile
import time

from enum import Enum
from .abstract_model_loader import AbstractModelLoader

logger = logging.getLogger(__name__)

class Framework(Enum):
    MXNET = 'mxnet'
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    ONNX = 'onnx'
    XGBOOST = 'xgboost'
    KERAS = 'keras'
    TFLITE = 'tflite'
    DARKNET = 'darknet'

FRAMEWORK_TO_MODEL_LOADER = {
    'mxnet': "MxNetModelLoader",
    'pytorch': "PyTorchModelLoader",
    'tensorflow': "TensorflowModelLoader",
    'onnx': "ONNXModelLoader",
    'xgboost': "XGBoostModelLoader",
    'keras': "KerasModelLoader",
    'tflite': "TFLiteModelLoader",
    'darknet': "DarkNetModelLoader",
    'sklearn': "SklearnModelLoader"
}

DOWNLOAD_DIR = '/compiler'
COMPILATION_START = 'COMPILATION_START'

def get_model_loader_for_framework(framework) -> AbstractModelLoader:
    if isinstance(framework, Framework):
        framework = framework.value
    module_path = f".{framework.lower()}_model_loader"
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, FRAMEWORK_TO_MODEL_LOADER[framework.lower()])


def __clean_model_files(model_files):
    """Clean model files, such as removing hidden files.

    Parameters
    ----------
    model_files : list of str
        Model files.

    Returns
    -------
    output : list of str
        Output model files
    """
    logger.warn("model_files: {}".format(model_files))
    output = []
    for model_file in model_files:
        # remove hidden files
        if os.path.basename(model_file).startswith("."):
            continue
        output.append(model_file)
    logger.warn("output files: {}".format(output))

    return output


def find_archive(output_directory='/compiler', sidecar=True):
    if sidecar:
        while COMPILATION_START not in os.listdir(output_directory):
            time.sleep(5)

    for f in os.listdir(output_directory):
        if f.endswith('.gz'):
            return os.path.join(output_directory, f)
    raise RuntimeError("InputConfiguration: Unable to find input archive")


def extract_model_artifacts(archive=None, output_directory='/compiler', sidecar=True) -> [str]:
    if not archive:
        archive = find_archive()

    base_dir = archive[:archive.rfind('/')]
    file_list = []
    try:
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(output_directory)
            file_list = tf.getnames()
    except Exception as e:
        logger.error(repr(e))
        raise RuntimeError("InputConfiguration: Unable to untar input model. "
                            "Please confirm the model is a tar.gz file")

    result = []
    for model_file in __clean_model_files(file_list):
        result.append(os.path.join(output_directory, model_file))

    logger.info(f"Successfully extracted model artifacts from {archive}")
    logger.info(f"files: {result}")
    return result


def get_framework():
    framework = os.environ['FRAMEWORK']
    return Framework(framework.lower())


def validate_input_shape(framework, input_shape) -> {str: list}:
    if isinstance(input_shape, str):
        try:
            input_shape = json.loads(input_shape)
        except Exception as e:
            raise RuntimeError(f"InputConfiguration: Cannot load DataInputConfig. Incorrect json format. {e}")

    if not isinstance(input_shape, (dict, list)):
        raise RuntimeError("InputConfiguration: DataInputConfig is not dictionary or list.")

    if framework.lower() == Framework.PYTORCH.value and isinstance(input_shape, list):
        input_shape = {'input' + str(i): k for i, k in enumerate(input_shape)}
    return input_shape


def load_model(model_artifacts: [str] = None, input_shape: {str: [int]} = None):
    framework = get_framework()
    if not model_artifacts:
        model_artifacts = extract_model_artifact()
    if not input_shape:
        input_shape = validate_input_shape(framework, os.environ.get('INPUT_SHAPE'))

    model_loader = get_model_loader_for_framework(framework)
    loader = model_loader(model_artifacts, input_shape)
    loader.load_model()
    return loader
