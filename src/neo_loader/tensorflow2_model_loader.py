import logging
import tvm
import tensorflow as tf

from typing import Dict, List, Optional
from pathlib import Path
from tvm import relay
from tvm.error import OpError
from tvm.relay.frontend.tensorflow_parser import TFParser
from tvm.relay.frontend import tensorflow2
from .abstract_model_loader import AbstractModelLoader
from .convert_layout_mixin import ConvertLayoutMixin
from .helpers.tf2_model_helper import TF2ModelHelper
from ._base import GraphIR

logger = logging.getLogger(__name__)


class Tensorflow2ModelLoader(AbstractModelLoader, ConvertLayoutMixin):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(Tensorflow2ModelLoader, self).__init__(model_artifacts, data_shape)
        logger.info("TF version: {}".format(tf.__version__))
        self.__model_path = None
        self.__output_tensor_names = None
        self.__tf_graph = None
        self.__tf2_model_helper = None

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> (tvm.IRModule, tvm.nd.NDArray):
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return []

    def __get_model_dir_from_model_artifacts(self) -> Optional[Path]:
        model_dirs = []

        for path in self.model_artifacts:
            if path.is_dir():
                if path.joinpath('variables').exists():
                    model_dirs.append(path)

                if path.joinpath('checkpoint').exists():
                    raise RuntimeError('InputConfiguration: TF Checkpoints are not supported. '
                                       'Please make sure the framework you select is correct.')
        if len(model_dirs) > 1:
            raise RuntimeError('InputConfiguration: Exactly one saved model is allowed for TensorFlow models.')
        elif len(model_dirs) == 1:
            return model_dirs[0]
        else:
            return None

    def __get_model_file_from_model_artifacts(self) -> Optional[Path]:
        model_files = self._get_files_from_model_artifacts_with_extensions(["pb", "pbtxt"])

        if len(model_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .pb or .pbtxt file is allowed for TensorFlow models.')
        elif len(model_files) == 1:
            return model_files[0]
        else:
            return None

    def __extract_model_path_from_model_artifacts(self) -> None:
        model_file = self.__get_model_file_from_model_artifacts()
        model_dir = self.__get_model_dir_from_model_artifacts()

        if model_dir:
            self.__model_path = model_dir
        elif model_file:
            self.__model_path = model_file
        else:
            raise RuntimeError('InputConfiguration: No valid TensorFlow model found in input files. '
                               'Please make sure the framework you select is correct.')

    def __extract_metadata_and_output_tensor_names_from_model(self) -> None:
        try:
            self.__tf2_model_helper.extract_input_and_output_tensors()
        except Exception as e:
            logger.exception("Framework cannot load model. {}".format(e))
            raise RuntimeError("InputConfiguration: Framework cannot load Tensorflow model: {}".format(e))

        try:
            self.__output_tensor_names = [name.rstrip(":0") for name in self.__tf2_model_helper.output_tensor_names]
            self._metadata = self.__tf2_model_helper.get_metadata()
        except Exception as e:
            logger.exception("Framework cannot load model.")
            raise RuntimeError("InputConfiguration: Framework cannot load Tensorflow model: {}".format(e))

    def __extract_tf_graph(self):
        # TODO: Add loading logic from TFParser, when TFParser starts to support TF2.
        try:
            self.__tf_graph = self.__tf2_model_helper.get_tf_graph_from_graph_model()
        except Exception as e:
            logger.exception("Failed to load TF model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: Framework cannot load Tensorflow model: {}".format(e))

    def load_model(self) -> None:
        self.__extract_model_path_from_model_artifacts()
        self.__tf2_model_helper = TF2ModelHelper(self.__model_path.as_posix(), self.data_shape)
        self.__extract_metadata_and_output_tensor_names_from_model()
        self.__extract_tf_graph()
        logger.info("Model Version tf-{}".format(self.__tf2_model_helper.get_tensorflow_version()))
        try:
            self._relay_module_object, self._params = relay.frontend.tensorflow2.from_tensorflow(
                self.__tf_graph, shape=self.data_shape, outputs=self.__output_tensor_names
            )
            self._relay_module_object = self.convert_layout(self._relay_module_object)
            self.update_missing_metadata()
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert tensorflow model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert Tensorflow model. Please make sure the framework you selected is correct. {}".format(e)
            msg += self.model_version_hint_message()
            raise RuntimeError(msg)

    def model_version_hint_message(self) -> str:
        model_version = self.__tf2_model_helper.get_tensorflow_version()
        if model_version.startswith("1."):
            return "\nTensorflow version selected: {}. Model version founded: {}.".format("2.x", model_version)
        return ""