import logging
import onnx

import tvm
from tvm import relay
from tvm.error import OpError
from typing import Dict, List
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from .convert_layout_mixin import DynamicToStaticMixin
from ._base import GraphIR

logger = logging.getLogger(__name__)


class ONNXModelLoader(AbstractModelLoader, DynamicToStaticMixin):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(ONNXModelLoader, self).__init__(model_artifacts, data_shape)

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> object:
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return []

    def __get_onnx_file_from_model_artifacts(self) -> Path:
        onnx_files = self._get_files_from_model_artifacts_with_extensions(["onnx"])

        if not onnx_files:
            raise RuntimeError("InputConfiguration: No .onnx file found for ONNX model. "
                               "Please make sure the framework you select is correct.")
        elif len(onnx_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .onnx file is allowed for ONNX models.')
        else:
            return onnx_files[0]

    def __get_onnx_model_from_model_artifacts(self) -> onnx.ModelProto:
        onnx_file = self.__get_onnx_file_from_model_artifacts()

        try:
            model = onnx.load(onnx_file.as_posix())
        except Exception as e:
            logger.exception("Failed to load onnx model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: Framework cannot load ONNX model. Please make sure the framework you select is correct. {}".format(e))
        else:
            return model

    def load_model(self) -> None:
        model = self.__get_onnx_model_from_model_artifacts()

        try:
            self._relay_module_object, self._params = relay.frontend.from_onnx(model, self.data_shape)
            self._relay_module_object = self.dynamic_to_static(self._relay_module_object)
            self.update_missing_metadata()
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert onnx model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert ONNX model. Please make sure the framework you selected is correct. {}".format(e)
            raise RuntimeError(msg)

