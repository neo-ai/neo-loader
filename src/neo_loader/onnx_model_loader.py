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
        logger.info("Model Version onnx-{}".format(self.__get_onnx_model_version(model)))

        try:
            self._relay_module_object, self._params = relay.frontend.from_onnx(model, self.data_shape)
            self._relay_module_object = self.dynamic_to_static(self._relay_module_object)
            self.update_missing_metadata()
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert onnx model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert ONNX model. Please make sure the framework you selected is correct. {}".format(e)
            msg += self.model_version_hint_message(model)
            raise RuntimeError(msg)

    def __get_onnx_model_version(self, model) -> str:
        # In latest onnx, there is a programmatic way to access version table
        # Hardcode for now
        # versions = []
        # for release_ver, ir_ver, opset_ver, _ in VERSION_TABLE
        #     if (model.ir_version, model.opset_import[0].version) == (ir_ver, opset_ver):
        #         versions.append(release_ver)
        # return ", ".join(versions)
        version_mapping = {
            # https://github.com/onnx/onnx/blob/rel-1.10.2/docs/Versioning.md#released-versions
            (3,1): "1.0",
            (3,5): "1.1",
            (3,6): "1.1.2",
            (3,7): "1.2",
            (3,8): "1.3",
            (4,9): "1.4.1",
            (5,10): "1.5.0",
            (6,11): "1.6.0",
            (7,12): "1.7.0",
            (7,13): "1.8.x",
            (7,14): "1.9.0",
            (8,15): "1.10.x",
        }
        try:

            logger.info(f"ONNX IR version {model.ir_version}")
            logger.info(f"ONNX OPSET version {model.opset_import[0].version}")

            version = version_mapping.get((model.ir_version, model.opset_import[0].version))
            if version == None:
                return "not found"
            return version
        except Exception as e:
            logger.warning(f"Error when loading ONNX model version: {e}")
        return "not found"

    def model_version_hint_message(self, model) -> str:
        model_version = self.__get_onnx_model_version(model)
        framework_version = onnx.__version__
        msg = f"\nONNX Version running: {framework_version}. Model Version found: {model_version}."
        return msg
