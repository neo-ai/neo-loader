import logging
import tvm

from typing import Dict, List
from pathlib import Path
from tvm import relay
from tvm.error import OpError
from .convert_layout_mixin import ConvertLayoutMixin
from .abstract_model_loader import AbstractModelLoader
from .helpers.tflite_model_helper import TFLiteModelHelper
from ._base import GraphIR

logger = logging.getLogger(__name__)


class TFLiteModelLoader(AbstractModelLoader, ConvertLayoutMixin):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(TFLiteModelLoader, self).__init__(model_artifacts, data_shape)
        self.__model_file = None
        self.__model = None
        self.__data_types = None

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> (tvm.IRModule, tvm.nd.NDArray):
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return []

    def __extract_model_file_from_model_artifacts(self) -> Path:
        model_files = self._get_files_from_model_artifacts_with_extensions([".tflite"])

        if len(model_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .tflite file is allowed for TFLite models.')
        elif len(model_files) == 1:
            self.__model_file = model_files[0]
        else:
            raise RuntimeError('InputConfiguration: No valid TFLite model found in input file. Please make sure the framework you selected is correct.')

    def __extract_model_data_types_and_metadata_from_model_file(self) -> None:
        tflite_model_helper = TFLiteModelHelper(self.__model_file.as_posix())
        try:
            tflite_model_helper.load_model()
        except Exception as ex:
            logger.exception("Failed to load TFLite model.%s" % repr(ex))
            raise RuntimeError("InputConfiguration: Framework cannot load TFLite model: {}".format(ex))
        else:
            try:
                tflite_model_helper.extract_input_and_output_tensors(user_shape_dict=self.data_shape)
                self.__model = tflite_model_helper.tflite_model
                self.__data_types = tflite_model_helper.input_dtypes_dict
                self._metadata = tflite_model_helper.get_metadata()
            except ImportError:
                raise
            except Exception as ex:
                logging.exception("Unable to infer tensor data type for all inputs/outputs %s" % repr(ex))
                raise RuntimeError("InputConfiguration: Framework cannot load TFLite model. Unable to infer tensor data type for all inputs/outputs: {}".format(ex))

    def load_model(self) -> None:
        self.__extract_model_file_from_model_artifacts()
        self.__extract_model_data_types_and_metadata_from_model_file()
        try:
            self._relay_module_object, self._params = relay.frontend.from_tflite(
                self.__model,
                shape_dict=self.data_shape,
                dtype_dict=self.__data_types
            )
            self._relay_module_object = self.convert_layout(self._relay_module_object)
            self.update_missing_metadata()
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert tensorflow model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert TFLite model. Please make sure the framework you selected is correct. {}".format(e)
            raise RuntimeError(msg)

