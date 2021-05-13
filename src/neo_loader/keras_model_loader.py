import logging
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tvm import relay
from typing import Dict, List, Tuple
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from ._base import GraphIR


logger = logging.getLogger(__name__)



class KerasModelLoader(AbstractModelLoader):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(KerasModelLoader, self).__init__(model_artifacts, data_shape)
        self.__model_file = None
        self.__model = None


    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> tuple:
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return []

    def __extract_model_file_from_model_artifacts(self) -> None:
        model_files = self._get_files_from_model_artifacts_with_extensions(["h5"])

        if not model_files:
            raise RuntimeError('InputConfiguration: No h5 file provided in {}.'.format(self.model_artifacts))
        elif len(model_files) > 1:
            file_paths = [file.as_posix() for file in model_files]
            raise RuntimeError('InputConfiguration: Multiple h5 files provided, {}, when only one is allowed.'.format(file_paths))
        else:
            self.__model_file = model_files[0]

    def __load_keras_model_from_model_artifacts(self) -> None:
        try:
            self.__model = load_model(self.__model_file.as_posix())
        except Exception as e:
            logger.exception("Failed to load keras model. %s" % repr(e))
            raise RuntimeError('InputConfiguration: Unable to load provided Keras model. Error: {}'.format(e))

    def __validata_data_shape_with_input_layer(self, layer: InputLayer) -> None:
        layer_input_shape = layer.input_shape[0]
        input_shape = self.data_shape[layer.name]
        if layer_input_shape[0] is not None and layer_input_shape[0] != input_shape[0]:
            raise RuntimeError(f'InputConfiguration: Input {layer.name} has wrong batch size in Input Shape dictionary.')
        layer_len = len(layer_input_shape)
        model_chw_shape = [layer_input_shape[3] if 3 < layer_len else None,
                           layer_input_shape[1] if 1 < layer_len else None,
                           layer_input_shape[2] if 2 < layer_len else None]
        input_chw_shape = input_shape[1:4]
        if model_chw_shape != input_chw_shape:
            msg = 'InputConfiguration: Input {} has wrong shape in Input Shape dictionary. ' \
                  'Input shapes should be provided in NCHW format. For example, ({},{},{},{})'
            raise RuntimeError(msg.format(layer.name, layer_input_shape[0], model_chw_shape[0], model_chw_shape[1], model_chw_shape[2]))

    def __validate_data_shape_with_model(self) -> None:
        for layer in  self.__model.layers:
            if isinstance(layer, InputLayer):
                if layer.name not in self.data_shape:
                    msg = 'InputConfiguration: Model contains input layer ({}) not specified in data_shape. ' \
                          'Please specify all input layers in data_shape.'
                    raise RuntimeError(msg.format(layer.name))
                self.__validata_data_shape_with_input_layer(layer)

    def load_model(self) -> None:
        self.__extract_model_file_from_model_artifacts()
        self.__load_keras_model_from_model_artifacts()
        self.__validate_data_shape_with_model()

        try:
            self._relay_module_object, self._params = relay.frontend.from_keras(self.__model, shape=self.data_shape)
            self.update_missing_metadata()
        except RuntimeError:
            raise
        except Exception as e:
            logger.exception("Failed to convert keras model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: TVM cannot convert the Keras model. {}".format(e))
