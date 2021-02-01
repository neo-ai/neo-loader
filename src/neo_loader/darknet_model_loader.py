import logging

from tvm import relay
from tvm.relay.testing import darknet
from tvm.error import OpError
from typing import Dict, List
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from ._base import GraphIR

logger = logging.getLogger(__name__)


class DarkNetModelLoader(AbstractModelLoader):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(DarkNetModelLoader, self).__init__(model_artifacts, data_shape)
        self.__model_objects = None
        self.__lib_file = 'libdarknet.so'

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def metadata(self) -> Dict[str, Dict]:
        return {}

    @property
    def model_objects(self) -> object:
        return self.__model_objects

    @property
    def aux_files(self) -> List[Path]:
        return []

    @property
    def data_shape(self) -> List:
        # Darknet frontend takes list(batch_size, c, h, w) instead of dict
        shape = super(DarkNetModelLoader, self).data_shape
        return list(shape.values())[0]

    def __get_darknet_file_from_model_artifacts(self) -> (Path, Path):
        config_files = self._get_files_from_model_artifacts_with_extensions(["cfg"])

        if not config_files:
            raise RuntimeError("InputConfiguration: No .cfg file found for DarkNet model. "
                               "Please make sure the framework you select is correct.")
        elif len(config_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .cfg file is allowed for DarkNet models.')
        
        weight_files = self._get_files_from_model_artifacts_with_extensions(["weights"])

        if not weight_files:
            raise RuntimeError("InputConfiguration: No .weights file found for DarkNet model. "
                               "Please make sure the framework you select is correct.")
        elif len(weight_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .weights file is allowed for DarkNet models.')
        
        return config_files[0], weight_files[0]

    def __get_darknet_model_from_model_artifacts(self) -> None:
        config_file, weight_file = self.__get_darknet_file_from_model_artifacts()
        
        try:
            # https://github.com/apache/incubator-tvm/blob/master/python/tvm/relay/testing/darknet.py#L140
            darknet_lib = darknet.__darknetffi__.dlopen(self.__lib_file)
            model = darknet_lib.load_network(config_file.as_posix().encode('utf-8'), weight_file.as_posix().encode('utf-8'), 0)
        except Exception as e:
            logger.exception("Failed to load DarkNet model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: Unable to load DarkNet model. {}".format(e))
        else:
            return model

    def load_model(self) -> None:
        model = self.__get_darknet_model_from_model_artifacts()

        try:
            self.__model_objects = relay.frontend.from_darknet(model, self.data_shape)
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert darknet model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert DarkNet model. Please make sure the framework you selected is correct. {}".format(e)
            raise RuntimeError(msg)

