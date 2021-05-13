import logging
import torch, torchvision

from tvm import relay
from typing import Dict, List, Tuple
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from ._base import GraphIR


logger = logging.getLogger(__name__)

FLOAT_32 = "float32"


class PyTorchModelLoader(AbstractModelLoader):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super(PyTorchModelLoader, self).__init__(model_artifacts, data_shape)
        self.__pth_file = None

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> tuple:
        return self._relay_module_object, self._params, FLOAT_32

    @property
    def aux_files(self) -> List[Path]:
        return []

    @property
    def data_shape(self) -> List[Tuple[str, Tuple[int]]]:
        # Pytorch frontend takes list of (name, shape) instead of dict
        shape = super(PyTorchModelLoader, self).data_shape
        return [(k, tuple(shape[k])) for k in sorted(shape)]

    def __extract_pth_file_from_model_artifact(self) -> Path:
        pth_files = self._get_files_from_model_artifacts_with_extensions(["pth"])

        if not pth_files:
            raise RuntimeError("InputConfiguration: No pth file found for PyTorch model. "
                               "Please make sure the framework you select is correct.")
        elif len(pth_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .pth file is allowed for PyTorch models.')
        else:
            self.__pth_file = pth_files[0]

    def __get_pytorch_trace_from_model_artifact(self) -> object:
        try:
            trace = torch.jit.load(self.__pth_file.as_posix(), map_location='cpu').float().eval()
        except RuntimeError:
            trace = torch.load(self.__pth_file.as_posix(), map_location="cpu").float().eval()

        try:
            inputs = list(map(lambda shape: torch.zeros(shape[1]), self.data_shape))
            return torch.jit.trace(trace, *inputs).float().eval()
        except RuntimeError:
            inputs = [inp.cuda() for inp in inputs]
            return torch.jit.trace(trace, *inputs).float().eval().cpu()

    def load_model(self) -> None:
        logger.info("Generating relay IR for pytorch model!")
        self.__extract_pth_file_from_model_artifact()

        try:
            trace = self.__get_pytorch_trace_from_model_artifact()
        except Exception as e:
            logger.warning("Failed to load pytorch model. %s" % repr(e))
            msg = 'InputConfiguration: Framework cannot load PyTorch model. {}'.format(e)
            try:
                # for FCOS models
                trace = torch.jit.load(self.__pth_file.as_posix(), map_location='cpu').float().eval()
                self._relay_module_object, self._params = relay.frontend.from_pytorch(trace, self.data_shape)
                self.update_missing_metadata()
            except Exception as e:
                logger.exception("Failed to load pytorch model. %s" % repr(e))
                raise RuntimeError(msg)
        else:
            try:
                self._relay_module_object, self._params = relay.frontend.from_pytorch(trace, self.data_shape)
                self.update_missing_metadata()
            except Exception as e:
                logger.exception("Failed to convert pytorch model. %s" % repr(e))
                msg = 'InputConfiguration: TVM cannot convert the PyTorch model. Invalid model or ' \
                      'input-shape mismatch. Make sure that inputs are lexically ordered and of ' \
                      'the correct dimensionality. {}'.format(e)
                raise RuntimeError(msg)
            else:
                logger.info("Successfully generated relay IR for pytorch model!")
