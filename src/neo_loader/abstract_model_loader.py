import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

from ._base import GraphIR

logger = logging.getLogger(__name__)


class AbstractModelLoader(ABC):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        if not data_shape:
            raise RuntimeError('InputConfiguration: data_shape not found. Make sure a valid data_shape parameter is provided.')
        if not model_artifacts:
            raise RuntimeError('InputConfiguration: No model file found.')
        self.__model_artifacts = model_artifacts
        self.__data_shape = data_shape
        self._relay_module_object = None
        self._params = None
        self._metadata = {}

    @property
    def data_shape(self) -> Any:
        return self.__data_shape

    @property
    @abstractmethod
    def model_objects(self) -> Any:
        pass

    @property
    def model_artifacts(self) -> List[Path]:
        return [Path(file) for file in self.__model_artifacts]

    @property
    def metadata(self) -> Dict[str, List]:
        return self._metadata

    @property
    @abstractmethod
    def aux_files(self) -> List[Path]:
        pass

    @property
    @abstractmethod
    def ir_format(self) -> GraphIR:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    def _get_files_from_model_artifacts_with_extensions(self, extensions: List[str], exclude_files: List[str] = []) -> List[Path]:
        for index, extension in enumerate(extensions):
            if not extension.startswith("."):
                extensions[index] = "." + extension

        return list(filter(
            lambda file: file.suffix in extensions and file.name not in exclude_files, self.model_artifacts
        ))

    def __convert_relay_shape(self, relay_shape) -> List[int]:
        """Convert Relay shape to a list of integers, with -1 in place of relay.Any()"""
        import tvm
        return [item.value if not isinstance(item, tvm.tir.expr.Any) else -1 for item in relay_shape]

    def __get_outputs_from_relay(self) -> List[Dict[str, Any]]:
        import tvm
        from tvm import relay
        from tvm import ir

        mod = relay.transform.InferType()(self._relay_module_object)

        if type(mod['main'].ret_type) is ir.tensor_type.TensorType:
            dshape = []
            for item in mod['main'].ret_type.shape:
                if type(item).__name__ == 'Any':
                    dshape.append(-1)
                else:
                    dshape.append(item.value)
            return [{'dtype': mod['main'].ret_type.dtype, 'shape': dshape}]
        else:
            return [{'dtype': out.dtype, 'shape': self.__convert_relay_shape(out.shape)} for out in mod['main'].ret_type.fields]

    def __update_input_data_from_data_shape(self) -> None:
        from tvm import relay
        if 'Inputs' not in self.metadata:
            inputs = []
            for key, value in self.__data_shape.items():
                inputs.append({
                    'name': key,
                    'shape': value
                })
            self._metadata['Inputs'] = inputs
        else:
            for inp in self._metadata['Inputs']:
                if None in inp['shape'] and inp['name'] in self.__data_shape:
                    inp['shape'] = self.__data_shape[inp['name']]

        # Get input dtypes from Relay
        if self.ir_format == GraphIR.relay:
            relay_var_dtypes = {}
            for p in self._relay_module_object["main"].params:
                if isinstance(p.type_annotation, relay.ty.TupleType):
                    if len(p.type_annotation.fields) != 1:
                        raise RuntimeError("Tuple input with multiple elements is currently not supported.")
                    relay_var_dtypes[p.name_hint] = p.type_annotation.fields[0].dtype
                else:
                    relay_var_dtypes[p.name_hint] = p.type_annotation.dtype
            for inp in self._metadata['Inputs']:
                if 'dtype' not in inp and inp['name'] in relay_var_dtypes:
                    inp['dtype'] = relay_var_dtypes[inp['name']]


    def __update_output_data_from_relay(self) -> None:
        from tvm import relay
        relay_output_metadata = self.__get_outputs_from_relay()
        if 'Outputs' not in self.metadata or len(relay_output_metadata) != len(self._metadata['Outputs']):
            self._metadata['Outputs'] = relay_output_metadata
        else:
            for index in range(len(relay_output_metadata)):
                self._metadata['Outputs'][index]['dtype'] = relay_output_metadata[index]['dtype']
                self._metadata['Outputs'][index]['shape'] = relay_output_metadata[index]['shape']

        # Add generic outputs names if none (needed by DLR for RelayVM).
        for i, out in enumerate(self._metadata['Outputs']):
            if 'name' not in out:
                out['name'] = "output_{}".format(i)


    def update_missing_metadata(self) -> None:
        self.__update_input_data_from_data_shape()
        if self.ir_format == GraphIR.relay:
            self.__update_output_data_from_relay()
