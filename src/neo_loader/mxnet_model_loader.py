import logging
import mxnet as mx

from tvm import relay
from tvm.error import OpError
from typing import Dict, List, Any
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from ._base import GraphIR


logger = logging.getLogger(__name__)


class MxNetModelLoader(AbstractModelLoader):

    SAGEMAKER_AUXILIARY_JSON_FILES = ['model-shapes.json', 'hyperparams.json']
    AMBARELLA_CONFIG_JSON_FILES = ['amba_config.json']

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        super().__init__(model_artifacts, data_shape)

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> Any:
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return list(filter(
            lambda file: file.name in self.SAGEMAKER_AUXILIARY_JSON_FILES + self.AMBARELLA_CONFIG_JSON_FILES,
            self.model_artifacts
        ))

    def __get_symbol_file_from_model_artifact(self) -> Path:
        json_files = self._get_files_from_model_artifacts_with_extensions(["json"],
                                                                          exclude_files=self.SAGEMAKER_AUXILIARY_JSON_FILES + self.AMBARELLA_CONFIG_JSON_FILES)

        if len(json_files) == 1:
            return json_files[0]

        if not json_files:
            raise RuntimeError("InputConfiguration: No symbol file found for MXNet model. "
                               "Please make sure the framework you select is correct.")

        if len(json_files) > 1:
            raise RuntimeError("InputConfiguration: Only one symbol file is allowed for MXNet model. "
                               "Please make sure the framework you select is correct.")

    def __get_param_file_from_model_artifact(self) -> Path:
        param_files = self._get_files_from_model_artifacts_with_extensions(["params"], exclude_files=self.SAGEMAKER_AUXILIARY_JSON_FILES)

        if not param_files:
            raise RuntimeError("InputConfiguration: No parameter file found for MXNet model. "
                               "Please make sure the framework you select is correct.")
        elif len(param_files) > 1:
            select_param_file = param_files[0]
            latest_checkpoint = int(select_param_file.name[-11:-7])
            for param_file in param_files:
                checkpoint = int(param_file.name[-11:-7])
                if checkpoint > latest_checkpoint:
                    select_param_file = param_file
                    latest_checkpoint = checkpoint

            logger.warning(f"InputConfiguration: Multiple parameter files found for MXNet model. "
                           f"Parameter file: {select_param_file.as_posix()} will be used.")
            return select_param_file
        else:
            return param_files[0]

    def __get_model_json_from_model_artifact(self) -> Dict:
        symbol_file = self.__get_symbol_file_from_model_artifact()
        logger.info(f"Loading model from {symbol_file.as_posix()}")
        try:
            model_json = mx.symbol.load(symbol_file.as_posix())
        except Exception as e:
            raise RuntimeError("InputConfiguration: Framework can't load the MXNet model: {}".format(e))
        else:
            logger.info(f"Successfully loaded model from {symbol_file.as_posix()}")
            return model_json

    def __get_saved_dict_from_model_artifact(self) -> mx.ndarray.NDArray:
        param_file = self.__get_param_file_from_model_artifact()
        logger.info(f"Loading weights from {param_file.as_posix()}")
        try:
            saved_dict = mx.ndarray.load(param_file.as_posix())
        except Exception as e:
            logger.exception("Failed to load mxnet model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: Framework can't load the MXNet model: {}".format(e))
        else:
            logger.info(f"Successfully loaded weights from {param_file.as_posix()}")
            return saved_dict

    def __get_arg_and_aux_params_from_model_artifact(self) -> (Dict, Dict):
        saved_dict = self.__get_saved_dict_from_model_artifact()
        arg_params, aux_params = {}, {}

        for key, value in saved_dict.items():
            if not key.startswith('arg:') and not key.startswith('aux:'):
                raise RuntimeError("InputConfiguration: Framework can't load the MXNet model: Please use HybridBlock.export() "
                                   "or gluoncv.utils.export_block() to export your model instead of "
                                   "Block.save_parameters().")

            prefix, name = key.split(':', 1)
            if prefix == "arg":
                arg_params[name] = value
            elif prefix == "aux":
                aux_params[name] = value

        return arg_params, aux_params

    def load_model(self) -> None:
        logger.info("Generating relay IR for mxnet model!")
        model_json = self.__get_model_json_from_model_artifact()
        arg_params, aux_params = self.__get_arg_and_aux_params_from_model_artifact()
        try:
            self._relay_module_object, self._params = relay.frontend.from_mxnet(
                model_json,
                self.data_shape,
                arg_params=arg_params,
                aux_params=aux_params
            )
            self.update_missing_metadata()
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert mxnet model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: TVM can't convert the MXNet model. {}".format(e))
        else:
            logger.info("Successfully generated relay IR for mxnet model!")
