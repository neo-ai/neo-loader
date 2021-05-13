import logging
import mxnet as mx

from tvm import relay
from tvm.error import OpError
from typing import Dict, List, Any
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from ._base import GraphIR
import json
import os


logger = logging.getLogger(__name__)
AP_CROSS_VALIDATION_PREFIX = "model0"

class MxNetModelLoader(AbstractModelLoader):

    SAGEMAKER_AUXILIARY_JSON_FILES = ['model-shapes.json', 'hyperparams.json']
    AMBARELLA_CONFIG_JSON_FILES = ['amba_config.json']
    METADATA_AUTOPILOT_JSON_FILES = ['model-metadata.json']

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

        
    def __get_symbol_file_prefix(self, symbol_file: Path):
        if "-symbol.json" in symbol_file.name:
            # MxNet standard
            return symbol_file.name.replace("-symbol.json", "")
        else:
            return symbol_file.name.replace(".json", "")
            

    def __get_symbol_file_from_model_artifact(self) -> Path:
        symbol_files = self._get_files_from_model_artifacts_with_extensions(["json"],
                                                                          exclude_files=self.SAGEMAKER_AUXILIARY_JSON_FILES + self.AMBARELLA_CONFIG_JSON_FILES + self.METADATA_AUTOPILOT_JSON_FILES)

        if len(symbol_files) == 1:
            return symbol_files[0]
        
        if not symbol_files:
            raise RuntimeError("InputConfiguration: No symbol file found for MXNet model. "
                               "Please make sure the framework you select is correct.")

        if len(symbol_files) > 1:
            # support SageMaker AP cross-validaiton, which has multiple models, 
            # fetch first model that matches prefix;
            fpath = list(filter(lambda file: file.name.startswith(AP_CROSS_VALIDATION_PREFIX), symbol_files))
            if len(fpath) == 1:
                return fpath[0]
            else:    
                raise RuntimeError("InputConfiguration: Only one symbol file is allowed for MXNet model. "
                               "Please make sure the framework you select is correct.")

    def __get_param_file_from_model_artifact(self) -> Path:
        param_files = self._get_files_from_model_artifacts_with_extensions(["params"], exclude_files=self.SAGEMAKER_AUXILIARY_JSON_FILES)
        symbol_file = self.__get_symbol_file_from_model_artifact()
    
        symbol_prefix = self.__get_symbol_file_prefix(symbol_file)
        target_params = list(filter(lambda file: file.name.startswith(symbol_prefix), param_files))
        
        if not param_files:
            raise RuntimeError("InputConfiguration: No parameter file found for MXNet model. "
                               "Please make sure the framework you select is correct.")
        elif not target_params:
            raise RuntimeError(f"InputConfiguration: No parameter file found for MXNet model: {symbol_file.as_posix()} "
                                "Please make sure the prefix of params file match the prefix of symbol file.")
                               
        elif len(target_params) > 1:
            select_param_file = target_params[0]
            latest_checkpoint = int(select_param_file.name[-11:-7])
            for param_file in target_params:
                checkpoint = int(param_file.name[-11:-7])
                if checkpoint > latest_checkpoint:
                    select_param_file = param_file
                    latest_checkpoint = checkpoint

            logger.warning(f"InputConfiguration: Multiple parameter files found for MXNet model. "
                           f"Parameter file: {select_param_file.as_posix()} will be used.")
            return select_param_file
        else:
            return target_params[0]

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

    def __get_metadata_file_from_model_artifact(self) -> Path:
        json_files_1 = list(filter(lambda file: file.name.endswith("model-metadata.json"), self.model_artifacts))

        if len(json_files_1) == 1:
            return json_files_1[0]

        if not json_files_1:
            raise RuntimeError("InputConfiguration: No metadata file found for Sagemaker MXNet model. "
                               "Please make sure the framework you select is correct.")

        if len(json_files_1) > 1:
            raise RuntimeError("InputConfiguration: Only one metadata file is allowed for Sagemaker MXNet model. "
                               "Please make sure the framework you select is correct.")

    def __convert_metadata_json_to_dict(self) -> Dict:
        metadata_json = self.__get_metadata_file_from_model_artifact()
        if metadata_json:
            try:
                with open(metadata_json) as metadata_json_file:
                    feature_list = json.load(metadata_json_file)
            except Exception as e:
                raise RuntimeError(f"InputConfiguration: Cannot load Metadata value. Incorrect json format. {e}")
        return feature_list

    def __check_feature_dim(self) -> str:
        feature_list = self.__convert_metadata_json_to_dict()
        if feature_list:
            if "feature_dim" in feature_list:
                return feature_list["feature_dim"]
        return None

    def load_model(self) -> None:
        logger.info("Generating relay IR for mxnet model!")
        model_json = self.__get_model_json_from_model_artifact()
        arg_params, aux_params = self.__get_arg_and_aux_params_from_model_artifact()
        compiler_option = json.loads(os.environ.get('COMPILER_OPTIONS')) if os.environ.get('COMPILER_OPTIONS') is not None else None
        current_data_shape = self.data_shape['data'] if "data" in self.data_shape else []
        if current_data_shape[-1:] == [-1] and compiler_option:
            if compiler_option.get("PLATFORM") == "AL2012":
                feature_dim = self.__check_feature_dim()
                if feature_dim:
                    keys = list(self.data_shape.keys())
                    current_data_shape[-1] = feature_dim
                    self.data_shape[keys[-1]] = current_data_shape
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
