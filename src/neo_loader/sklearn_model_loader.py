import logging
import joblib
from sagemaker_sklearn_extension import externals

from tvm import relay
from tvm.error import OpError
from typing import Dict, List
from pathlib import Path
from .abstract_model_loader import AbstractModelLoader
from .convert_layout_mixin import DynamicToStaticMixin
from ._base import GraphIR

logger = logging.getLogger(__name__)

FLOAT_32 = "float32"

class SklearnModelLoader(AbstractModelLoader, DynamicToStaticMixin):

    def __init__(self, model_artifacts: List[str], data_shape: Dict[str, List[int]]) -> None:
        # hardcode input name to "input"
        # Will be exploded into "input_float", "input_string" 
        _data_shape = {}
        for key, value in data_shape.items():
            _data_shape = {"input": value}

        super(SklearnModelLoader, self).__init__(model_artifacts, _data_shape)
        self.mappings = []
        self.date_col = 0
        self.col_to_mapping = {}
        self.transform_func = "transform"

    @property
    def ir_format(self) -> GraphIR:
        return GraphIR.relay

    @property
    def model_objects(self) -> object:
        return self._relay_module_object, self._params

    @property
    def aux_files(self) -> List[Path]:
        return []

    def __get_sklearn_file_from_model_artifacts(self) -> Path:
        sklearn_files = self._get_files_from_model_artifacts_with_extensions(["joblib"])

        if not sklearn_files:
            raise RuntimeError("InputConfiguration: No .joblib file found for Scikit-Learn model. "
                               "Please make sure the framework you select is correct.")
        elif len(sklearn_files) > 1:
            raise RuntimeError('InputConfiguration: Exactly one .joblib file is allowed for Scikit-Learn models.')
        else:
            return sklearn_files[0]

    def __get_sklearn_model_from_model_artifacts(self) -> externals.automl_transformer:
        sklearn_file = self.__get_sklearn_file_from_model_artifacts()

        try:
            model = joblib.load(sklearn_file.as_posix())
        except Exception as e:
            logger.exception("Failed to load Scikit-Learn model. %s" % repr(e))
            raise RuntimeError("InputConfiguration: Framework cannot load Scikit-Learn model. Please make sure the framework you select is correct. {}".format(e))
        else:
            return model
    
    def __build_numeric_mapping(self, categories, cols) -> dict:
        for j in range(len(cols)):
            try:
                converted_categories = categories[j].copy()
                for i, cat in enumerate(categories[j]):
                    converted_categories[i] = float(cat)
                categories[j] = converted_categories
            except ValueError:
                mapping = {}
                for i in range(len(categories[j])):
                    item = categories[j][i]
                    mapping[item] = i            
                    categories[j][i] = i
                self.col_to_mapping[cols[j]] = mapping
    
    def __update_categorical_mapping(self, num_cols) -> None:
        for i in range(num_cols):
            if i in self.col_to_mapping:
                self.mappings.append(self.col_to_mapping[i])
            else:
                self.mappings.append({})

    def __build_inverse_label_mapping(self, transformer) -> None:
        mapping = {}
        for i, cls in enumerate(transformer.classes_):
            mapping[i] = cls
        self.mappings = {"CategoricalString": mapping}
        if transformer.fill_unseen_labels:
            self.mappings["UnseenLabel"] = transformer.fill_label_value
    
    def update_func_name(self, func_name) -> None:
        self.transform_func = func_name

    def update_missing_metadata(self):
        # Replace "input" data shape with actual shapes from TVM module (could be "input" still).
        # For ambigious columns, there may be  some combination of "input_float" and "input_string"
        # (one of these or both) depending on which transfers are used. If input wasn't used, TVM
        # deletes it so this is why we use the TVM module as source of ground truth.
        input_shape = self.data_shape["input"]
        del self.data_shape["input"]
        for inp in self._relay_module_object["main"].params:
            self.data_shape[inp.name_hint] = input_shape
        super().update_missing_metadata()
        for input_config in self._metadata["Inputs"]:
            input_config["dtype"] = FLOAT_32
        for output_config in self._metadata["Outputs"]:
            output_config["name"] = "output"
            
    def load_model(self) -> None:
        model = self.__get_sklearn_model_from_model_artifacts()
        if self.transform_func == "transform":
            if len(self.data_shape["input"]) != 2:
                raise RuntimeError("InputConfiguration: InputShape for Sklearn model must have two dimensions, but got {}.".format(len(self.data_shape["input"])))
            if self.data_shape["input"][-1] == -1:
                raise RuntimeError("InputConfiguration: InputShape for Sklearn model must have a static value for the second dimension, equal to the number of input columns or features.")

            for _, transformer in model.feature_transformer.steps:
                if type(transformer).__name__ == "ColumnTransformer":
                    dropped_transformers = []
                    for name, pipeline, cols in transformer.transformers_:
                        if pipeline == "drop":
                            continue
                        dropped_transformers.append((name, pipeline, cols))
                        mod = pipeline.steps[0][1]
                        if type(mod).__name__ == "ThresholdOneHotEncoder" or type(mod).__name__ == "RobustOrdinalEncoder":
                            self.__build_numeric_mapping(mod.categories_, cols)
                        if name == "datetime_processing":
                            self.date_col = cols
                    transformer.transformers_ = dropped_transformers

            self.__update_categorical_mapping(self.data_shape["input"][-1]) 

        elif self.transform_func == "inverse_transform": 
            if type(model.target_transformer).__name__ == 'RobustLabelEncoder':
                self.__build_inverse_label_mapping(model.target_transformer)

        try:
            num_rows = self.data_shape["input"][0] if self.data_shape["input"][0] != -1 else relay.Any()
            num_cols = self.data_shape["input"][-1] if self.data_shape["input"][-1] != -1 else relay.Any()
            self._relay_module_object, self._params = relay.frontend.from_auto_ml(model, (num_rows, num_cols), FLOAT_32, self.transform_func)
            self._relay_module_object = self.dynamic_to_static(self._relay_module_object)
            self.update_missing_metadata()
            
        except OpError:
            raise
        except Exception as e:
            logger.exception("Failed to convert Scikit-Learn model. %s" % repr(e))
            msg = "InputConfiguration: TVM cannot convert Scikit-Learn model. Please make sure the framework you selected is correct. {}".format(e)
            raise RuntimeError(msg)

