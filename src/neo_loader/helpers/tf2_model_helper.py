import tensorflow as tf

from typing import List, Dict
from enum import Enum
from .abstract_model_helper import ModelHelper


class TFModelFormat(Enum):
    FrozenGraphModel = "frozen_graph_model"
    SavedModel = "saved_model"


class TF2ModelHelper(ModelHelper):

    UNLIKELY_OUTPUT_TYPES = {"Const", "Assign", "NoOp", "Placeholder"}

    def __init__(self, model_path: str, data_shape: Dict[str, List[int]]) -> None:
        super(TF2ModelHelper, self).__init__(model_path)
        self.__input_tensor_names = []
        self.__output_tensor_names = []
        self.__input_tensors = []
        self.__output_tensors = []
        self.__data_shape = data_shape
        self.__frozen_func = None
        self.__model_tf_version = "not found"

    @property
    def model_type(self) -> bool:
        # Frozen graph has been sort of being deprecated by TensorFlow2
        if self.model_path.is_dir() and self.model_path.joinpath('variables').exists():
            return TFModelFormat.SavedModel
        else:
            raise Exception(f"Encountered invalid model file format. Support saved model format in TensorFlow2. {self.model_path.as_posix}")

    @property
    def input_tensors(self) -> List[tf.Tensor]:
        return self.__input_tensors

    @property
    def output_tensors(self) -> List[tf.Tensor]:
        return self.__output_tensors

    @property
    def input_tensor_names(self) -> List[str]:
        return self.__input_tensor_names

    @property
    def output_tensor_names(self) -> List[str]:
        return self.__output_tensor_names

    def __get_tag_set(self) -> str:
        try:
            from tensorflow.python.tools import saved_model_utils
        except ImportError:
            raise ImportError(
                "InputConfiguration: Unable to import saved_model_utils which is "
                "required to get tag set from saved model.")

        tag_sets = saved_model_utils.get_saved_model_tag_sets(self.model_path.as_posix())
        return tag_sets[0]

    def __extract_input_and_output_tensors_from_saved_model(self) -> None:
        self.__frozen_func = self.__get_frozen_func_from_graph_model()
        for tensor in self.__frozen_func.inputs:
            self.__input_tensor_names.append(tensor.name)
        for tensor in self.__frozen_func.outputs:
            self.__output_tensor_names.append(tensor.name)

        self.__input_tensors = self.__frozen_func.inputs
        self.__output_tensors = self.__frozen_func.outputs

    def extract_input_and_output_tensors(self, user_shape_dict=None) -> None:
        if self.model_type == TFModelFormat.SavedModel:
            self.__extract_input_and_output_tensors_from_saved_model()

    def get_metadata(self) -> {str: List}:
        # We need to strip the trailing ":0" from the input names since DLR cannot handle it.
        # RelayTVM handles it gracefully and we can pass the names as is to relay.
        # https://github.com/apache/incubator-tvm/blob/master/python/tvm/relay/frontend/tensorflow.py#L2860
        return {
            "Inputs": [
                {'name': tensor.name.replace(":0", ""), 'dtype': tensor.dtype.name, 'shape': tensor.shape.as_list()}
                for tensor in self.input_tensors
            ],
            "Outputs": [
                {'name': tensor.name, 'dtype': tensor.dtype.name, 'shape': tensor.shape.as_list() if tensor.shape else None}
                for tensor in self.output_tensors
            ]
        }

    def __get_frozen_func_from_graph_model(self):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        tags = self.__get_tag_set()
        loaded = tf.saved_model.load(self.model_path.as_posix(), tags=tags)
        self.__model_tf_version = loaded.tensorflow_version
        for shape in self.__data_shape.values():
            tensor_spec = tf.TensorSpec(tuple(shape))
            break
        if len(loaded.signatures) == 0:
            f = loaded.__call__.get_concrete_function(tensor_spec)
        elif 'serving_default' in loaded.signatures:
            f = loaded.signatures['serving_default']
        else:
            f = loaded.signatures[list(loaded.signatures.keys())[0]]
        frozen_func = convert_variables_to_constants_v2(f, lower_control_flow=False)
        return frozen_func

    def get_tf_graph_from_graph_model(self):
        tf_graph = self.__frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_graph

    def get_tensorflow_version(self) -> str:
        return self.__model_tf_version
