import tensorflow as tf

from typing import List, Dict
from pathlib import Path
from enum import Enum
from collections import OrderedDict
from .abstract_model_helper import ModelHelper


class TFModelFormat(Enum):
    FrozenGraphModel = "frozen_graph_model"
    SavedModel = "saved_model"


class TFModelHelper(ModelHelper):

    UNLIKELY_OUTPUT_TYPES = {"Const", "Assign", "NoOp", "Placeholder"}

    def __init__(self, model_path: str, data_shape: Dict[str, List[int]]) -> None:
        super(TFModelHelper, self).__init__(model_path)
        tf.enable_eager_execution()
        self.__input_tensor_names = []
        self.__output_tensor_names = []
        self.__input_tensors = []
        self.__output_tensors = []
        self.__data_shape = data_shape

    @property
    def model_type(self) -> bool:
        if self.model_path.is_file() and self.model_path.suffix in ['.pb', '.pbtxt']:
            return TFModelFormat.FrozenGraphModel
        elif self.model_path.is_dir() and self.model_path.joinpath('variables').exists():
            return TFModelFormat.SavedModel
        else:
            raise Exception(f"Encountered invalid model file format. {self.model_path.as_posix}")

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

    def __get_graph_from_frozen_graph_model(self) -> tf.Graph:
        with tf.gfile.GFile(self.model_path.as_posix(), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            # Setting the name to empty string will ensure that the the prefix will be empty string.
            # it is not requred since we are not modifying the graph here. Prefix is only used for
            # distinguishing between the nodes of the imported graph and the modified nodes.
            tf.import_graph_def(graph_def, name="")
        return graph

    def __extract_input_and_output_tensors_from_frozen_graph(self) -> None:
        # https://github.com/neo-ai/neo-ai-dlr/blob/master/python/dlr/tf_model.py#L37
        tf.reset_default_graph()
        graph = self.__get_graph_from_frozen_graph_model()
        input_tensors = OrderedDict()
        output_tensors = OrderedDict()

        for op in graph.get_operations():
            if op.type == 'Placeholder' and op.inputs.__len__() == 0 and op.outputs.__len__() == 1:
                input_tensors[op.outputs[0].name] = op.outputs[0]

            if op.type not in self.UNLIKELY_OUTPUT_TYPES and op.outputs.__len__() == 1:
                output_tensors[op.outputs[0].name] = op.outputs[0]

        output_tensor_names = output_tensors.keys()

        for op in graph.get_operations():
            for in_t in op.inputs:
                if in_t.name in output_tensor_names:
                    output_tensors.pop(in_t.name)
            for cont_op in op.control_inputs:
                for out_t in cont_op.outputs:
                    if out_t.name in output_tensor_names:
                        output_tensors.pop(out_t.name)


        tf.reset_default_graph()

        self.__input_tensor_names = list(input_tensors.keys())
        self.__output_tensor_names = list(output_tensors.keys())
        self.__input_tensors = list(input_tensors.values())
        self.__output_tensors = list(output_tensors.values())

    def __get_tag_set(self) -> str:
        try:
            from tensorflow.contrib.saved_model.python.saved_model import reader
        except ImportError:
            raise ImportError(
                "InputConfiguration: Unable to import saved_model.reader which is "
                "required to get tag set from saved model.")

        tag_sets = reader.get_saved_model_tag_sets(self.model_path.as_posix())
        return tag_sets[0]

    def __extract_input_and_output_tensors_from_saved_model(self) -> None:
        # https://github.com/apache/incubator-tvm/blob/master/python/tvm/relay/frontend/tensorflow_parser.py#L73
        tf.reset_default_graph()

        tags = self.__get_tag_set()
        input_tensors = OrderedDict()
        output_tensors = OrderedDict()

        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, tags, self.model_path.as_posix())
            for sig_def in meta_graph_def.signature_def.values():
                for input_tensor in sig_def.inputs.values():
                    input_tensors[input_tensor.name] = tf.get_default_graph().get_tensor_by_name(input_tensor.name)
                for output_tensor in sig_def.outputs.values():
                    output_tensors[output_tensor.name] = tf.get_default_graph().get_tensor_by_name(output_tensor.name)

        tf.reset_default_graph()

        self.__input_tensor_names = list(input_tensors.keys())
        self.__output_tensor_names = list(output_tensors.keys())
        self.__input_tensors = list(input_tensors.values())
        self.__output_tensors = list(output_tensors.values())

    def __extract_input_and_output_tensors_from_saved_model_v2(self) -> None:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        tags = self.__get_tag_set()
        loaded = tf.compat.v2.saved_model.load(self.model_path.as_posix(), tags=tags)
        for shape in self.__data_shape.values():
            tensor_spec = tf.TensorSpec(tuple(shape))
        if len(loaded.signatures) == 0:
            f = loaded.__call__.get_concrete_function(tensor_spec)
        elif 'serving_default' in loaded.signatures:
            f = loaded.signatures['serving_default']
        else:
            f = loaded.signatures[list(loaded.signatures.keys())[0]]
        frozen_func = convert_variables_to_constants_v2(f, lower_control_flow=True)

        for tensor in frozen_func.inputs:
            self.__input_tensor_names.append(tensor.name)
        for tensor in frozen_func.outputs:
            self.__output_tensor_names.append(tensor.name)

        self.__input_tensors = frozen_func.inputs
        self.__output_tensors = frozen_func.outputs

    def extract_input_and_output_tensors(self, user_shape_dict=None) -> None:
        if self.model_type == TFModelFormat.SavedModel:
            self.__extract_input_and_output_tensors_from_saved_model()
        else:
            self.__extract_input_and_output_tensors_from_frozen_graph()

    def extract_input_and_output_tensors_v2(self, user_shape_dict=None) -> None:
        if self.model_type == TFModelFormat.SavedModel:
            self.__extract_input_and_output_tensors_from_saved_model_v2()
        else:
            self.__extract_input_and_output_tensors_from_frozen_graph()

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

    def get_tf_graph_from_graph_model_v2(self):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        tags = self.__get_tag_set()
        loaded = tf.compat.v2.saved_model.load(self.model_path.as_posix(), tags=tags)
        for shape in self.__data_shape.values():
            tensor_spec = tf.TensorSpec(tuple(shape))
            break
        if len(loaded.signatures) == 0:
            f = loaded.__call__.get_concrete_function(tensor_spec)
        elif 'serving_default' in loaded.signatures:
            f = loaded.signatures['serving_default']
        else:
            f = loaded.signatures[list(loaded.signatures.keys())[0]]
        frozen_func = convert_variables_to_constants_v2(f, lower_control_flow=True)
        tf_graph = frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_graph
