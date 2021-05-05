from collections import OrderedDict
from .abstract_model_helper import ModelHelper
from tflite.Tensor import Tensor
from tflite.Model import Model
from tflite.TensorType import TensorType
from typing import List

class TFLiteModelHelper(ModelHelper):
    TFLITE_TENSOR_TYPE_TO_DTYPE = {}
    TFLITE_TENSOR_TYPE_TO_DTYPE[TensorType.UINT8] = "uint8"
    TFLITE_TENSOR_TYPE_TO_DTYPE[TensorType.FLOAT32] = "float32"
    TFLITE_TENSOR_TYPE_TO_DTYPE[TensorType.INT32] = "int32"
    TFLITE_TENSOR_TYPE_TO_DTYPE[TensorType.INT64] = "int64"

    def __init__(self, model_path: str) -> None:
        super(TFLiteModelHelper, self).__init__(model_path)
        self.__tflite_model = None
        self.__input_dtypes_dict = {}
        self.__input_tensors = []
        self.__output_tensors = []

    @property
    def input_tensors(self) -> List[Tensor]:
        return self.__input_tensors

    @property
    def output_tensors(self) -> List[Tensor]:
        return self.__output_tensors

    @property
    def input_dtypes_dict(self) -> {str: str}:
        dtypes_inputs = {}
        for tensor in self.input_tensors:
            dtypes_inputs[tensor.Name().decode("utf-8")] = self.TFLITE_TENSOR_TYPE_TO_DTYPE[tensor.Type()]
        return dtypes_inputs

    @property
    def tflite_model(self) -> Model:
        return self.__tflite_model

    @staticmethod
    def get_supported_tflite_input_tensor_type() -> List[TensorType]:
        return [TensorType.FLOAT32, TensorType.UINT8]

    def load_model(self) -> None:
        try:
            import tflite.Model
        except ImportError:
            raise ImportError("The tflite package must be installed")

        with open(self.model_path, "rb") as f:
            tflite_model_buf = f.read()

        try:
            import tflite
            self.__tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model
            self.__tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    def extract_input_and_output_tensors(self, user_shape_dict=None) -> None:
        if user_shape_dict is None:
            raise Exception("Model input names and shapes must be provided")

        subgraph = self.tflite_model.Subgraphs(0)
        input_tensors = self.__get_input_tensors(subgraph, user_shape_dict)
        output_tensors = self.__get_output_tensors(subgraph)
        self.__input_tensors = list(input_tensors.values())
        self.__output_tensors = list(output_tensors.values())

    def __get_input_tensors(self, subgraph, user_shape_dict):
        input_tensors = OrderedDict()
        model_inputs = subgraph.InputsAsNumpy()
        for model_input in model_inputs:
            model_input_tensor = subgraph.Tensors(model_input)
            model_input_name = model_input_tensor.Name().decode("utf-8")
            if model_input_tensor.Type() not in self.get_supported_tflite_input_tensor_type():
                raise Exception("Unsupported input data type for input {} with tflite tensor type {}".format(model_input_name, str(model_input_tensor.Type())))

            if model_input_name not in user_shape_dict:
                raise Exception("Please specify all input layers in data_shape.")
            input_tensors[model_input_name] = model_input_tensor
        return input_tensors

    def __get_output_tensors(self, subgraph):
        output_tensors = OrderedDict()
        model_outputs = subgraph.OutputsAsNumpy()
        for model_output in model_outputs:
            model_output_tensor = subgraph.Tensors(model_output)
            model_output_name = model_output_tensor.Name().decode("utf-8")
            output_tensors[model_output_name] = model_output_tensor
        return output_tensors

    def get_metadata(self) -> {str: List}:
        return {
            "Inputs": [
                {'name': tensor.Name().decode("utf-8"), 'dtype': self.TFLITE_TENSOR_TYPE_TO_DTYPE[tensor.Type()], 'shape': tensor.ShapeAsNumpy().tolist()}
                for tensor in self.input_tensors
            ],
            "Outputs": [
                {'name': tensor.Name().decode("utf-8"), 'dtype':  self.TFLITE_TENSOR_TYPE_TO_DTYPE[tensor.Type()], 'shape': tensor.ShapeAsNumpy().tolist()}
                for tensor in self.output_tensors
            ]
        }
