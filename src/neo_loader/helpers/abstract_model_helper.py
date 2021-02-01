from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class ModelHelper(ABC):

    def __init__(self, model_path: str) -> None:
        self.__model_path = model_path

    @property
    def model_path(self) -> Path:
        return Path(self.__model_path).resolve()

    @property
    @abstractmethod
    def input_tensors(self) -> List[object]:
        pass

    @property
    @abstractmethod
    def output_tensors(self) -> List[object]:
        pass

    @abstractmethod
    def get_metadata(self) -> {str: List}:
        pass

    @abstractmethod
    def extract_input_and_output_tensors(self, user_shape_dict=None) -> None:
        pass
