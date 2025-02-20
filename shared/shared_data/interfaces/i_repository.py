from abc import ABC, abstractmethod
from typing import List
from pandas import DataFrame


class IRepository(ABC):
    @abstractmethod
    def get_all(self) -> DataFrame:
        pass

    @abstractmethod
    def create(self, data: dict) -> None:
        pass

    @abstractmethod
    def update(self, row: int, column: int, value: (str | int)) -> None:
        pass

    @abstractmethod
    def delete(self, indices: (int | List[int])) -> None:
        pass
