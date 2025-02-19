from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

T = TypeVar('T')


class IRepository(ABC, Generic[T]):
    @abstractmethod
    def get_all(self) -> (T | List[T]):
        pass

    @abstractmethod
    def create(self, data: T) -> None:
        pass

    @abstractmethod
    def update(self, data: T) -> None:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass
