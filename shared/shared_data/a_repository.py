from typing import TypeVar, List
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from .interfaces.i_repository import IRepository


def read_file(file_path: str) -> DataFrame:
    try:
        file_format = Path(file_path).suffix.lower()
        
        if file_format == '.csv':
            return pd.read_csv(file_path)
        if file_format == '.xlsx':
            return pd.read_excel(file_path)
    except Exception as e:
        raise Exception("File format not supported") from e
            

class RepositoryBase(IRepository):
    def __init__(self, file_path):
        self._df = read_file(file_path=file_path)            
    
    def get_all(self) -> DataFrame:
        return self._df

    def create(self, data: dict) -> None:
        self._df = self._df.append(data, ignore_index=True)

    def update(self, row: int, column: int, value: (str | int)) -> None:
        self._df = self._df.at[row, column] = value

    def delete(self, indices: (int | List[int])) -> None:
        self._df = self._df.drop(index=indices, ignore_index=True)