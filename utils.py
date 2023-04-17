"""Define DTADataset in order to pickle can find
that class definition when load the objects from files. """

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric import data as geo_data
from torch_geometric.data import InMemoryDataset

from src.exception import CustomException


@dataclass()
class Graph:
    """Data structure represents graph elements and info."""

    size: np.ndarray
    features: np.ndarray
    edge_index: np.ndarray

class DTADataset(InMemoryDataset):
    """
    Dataset class from pytorch inherited from InMemoryDataset
    to add its functionality.
    """

    def __init__(
        self,
        root: str,
        drugs: np.ndarray,
        y: np.ndarray,
        target_key: np.ndarray,
        smile_graph_dict: dict[str, Graph],
        target_graph_dict: dict[str, Graph],
    ) -> None:
        super().__init__(root)
        self.dataset_path = Path(root)
        self.process(drugs, target_key, y, smile_graph_dict, target_graph_dict)
        self.dataset_name = self.dataset_path.stem

    @property
    def processed_file_names(self) -> None:
        pass

    def download(self) -> None:
        pass

    def _download(self) -> None:
        pass

    def _process(self) -> None:
        pass

    def process(
        self,
        drugs: np.ndarray,
        target_keys: np.ndarray,
        y: np.ndarray,
        smile_graph_dict: dict[str, Graph],
        target_graph_dict: dict[str, Graph],
    ) -> None:
        if not all([len(drugs) == len(target_keys), len(drugs) == len(y)]):
            e = "The three lists must be the same length!"
            raise CustomException(e, sys)
        data_list_mol = []
        data_list_pro = []
        data_len = len(drugs)
        for i in range(data_len):
            smile = drugs[i]
            tar_key = target_keys[i]
            label = y[i]
            # convert SMILES to molecular representation using rdkit
            smile_graph: Graph = smile_graph_dict[str(smile)]
            target_graph: Graph = target_graph_dict[str(tar_key)]

            GCNData_mol = _convert(smile_graph, float(label))
            GCNData_pro = _convert(target_graph, float(label))
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        self.data_mol: list[geo_data.Data] = data_list_mol
        self.data_pro: list[geo_data.Data] = data_list_pro

    def __len__(self) -> int:
        return len(self.data_mol)

    def __getitem__(self, idx: int) -> tuple[geo_data.Data, geo_data.Data]:
        return self.data_mol[idx], self.data_pro[idx]  # type: ignore


def _convert(graph: Graph, labels: float) -> geo_data.Data:
    gcn_data = geo_data.Data(
        x=torch.Tensor(graph.features),
        edge_index=torch.LongTensor(graph.edge_index).transpose(1, 0),
        y=torch.FloatTensor([labels]),
    )
    gcn_data.__setitem__("size", torch.LongTensor(graph.size))

    return gcn_data