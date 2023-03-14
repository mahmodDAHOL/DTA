"""Usefule functions and calsses utils."""

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import device
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric import data as geo_data
from torch_geometric.data import Batch, InMemoryDataset
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from exception import CustomException
from gnn import GNNNet
from logger import logging


@dataclass
class Graph:
    """Data structure represents graph elements and info."""

    size: np.ndarray
    features: np.ndarray
    edge_index: np.ndarray


# initialize the dataset
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
        self.processed_data_path = self.dataset_path.joinpath("processed")
        self.processed_data_path.mkdir(exist_ok=True)

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


def to_graph(
    dataset_path: Path, file_name: str, dict_data: dict, to_graph_func: Callable
) -> dict[str, Graph]:
    """Convert proteins or smiles data to graph objects."""
    processed_file = dataset_path.joinpath(file_name)
    data_type = file_name.split("_")[0]
    if not processed_file.exists():
        if data_type == "smiles":
            logging.info("converting smiles into graphs...")
            graph_dict = {}
            for key in tqdm(
                dict_data.values(), desc=f"converting {data_type} to graphs"
            ):
                graph_dict[key] = to_graph_func(key)
        else:
            logging.info("converting proteins into graphs..")
            graph_dict = {}
            for key in tqdm(dict_data.keys(), desc=f"converting {data_type} to graphs"):
                graph = to_graph_func(dataset_path, key, dict_data[key])
                if graph != 0:
                    graph_dict[key] = to_graph_func(dataset_path, key, dict_data[key])
        logging.info(f"saving {data_type} in {processed_file}")
        pickle.dump(graph_dict, processed_file.open("wb"))
    else:
        logging.info(f"loading {data_type} from {processed_file}")
        graph_dict = pickle.load(processed_file.open("rb"))

    return graph_dict


# training function at each epoch
def train(
    model: GNNNet, device: device, train_loader: DataLoader, optimizer: Adam, epoch: int
) -> None:
    """Train GNN of the training data."""
    model.train()
    LOG_INTERVAL = 10
    BATCH_SIZE = 512
    loss_fn = torch.nn.MSELoss()
    data_len = len(train_loader.dataset)  # type: ignore
    for idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            logging.info(
                f"Train epoch: {epoch} [{idx*BATCH_SIZE}/{data_len} \
                ({100*idx/len(train_loader)}%)]\tLoss: {loss.item()}"
            )


def predicting(
    model: GNNNet, device: torch.device, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Predicting labels of data."""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def collate(data_list: list) -> tuple[BaseData, BaseData]:
    """Prepare the protein and drug pairs."""
    batch_a = Batch.from_data_list([data[0] for data in data_list])
    batch_b = Batch.from_data_list([data[1] for data in data_list])
    return batch_a, batch_b
