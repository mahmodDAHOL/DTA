import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import device
from torch.optim import Adam
from torch.utils.data import DataLoader

# torch_geometric.data provides useful functionality for analyzing graph structures, and provides basic PyTorch tensor functionalities.
from torch_geometric import data as DATA
from torch_geometric.data import Batch, InMemoryDataset

from gnn import GNNNet


@dataclass
class Graph:
    size: np.ndarray
    features: np.ndarray
    edge_index: np.ndarray


# initialize the dataset
class DTADataset(InMemoryDataset):
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

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(
        self,
        drugs: np.ndarray,
        target_key: np.ndarray,
        y: np.ndarray,
        smile_graph_dict: dict[str, Graph],
        target_graph_dict: dict[str, Graph],
    ):
        assert len(drugs) == len(target_key) and len(drugs) == len(
            y
        ), "The three lists must be the same length!"
        data_list_mol = []
        data_list_pro = []
        data_len = len(drugs)
        for i in range(data_len):
            smiles = drugs[i]
            tar_key = target_key[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            smile_graph: Graph = smile_graph_dict[smiles]
            target_graph: Graph = target_graph_dict[tar_key]

            GCNData_mol = _convert(smile_graph, labels)
            GCNData_pro = _convert(target_graph, labels)

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        self.data_mol: list[DATA.Data] = data_list_mol
        self.data_pro: list[DATA.Data] = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]  # type: ignore

    def save(self, fold_number: int, type: str):
        dataset_name = self.dataset_path.stem
        processed_data_path = self.dataset_path.joinpath(f"processed/fold{fold_number}")
        file_path = processed_data_path.joinpath(
            f"{type}-{dataset_name}-{fold_number}.pl"
        )

        folder_path = file_path.parent
        folder_path.mkdir(exist_ok=True)
        for folder in reversed(list(folder_path.parents)):
            folder.mkdir(exist_ok=True)
            if not file_path.exists():
                pickle.dump(self, open(file_path, "wb"))


def _convert(graph: Graph, labels) -> DATA.Data:
    GCNData = DATA.Data(
        x=torch.Tensor(graph.features),
        edge_index=torch.LongTensor(graph.edge_index).transpose(1, 0),
        y=torch.FloatTensor([labels]),
    )
    GCNData.__setitem__("size", torch.LongTensor(graph.size))

    return GCNData


# training function at each epoch
def train(
    model: GNNNet, device: device, train_loader: DataLoader, optimizer: Adam, epoch: int
):

    print(f"Training on {len(train_loader.dataset)} samples...")  # type: ignore
    model.train()
    LOG_INTERVAL = 10
    BATCH_SIZE = 512
    loss_fn = torch.nn.MSELoss()
    data_len = len(train_loader.dataset)  # type: ignore
    # train_loader :: (DataBatch(x=[16206, 78], edge_index=[2, 52022], y=[512], c_size=[512], batch=[16206], ptr=[513]),
    #   DataBatch(x=[404817, 54], edge_index=[2, 2853143], y=[512], target_size=[512], batch=[404817], ptr=[513]))
    for idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            print(
                f"Train epoch: {epoch} [{idx*BATCH_SIZE}/{data_len} ({100*idx/len(train_loader)}%)]\tLoss: {loss.item()}"
            )


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f"Make prediction for {len(loader.dataset)} samples...")
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB
