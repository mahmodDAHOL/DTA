"""Useful functions and calsses utils."""

import pickle
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from torch import device
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import data as geo_data
from torch_geometric.data import Batch, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.utils.convert import to_networkx
from torchvision import transforms
from tqdm import tqdm

from .constants import project_path
from .emetrics import get_cindex
from .exception import CustomException
from .gnn import GNNNet
from .logger import logging


@dataclass(slots=True)
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
    model: GNNNet,
    device: device,
    train_loader: DataLoader,
    optimizer: Adam,
    epoch: int,
    batch_size: int,
    writer: SummaryWriter,
) -> None:
    """Train GNN of the training data."""
    model.train()
    LOG_INTERVAL = 10
    loss_fn = torch.nn.MSELoss()
    data_len = len(train_loader.dataset)  # type: ignore

    for idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        mol_x, mol_edge_index, mol_batch = \
            data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_batch = \
            data_pro.x, data_pro.edge_index, data_pro.batch

        output = model(
            mol_x, mol_edge_index, mol_batch,
            target_x, target_edge_index, target_batch)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        cindex = get_cindex(output, data_mol.y.view(-1, 1).float().to(device))
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("cindex/train", cindex, epoch)

        loss.backward()
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            logging.info(
              f"Train epoch: {epoch} [{idx*batch_size}/{data_len}] Loss: {loss.item()}"
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
            mol_x, mol_edge_index, mol_batch = \
                data_mol.x, data_mol.edge_index, data_mol.batch
            target_x, target_edge_index, target_batch = \
                data_pro.x, data_pro.edge_index, data_pro.batch

            output = model(
                mol_x, mol_edge_index, mol_batch,
                target_x, target_edge_index, target_batch)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def collate(data_list: list) -> tuple[BaseData, BaseData]:
    """Prepare the protein and drug pairs."""
    batch_a = Batch.from_data_list([data[0] for data in data_list])
    batch_b = Batch.from_data_list([data[1] for data in data_list])
    return batch_a, batch_b

def convert_graph_to_img(data: tuple[geo_data.Data, geo_data.Data],
                         graph_num: int) -> Path:
    """
    Get tuple of molecule and protien data and
    get their graph images then merge them in one image.
    """
    data_mol, data_pro = data
    data_mol, data_pro = to_networkx(data_mol), to_networkx(data_pro)
    mol_img = project_path.joinpath(f"graph_images/mol_graphs/img{graph_num}.png")
    pro_img = project_path.joinpath(f"graph_images/pro_graphs/img{graph_num}.png")
    mol_img.parent.mkdir(exist_ok=True, parents=True)
    pro_img.parent.mkdir(exist_ok=True, parents=True)
    nx.draw(data_mol, node_size=500, node_color="yellow",
            font_size=8, font_weight="bold")
    plt.savefig(mol_img)
    plt.clf()
    nx.draw(data_pro, node_size=20, node_color="blue",
            font_size=8, font_weight="bold")
    plt.savefig(pro_img)
    plt.clf()
    return merge_images(mol_img, pro_img, graph_num)


def merge_images(image_path_1: Path, image_path_2: Path, graph_num: int) -> Path:
    """Get two images path and number to be name of merged image."""
    pro_img = Image.open(str(image_path_1))
    mol_img = Image.open(str(image_path_2))
    width, height = pro_img.size
    mol_img = mol_img.resize((width, height))
    merged_image = Image.new("RGB", (width * 2, height))
    merged_image.paste(pro_img, (0, 0))
    merged_image.paste(mol_img, (width, 0))
    merged_path = project_path.joinpath(
        f"graph_images/merged_images/img{graph_num}.png")
    merged_path.parent.mkdir(exist_ok=True, parents=True)
    merged_image.save(str(merged_path))
    return merged_path

def plot_sample(train_loader: DataLoader, model: GNNNet, writer: SummaryWriter) -> None:
    """Plot graph structure, protien and molecule to tensorboard."""
    data_len = len(train_loader.dataset)  # type: ignore
    sample = random.choice(range(data_len))
    data = train_loader.dataset[sample]
    data_mol = data[0].to(device)
    data_pro = data[1].to(device)
    mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
    target_x, target_edge_index, target_batch = data_pro.x, \
                                                data_pro.edge_index, data_pro.batch
    mol_batch, target_batch = torch.Tensor([1]).type(torch.int64), \
                                torch.Tensor([1]).type(torch.int64)
    merged_img_path = convert_graph_to_img(data, sample)
    image = Image.open(str(merged_img_path))
    transform = transforms.Compose([transforms.Resize((1280, 1280)),
                                    transforms.PILToTensor()])
    img_as_tensor = transform(image)
    writer.add_image("protien and molecule graph", img_as_tensor)
    writer.add_graph(model,
                     [mol_x, mol_edge_index, mol_batch,
                      target_x, target_edge_index, target_batch]
                     )


def save_model(
    dataset_name: str, epoch: int, models_dir: Path, model: GNNNet, period: int
) -> int:
    """Save the new model and remove old one."""
    model_file_name = f"{dataset_name}_epoch-{epoch}.model"
    new_model_file_path = models_dir.joinpath(model_file_name)
    torch.save(model.state_dict(), new_model_file_path)
    if epoch == 0:
        return 0
    old_model_file_name = f"{dataset_name}_epoch-{epoch-period}.model"
    old_model_file_path = models_dir.joinpath(old_model_file_name)
    old_model_file_path.unlink(missing_ok=True)
    return 1
