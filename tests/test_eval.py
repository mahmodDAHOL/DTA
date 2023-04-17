"""Contain functions for test the model from test dataset."""
import sys

import torch
from torch.utils.data import DataLoader

from src.exception import CustomException
from src.gnn import GNNNet
from src.utils import collate, predicting

from src import constants
from src.constants import project_path
from src.data_process import create_test_data
from src.eval import calculate_metrics

model_st = GNNNet.__name__

def test_eval() -> None:
    fold_number = 1
    cuda_name = ""
    dataset_name = "davis"

    dataset_path = (
        constants.project_path.joinpath("tests/data/davis")
        if dataset_name == "davis"
        else constants.project_path.joinpath("tests/data/kiba")
    )
    models_dir = project_path.joinpath(f"models/{fold_number}")
    model_path = next(models_dir.glob("*")) if list(models_dir.glob("*")) else None
    if not model_path.exists():
        e = "there is no model to test it."
        raise CustomException(e, sys)

    TEST_BATCH_SIZE = 16

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    test_data = create_test_data(dataset_path)

    test_loader = DataLoader(
        test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    Y, P = predicting(model, device, test_loader)
    calculate_metrics(Y, P, dataset_name, fold_number)

