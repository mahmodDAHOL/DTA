"""Train GNN model."""
import shelve

import torch
import typer
from torch.utils.data import DataLoader

import constants
from data_process import create_train_data
from emetrics import get_mse
from gnn import GNNNet
from logger import logging
from utils import collate, predicting, train


def main(
    cuda_name: str = typer.Option(..., prompt=True),
    fold_number: int = typer.Option(..., prompt=True),
) -> None:
    """Training."""
    datasets = ["davis", "kiba"]

    BATCH_SIZE = 512
    LR = 0.001
    NUM_EPOCHS = 2000

    models_dir = constants.project_path.joinpath("models")
    results_dir = constants.project_path.joinpath("results")

    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device(cuda_name if USE_CUDA else "cpu")
    model = GNNNet()
    model.to(device)
    model_st = GNNNet.__name__
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for dataset_name in datasets:
        model_file_name = f"model_{model_st}_{dataset_name}_{fold_number}.model"
        dataset_path = (
            constants.davis_dataset_path
            if dataset_name == "davis"
            else constants.kiba_dataset_path
        )

        train_data, valid_data = create_train_data(dataset_path, fold=fold_number)

        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
        )
        valid_loader = DataLoader(
            valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
        )

        best_mse = 1000
        best_test_mse = 1000
        best_epoch = -1
        model_file_path = models_dir.joinpath(model_file_name)

        for epoch in range(1, NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch)
            logging.info("predicting for valid data")
            G, P = predicting(model, device, valid_loader)
            val = get_mse(G, P)
            logging.info(f"valid result: {val} {best_mse}")
            if val < best_mse:
                best_mse = val
                best_epoch = epoch
                torch.save(model.state_dict(), model_file_path)
                logging.info(
                    f"""rmse improved at epoch {best_epoch}; {best_test_mse}
                    {best_mse} {model_st} {dataset_path} {fold_number}"""
                )
            else:
                logging.info(
                    f"""No improvement since epoch{best_epoch}; {best_test_mse}
                    {best_mse} {model_st} {dataset_path} {fold_number}"""
                )


if __name__ == "__main__":
    typer.run(main)
