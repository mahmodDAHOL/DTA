"""Train GNN model."""
import torch
import typer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants
from data_process import create_train_data
from emetrics import get_mse, get_cindex
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
        writer = SummaryWriter(f'runs/{dataset_name}_fold{fold_number}')
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
        best_epoch = -1
        model_file_path = models_dir.joinpath(model_file_name)

        for epoch in range(1, NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch, writer)
            G, P = predicting(model, device, valid_loader)
            val_loss = get_mse(G, P)
            cindex = get_cindex(G, P)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("cindex/validation", cindex, epoch)

            logging.info(f"validation result: {val_loss:.4f} | {best_mse=:.4f}")
            if val_loss < best_mse:
                best_mse = val_loss
                best_epoch = epoch
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), model_file_path)
                logging.info(f"rmse improved at epoch {best_epoch} | {best_mse=:.4f}")
            else:
                logging.info(f"No improvement since epoch {best_epoch} | {best_mse=:.4f}")

    writer.flush()
    writer.close()

if __name__ == "__main__":
    typer.run(main)
