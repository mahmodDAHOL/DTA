"""Train GNN model."""
import torch
import typer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import constants
from data_process import create_train_data
from emetrics import get_cindex, get_mse
from gnn import GNNNet
from logger import logging
from utils import collate, predicting, train


def main(
    cuda_name: str = typer.Option(..., prompt=True),
    dataset_name: str = typer.Option(..., prompt=True),
    fold_number: int = typer.Option(..., prompt=True),
) -> None:
    """Training."""
    BATCH_SIZE = 512
    LR = 0.001
    NUM_EPOCHS = 2000
    PERIOD = 5

    models_dir = Path(f"/content/drive/MyDrive/models/{fold_number}")
    results_dir = constants.project_path.joinpath("results")

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device(cuda_name if USE_CUDA else "cpu")
    model = GNNNet()

    old_model_path = next(models_dir.glob("*")) if list(models_dir.glob("*")) else None
    if old_model_path:
        epoch_start_num = int(old_model_path.stem.split('-')[-1])
        model.load_state_dict(torch.load(old_model_path))
    else:
        epoch_start_num = 0
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    run_path = Path(f"/content/drive/MyDrive/runs/{dataset_name}_fold{fold_number}")
    writer = SummaryWriter(run_path)

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

    for epoch in range(epoch_start_num, NUM_EPOCHS):
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
            if epoch % PERIOD == 0:
                model_file_name = f"{dataset_name}_epoch-{epoch}.model"
                new_model_file_path = models_dir.joinpath(model_file_name)
                torch.save(model.state_dict(), new_model_file_path)
                if epoch == 0:
                    continue
                old_model_file_name = f"{dataset_name}_epoch-{epoch-PERIOD}.model"
                old_model_file_path = models_dir.joinpath(old_model_file_name)
                old_model_file_path.unlink(missing_ok=True)
            logging.info(f"rmse improved at epoch {best_epoch} | {best_mse=:.4f}")
        else:
            logging.info(f"No improvement since epoch {best_epoch} | {best_mse=:.4f}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    typer.run(main)
