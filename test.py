"""Contain functions for test the model from test dataset."""
import shelve
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

import constants
from data_process import create_test_data
from emetrics import (
    get_ci,
    get_cindex,
    get_mse,
    get_pearson,
    get_rm2,
    get_rmse,
    get_spearman,
)
from gnn import GNNNet
from utils import collate, predicting

project_path = Path.cwd()
model_st = GNNNet.__name__


def calculate_metrics(labels: np.ndarray, predicteds: np.ndarray, dataset: str) -> None:
    """Calculate the diffrence between the actual and predicted values."""
    cindex = get_cindex(labels, predicteds)  # DeepDTA
    cindex2 = get_ci(labels, predicteds)  # GraphDTA
    rm2 = get_rm2(labels, predicteds)  # DeepDTA
    mse = get_mse(labels, predicteds)
    pearson = get_pearson(labels, predicteds)
    spearman = get_spearman(labels, predicteds)
    rmse = get_rmse(labels, predicteds)

    print("metrics for ", dataset)
    print("cindex:", cindex)
    print("cindex2", cindex2)
    print("rm2:", rm2)
    print("mse:", mse)
    print("pearson", pearson)

    result_file_name = project_path.joinpath(f"results\result_{model_st}_{dataset}.txt")
    result_str = f"""dataset \r\n rmse:{str(rmse)}  mse:{str(mse)}
                     pearson:{str(pearson)} spearman:{str(spearman)}
                     ci:{str(cindex)} rm2:{str(rm2)}"""
    print(result_str)
    result_file_name.open("w").writelines(result_str)


def plot_density(
    labels: np.ndarray, predicteds: np.ndarray, fold: int = 0, dataset: str = "davis"
) -> None:
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.scatter(predicteds, labels, color="blue", s=40)
    plt.title("density of " + dataset, fontsize=30, fontweight="bold")
    plt.xlabel("predicted", fontsize=30, fontweight="bold")
    plt.ylabel("measured", fontsize=30, fontweight="bold")
    if dataset == "davis":
        plt.plot([5, 11], [5, 11], color="black")
    else:
        plt.plot([6, 16], [6, 16], color="black")
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight="bold")
    plt.savefig(
        project_path.joinpath("results", f"{dataset}_{str(fold)}.png"),
        dpi=500,
        bbox_inches="tight",
    )


def main(
    dataset_name: str = typer.Option(..., prompt=True),
    cuda_name: str = typer.Option(..., prompt=True),
    fold_number: int = typer.Option(..., prompt=True),
) -> None:
    dataset_path = (
        constants.davis_dataset_path
        if dataset_name == "davis"
        else constants.kiba_dataset_path
    )
    TEST_BATCH_SIZE = 512
    models_dir = project_path.joinpath("models")

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model_file_name = models_dir.joinpath(f"model_{model_st}_{dataset_name}.model")

    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))

    processed_data_path = dataset_path.joinpath("processedDB")
    if processed_data_path:
        with shelve.open(str(processed_data_path)) as db:
            test_data_name = f"test-{dataset_name}-{fold_number}"
            test_data = db[test_data_name]
    else:
        test_data = create_test_data(dataset_path)

    test_loader = DataLoader(
        test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    Y, P = predicting(model, device, test_loader)
    calculate_metrics(Y, P, dataset_name)


if __name__ == "__main__":
    typer.run(main)
