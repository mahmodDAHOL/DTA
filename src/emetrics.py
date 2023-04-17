"""Contains all functions for evaluating the model performance."""
from math import sqrt

import numpy as np
from scipy import stats


def get_cindex(label: np.ndarray, predicted: np.ndarray) -> float:
    summ = 0
    pair = 0

    for i in range(1, len(label)):
        for j in range(0, i):
            if i is not j and label[i] > label[j]:
                pair += 1
                summ += 1 * (predicted[i] > predicted[j]) + 0.5 * (
                    predicted[i] == predicted[j]
                )

    if pair != 0:
        return summ / pair
    return 0


def r_squared_error(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.array([np.mean(y_obs) for y in y_obs])
    y_pred_mean = np.array([np.mean(y_pred) for y in y_pred])

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs: list, y_pred: list) -> float:
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.array([np.mean(y_obs) for y in y_obs])
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig: np.ndarray, ys_line: np.ndarray) -> float:
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rmse(y: np.ndarray, f: np.ndarray) -> float:
    return sqrt(((y - f) ** 2).mean(axis=0))


def get_mse(y: np.ndarray, f: np.ndarray) -> float:
    return ((y - f) ** 2).mean(axis=0)


def get_pearson(y: np.ndarray, f: np.ndarray) -> float:
    return np.corrcoef(y, f)[0, 1]


def get_spearman(y: np.ndarray, f: np.ndarray) -> float:
    return stats.spearmanr(y, f)[0]


def get_ci(y: np.ndarray, f: np.ndarray) -> float:
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return S / z
