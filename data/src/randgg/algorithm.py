import pandas as pd
import numpy as np
import math

from scipy.optimize import brentq, newton


def sign(x): return math.copysign(1, x)


def compute_closest_intersections(df: pd.DataFrame):
    """
    Compute the closest intersection point from the data point for each column
    :type df: object
    """
    closest_intersection = {}
    sizes = list(df)[1:]
    for s in sizes:
        x = df[s].iloc[(df[s] - 5000).abs().argsort()[:1]]
        v = x.index.values[0]
        closest_intersection[s] = (v, df['p'][v])
    return closest_intersection


def compute_regressions(df: pd.DataFrame, col, max_degree, nb_samples):
    """
    Compute the logistic regression of a column for degree from 1 to max_degree
    """
    mid = nb_samples / 2
    x = df['p'].values
    y = df[col].values
    m = df.shape[0]
    res = {}
    for d in range(1, max_degree + 1):
        fit = np.polyfit(x, y, d)
        fit_fn = np.poly1d(fit)
        y_pred = fit_fn(x)
        mse = np.sum((y_pred - y) ** 2)
        rmse = np.sqrt(mse / m)

        fn = lambda x: fit_fn(x) - mid
        a = fn(0.0)
        b = fn(1.0)
        if sign(a) != sign(b):
            p = brentq(fn, 0.0, 1.0)
        else:
            p = newton(fn, 0.1, maxiter = 10000)
        res[d] = (fit, fit_fn, rmse, p)
    return res


def compute_best_p(df: pd.DataFrame, nb_samples, max_degree = 12):
    res = {}
    for col in list(df)[1:]:
        regs = compute_regressions(df, col, max_degree = max_degree, nb_samples = nb_samples)
        regs_sort = sorted(list(regs.items()), key = lambda x: x[1][2])
        best = regs_sort[0]
        res[col] = best
    return res


def print_best_p(df: pd.DataFrame, nb_samples, max_degree = 12):
    best_ps = compute_best_p(df, nb_samples, max_degree)
    for k, v in best_ps.items():
        print('Best p for %s vertices : %f, regression error : %f' % (k, v[1][3], v[1][2]))


def best_p(df: pd.DataFrame, col, nb_samples):
    regs = compute_regressions(df, col, 12, nb_samples)
