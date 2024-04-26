import numpy as np
import pandas as pd
import os
from optimize import Adam
import matplotlib.pyplot as plt
from sim import sim_setup
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier
from time import time
import scipy.linalg as slin
import scipy.optimize as sopt
from sim import utils
from multiprocessing import Pool
from multiprocessing import freeze_support


def net_fpca(Y, L, K=5):
    """
    functional pca to change the functional data to K principle scores
    :param Y: the N * sum(L) * T 3d-array observation data
    :param K: the number of functional principle component
    :return:
    a: the N * sum(L) * K 3d-array represents principle scores
    c: the
    v: the sum(L) * K * T 2d-array represents 1th-Kth functional principle components
    """
    P = len(L)
    (N, sum_L, T) = Y.shape
    a = np.zeros((N, sum_L, K))
    c = np.zeros((N, sum_L, K))
    v = np.zeros((sum_L, K, T))
    for j in range(sum_L):
        fdata_ij = FDataGrid(Y[:, j, :].reshape((N, T)), np.arange(0, 1, 1 / T))
        fpca_grid = FPCA(K, centering=True)
        fpca_grid.fit(fdata_ij)
        v[j, :, :] = fpca_grid.components_.data_matrix.reshape((K, T))
        fpca_grid = FPCA(K, centering=True)
        a[:, j, :] = fpca_grid.fit_transform(fdata_ij).reshape((N, K))
    for j in range(sum_L):
        for k in range(K):
            mean = np.mean(a[:, j, k])
            sd = np.std(a[:, j, k])
            c[:, j, k] = (a[:, j, k] - mean) / 1
    return a, c, v


def notears_separate(X, lambda1, node, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    def _bounds():
        bounds = []
        for _ in range(2):
            for i in range(d):
                for j in range(d):
                    if node[i] == node[j]:
                        bounds.append((0, 0))
                    else:
                        bounds.append((0, None))
        return bounds
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)

    bnds = _bounds()
    # if loss_type == 'l2':
    #     X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        w_loss, _ = _loss(_adj(w_new))
        print(w_loss)
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def est_function(a_est, N, L, K, T, v):
    a_est = a_est.reshape((N, sum(L), K))
    Y_est = np.zeros((N, sum(L), T))
    for j in range(sum(L)):
        Y_est[:, j, :] = a_est[:, j, :] @ v[j, :, :]
    return Y_est



def single_test(N, T, K, P, L, sigma=None, s0=None, h=None, E_true=None, W_true=None):
    if h is None:
        E_true, W_true, g, h, true_a = sim_setup.generate_DAG(N=N, T=T, K=K, P=P, L=L, s0=s0, sigma=sigma)
    Y = sim_setup.matrix_form(h)
    a, c, v = net_fpca(Y, L, K=K)
    node = {}
    for p in range(P):
        for l in range(L[p] * K):
            node[l + sum(L[:p]) * K] = p
    E_est = notears_separate(X=true_a.reshape(N, sum(L) * K), lambda1=0.1, node=node)
    np.savetxt('NS-N30.csv', E_est)
    return 0
    X = a.reshape(N, sum(L) * K)
    B_est = np.zeros((P, P))
    for i in range(K * sum(L)):
        for j in range(K * sum(L)):
            if E_est[i, j] > 0:
                B_est[node[i], node[j]] += 1
    for i in range(P):
        B_est[i, i] = 0
        for j in range(i):
            if B_est[i, j] == B_est[j, i] == 0:
                continue
            if B_est[i, j] > 0 and B_est[j, i] > 0:
                B_est[i, j] = B_est[j, i] = -1
            if B_est[i, j] > B_est[j, i]:
                B_est[i, j] = 1
                B_est[j, i] = 0
            else:
                B_est[i, j] = 0
                B_est[j, i] = 1
    # for i in range(K * sum(L)):
    #     for j in range(K * sum(L)):
    #         E_est[i, j] *= B_est[node[i], node[j]]
    acc = utils.count_accuracy(E_true, B_est)
    Y_est = est_function(a.reshape(N, sum(L) * K) @ E_est, N, L, K, T, v)
    mse = np.sum((Y_est - Y) ** 2) / N / T / sum(L)
    acc['mse'] = mse
    return acc


if __name__ == '__main__':
    P = 10
    L = np.full(P, 2)
    single_test(N=30, T=100, K=2, P=P, s0=14, L=L, sigma=0.1)