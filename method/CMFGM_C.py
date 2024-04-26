import numpy as np
import pandas as pd
import os
from sim import sim_setup
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier
from time import time
from sim import utils
import igraph
import scipy.linalg as slin
import scipy.optimize as sopt


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
    v = np.zeros((P, K, T))
    for j in range(P):
        fdata_ij = FDataGrid(Y[:, sum(L[:j]):sum(L[:j + 1]), :].reshape((N * L[j], T)), np.arange(0, 1, 1 / T))
        fpca_grid = FPCA(K, centering=True)
        fpca_grid.fit(fdata_ij)
        v[j, :, :] = fpca_grid.components_.data_matrix.reshape((K, T))
        fpca_grid = FPCA(K, centering=True)
        a[:, sum(L[:j]):sum(L[:j + 1]), :] = fpca_grid.fit_transform(fdata_ij).reshape((N, L[j], K))
    for j in range(sum_L):
        for k in range(K):
            mean = np.mean(a[:, j, k])
            sd = np.std(a[:, j, k])
            c[:, j, k] = (a[:, j, k] - mean) / 1
    return a, c, v


def notears_multifunctional(X, L,
                            lambda1,
                            Cov,
                            max_iter=100,
                            rho_max=1e+16,
                            h_tol=1e-8,
                            c_threshold=0.3,
                            bnds=None):
    def _bounds():
        bnds = []
        for i in range(P * K):
            for j in range(P * K):
                bnd = (0, 0) if int(i / K) == int(j / K) else (None, None)
                bnds.append(bnd)
        for i in range(sum(L)):
            for j in range(sum(L)):
                bnd = (0, 0) if node[i] == node[j] else (None, None)
                bnds.append(bnd)
        return bnds
    def _adj(c):
        CK = (c[: P * K * P * K]).reshape(P * K, P * K)
        loc = P * K * P * K
        CL = (c[loc:loc + sum(L) * sum(L)]).reshape(sum(L), sum(L))
        return CK, CL
    def _loss(CK, CL):
        C = np.zeros((sum(L) * K, sum(L) * K))
        for i in range(P):
            for j in range(P):
                C_ij = np.kron(CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])], CK[i * K:(i + 1) * K, j * K:(j + 1) * K])
                C[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = C_ij
        M = X @ C
        R = X - M
        U = X.T @ R + C @ (sum(Cov).T + sum(Cov))
        I = np.identity(sum(L) * K)
        loss = 0.5 / X.shape[0] * (np.sum(R ** 2) + np.trace((I - C) @ sum(Cov) @ (I - C).T))
        G_CKloss = np.zeros((P * K, P * K))
        G_CLloss = np.zeros((sum(L), sum(L)))
        # G_CKloss = - 1.0 / X.shape[0] * X.T @ R
        for i in range(P * K):
            for j in range(P * K):
                ni = int(i / K)
                nj = int(j / K)
                G_CKloss[i, j] = - 1.0 / X.shape[0] * \
                                 (U[i % K + int(sum_L[ni]) * K:int(sum_L[ni + 1]) * K:K, j % K + int(sum_L[nj]) * K:int(sum_L[nj + 1]) * K:K]
                                  * CL[int(sum_L[ni]):int(sum_L[ni + 1]), int(sum_L[nj]): int(sum_L[nj + 1])]).sum()

        for i in range(sum(L)):
            for j in range(sum(L)):
                ni, nj = node[i], node[j]
                G_CLloss[i, j] = - 1.0 / X.shape[0] * (U[i * K:(i + 1) * K, j * K:(j + 1) * K]
                                                 * CK[ni * K:(ni + 1) * K, nj * K:(nj + 1) * K]).sum()
        return loss, G_CKloss, G_CLloss
    def _f(CK, CL):
        # f(C)_ij = (CK_i'j' ** 2 + CL_i'j' ** 2)
        fC = np.zeros((P, P))
        g_CKf = np.zeros((P, P, P * K, P * K))
        g_CLf = np.zeros((P, P, sum(L), sum(L)))
        sumCL = np.zeros((P, P))
        sumCK = np.zeros((P, P))
        for i in range(sum(L)):
            for j in range(sum(L)):
                sumCL[node[i], node[j]] += CL[i, j] ** 2
        for i in range(P * K):
            for j in range(P * K):
                sumCK[int(i / K), int(j / K)] += CK[i, j] ** 2
        for i in range(P):
            for j in range(P):
                fC[i, j] = sumCL[i, j] * sumCK[i, j]
        for i in range(P):
            for j in range(P):
                for _i in range(i * K, (i + 1) * K):
                    for _j in range(j * K, (j + 1) * K):
                        g_CKf[i, j, _i, _j] = 2 * sumCL[i, j] * CK[_i, _j]
        for i in range(P):
            for j in range(P):
                for _i in range(sum(L[:i]), sum(L[:i + 1])):
                    for _j in range(sum(L[:j]), sum(L[:j + 1])):
                        g_CLf[i, j, _i, _j] = 2 * sumCK[i, j] * CL[_i, _j]
        g_cKf = g_CKf.reshape(P * P, P * K * P * K)
        g_cLf = g_CLf.reshape(P * P, sum(L) * sum(L))
        fc = fC.reshape(P * P)
        return fc, g_cKf, g_cLf
    def _l1(CK, CL):
        # f(C)_ij = (CK_i'j' ** 2 + CL_i'j' ** 2)
        fC = np.zeros((P, P))
        g_CKf = np.zeros((P, P, P * K, P * K))
        g_CLf = np.zeros((P, P, sum(L), sum(L)))
        sumCL = np.zeros((P, P))
        sumCK = np.zeros((P, P))
        for i in range(sum(L)):
            for j in range(sum(L)):
                sumCL[node[i], node[j]] += CL[i, j] ** 2
        for i in range(P * K):
            for j in range(P * K):
                sumCK[int(i / K), int(j / K)] += CK[i, j] ** 2
        for i in range(P):
            for j in range(P):
                fC[i, j] = sumCL[i, j] * sumCK[i, j]
        for i in range(P):
            for j in range(P):
                for _i in range(i * K, (i + 1) * K):
                    for _j in range(j * K, (j + 1) * K):
                        g_CKf[i, j, _i, _j] = sumCL[i, j] * CK[_i, _j] * 2
        for i in range(P):
            for j in range(P):
                for _i in range(sum(L[:i]), sum(L[:i + 1])):
                    for _j in range(sum(L[:j]), sum(L[:j + 1])):
                        g_CLf[i, j, _i, _j] = sumCK[i, j] * CL[_i, _j] * 2
        g_cKf = g_CKf.reshape(P * P, P * K * P * K)
        g_cLf = g_CLf.reshape(P * P, sum(L) * sum(L))
        fc = fC.reshape(P * P)
        return sum(fc), sum(g_cKf), sum(g_cLf)
    def _h(CK, CL):
        fc, g_cKf, g_cLf = _f(CK, CL)
        fC = fc.reshape((P, P))
        E = slin.expm(fC)
        h = np.trace(E) - P
        G_CKh = (E.T.reshape(P * P) @ g_cKf).reshape(P * K, P * K)
        G_CKl = (E.T.reshape(P * P) @ g_cLf).reshape(sum(L), sum(L))
        return h, G_CKh, G_CKl, g_cKf, g_cLf
    def _func(c):
        CK, CL = _adj(c)
        loss, G_CKloss, G_CLloss = _loss(CK, CL)
        h, G_CKh, G_CLh, g_cKf, g_cLf = _h(CK, CL)
        l1, g_cKl1, g_cLl1 = _l1(CK, CL)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * l1
        G_CKsmooth = G_CKloss + (rho * h + alpha) * G_CKh + lambda1 * g_cKl1.reshape(P * K, P * K)
        G_CLsmooth = G_CLloss + (rho * h + alpha) * G_CLh + lambda1 * g_cLl1.reshape(sum(L), sum(L))
        g_obj = np.concatenate((G_CKsmooth, G_CLsmooth), axis=None)
        return obj, g_obj

    (N, _, K) = X.shape
    P = L.shape[0]
    X = X.reshape(N, sum(L) * K)
    # c = [c_k c_l]
    c_est, rho, alpha, h = np.zeros(P * K * P * K + sum(L) * sum(L)), 1.0, 0.0, np.inf
    c_est[P * K * P * K:P * K * P * K + sum(L) * sum(L)] = 1
    node = {}
    for p in range(P):
        for l in range(L[p]):
            node[l + sum(L[:p])] = p
    for l in range(sum(L)):
        for _l in range(sum(L)):
            c_est[P * K * P * K + l * sum(L) + _l] = 1 - (node[l] == node[_l])
    sum_L = np.zeros(P + 1)
    for p in range(P + 1):
        sum_L[p] = int(sum(L[:p]))
    if bnds is None:
        bnds = _bounds()
    for _ in range(max_iter):
        c_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, c_est, method='l-bfgs-b', jac=True, bounds=bnds)
            c_new = sol.x
            CK_new, CL_new = _adj(c_new)
            h_new, _, _, _, _ = _h(CK_new, CL_new)
            if h_new > 0.25 * h:
                rho *= 10
                c_est = c_new
            else:
                break
        fun_est, _ = _func(c_est)
        fun_new, _ = _func(c_new)
        print(fun_est, fun_new, h, h_new, rho, alpha)
        c_est, h = c_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    CL_est, CK_est = _adj(c_est)
    CL_est[np.abs(CL_est) < c_threshold] = 0
    CK_est[np.abs(CK_est) < c_threshold] = 0
    return CL_est, CK_est


def update_C(X, L, lambda1, Cov):
    t = 1
    CK, CL = notears_multifunctional(X, L, Cov=Cov, lambda1=lambda1, c_threshold=0, h_tol=1e-8, rho_max=1e+16)
    C = np.zeros((sum(L) * K, sum(L) * K))
    C_est = np.zeros((sum(L) * K, sum(L) * K))
    E_est = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            CL_ij = CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])].copy()  # a Li * Lj matrix
            CK_ij = CK[i * K:(i + 1) * K, j * K:(j + 1) * K].copy()
            C_est[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = np.kron(CL_ij, CK_ij)
            E_est[i, j] = np.sqrt(np.sum(CK_ij ** 2) * np.sum(CL_ij ** 2) / L[i] / L[j] / K)
            if E_est[i, j] < t * lambda1:
                CL_ij *= 0
            else:
                CL_ij *= (1 - t * lambda1 / E_est[i, j])
            C_est[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = np.kron(CL_ij, CK_ij)
            E_est[i, j] = np.sqrt(np.sum(CK_ij ** 2) * np.sum(CL_ij ** 2) / L[i] / L[j] / K)
    E_est[np.abs(E_est) < 0.3] = 0
    X = X.reshape((N, sum(L) * K))
    R = X - X @ C_est
    return C_est, np.sqrt(np.sum(R ** 2) / sum(L) / K / N), E_est, CK, CL


def update_X(Y, U, C, sigma, omega, E_est, K, L):
    [N, _, T] = Y.shape
    P = len(L)
    G = igraph.Graph.Adjacency(E_est)
    ordered_vertices = G.topological_sorting()
    X = np.zeros((N, sum(L) * K))
    A = np.zeros((N, sum(L) * K, sum(L) * K))
    B = np.zeros((N, sum(L) * K, sum(L) * T))
    mu = np.zeros((N, sum(L) * K))
    Sigma = np.zeros((N, sum(L) * K, sum(L) * K))
    for i in range(N):
        A_post = np.zeros((sum(L) * K, sum(L) * K))
        B_post = np.zeros((sum(L) * K, sum(L) * T))
        mu_post = np.zeros((sum(L) * K))
        for j in ordered_vertices:
            # update for prior
            pa_j = G.neighbors(j, mode=igraph.IN)

            Aj = np.zeros((L[j] * K, sum(L) * K))
            Bj = np.zeros((L[j] * K, sum(L) * T))
            muj = np.zeros(L[j] * K)
            muj_post = np.zeros(L[j] * K)

            for k in pa_j:
                muj += mu_post[sum(L[:k]) * K:sum(L[:k + 1]) * K] \
                        @ C[sum(L[:k]) * K:sum(L[:k + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1] * K)]
                Aj += C[sum(L[:k]) * K:sum(L[:k + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1] * K)].T \
                      @ A_post[sum(L[:k]) * K:sum(L[:k + 1] * K)]
                Bj += C[sum(L[:k]) * K:sum(L[:k + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1] * K)].T \
                      @ B_post[sum(L[:k]) * K:sum(L[:k + 1] * K)]
            Aj_epsilon = np.zeros((L[j] * K, sum(L) * K))
            Aj_epsilon[:, sum(L[:j]) * K:sum(L[:j + 1] * K)] = np.identity(L[j] * K)
            Aj += Aj_epsilon
            Sigmaj = sigma ** 2 * Aj @ Aj.T + omega ** 2 * Bj @ Bj.T

            # update for posterior
            Aj_post = np.zeros((L[j] * K, sum(L) * K))
            Bj_post = np.zeros((L[j] * K, sum(L) * T))
            for l in range(L[j]):
                Sigmajl = Sigmaj[l * K:(l + 1) * K, l * K:(l + 1) * K]
                Uj = U[j].T
                muj_post[l * K:(l + 1) * K] = muj[l * K:(l + 1) * K] + Sigmajl @ Uj.T \
                                              @ np.linalg.inv(Uj @ Sigmajl @ Uj.T + np.identity(T) * omega ** 2) \
                                              @ (Y[i, sum(L[:j]) + l] - Uj @ muj[l * K:(l + 1) * K])
                Aj_post[l * K:(l + 1) * K] = Aj[l * K:(l + 1) * K] - Sigmajl @ Uj.T \
                                             @ np.linalg.inv(Uj @ Sigmajl @ Uj.T + np.identity(T) * omega ** 2) \
                                             @ Uj @ Aj[l * K:(l + 1) * K]
                Bj_epsilon = np.zeros((T, sum(L) * T))
                Bj_epsilon[:, (sum(L[:j]) + l) * T:(sum(L[:j]) + l + 1) * T] = np.identity(T)
                Bj_post[l * K:(l + 1) * K] = Bj[l * K:(l + 1) * K] - Sigmajl @ Uj.T \
                                             @ np.linalg.inv(Uj @ Sigmajl @ Uj.T + np.identity(T) * omega ** 2) \
                                             @ (Uj @ Bj[l * K:(l + 1) * K] + Bj_epsilon)
            mu_post[sum(L[:j]) * K:sum(L[:j + 1] * K)] = muj_post
            A_post[sum(L[:j]) * K:sum(L[:j + 1] * K)] = Aj_post
            B_post[sum(L[:j]) * K:sum(L[:j + 1] * K)] = Bj_post
        A[i] = A_post
        B[i] = B_post
        mu[i] = mu_post
        # calculate covariance
        Sigma[i] = sigma ** 2 * A[i] @ A[i].T + omega ** 2 * B[i] @ B[i].T

    # kalman smoothing
    mu_smooth = np.zeros((N, sum(L) * K))
    Sigma_smooth = np.zeros((N, sum(L) * K, sum(L) * K))
    Ls = []
    for order_j, j in enumerate(list(reversed(ordered_vertices))):
        Ls.append(L[j])

    for i in range(N):
        mu_pri = np.zeros((sum(L) * K))
        mu_post = np.zeros((sum(L) * K))
        cov_pri = np.zeros((sum(L) * K, sum(L) * K))
        A_pri = np.zeros((sum(L) * K, sum(L) * K))
        B_pri = np.zeros((sum(L) * K, sum(L) * T))
        A_post = np.zeros((sum(L) * K, sum(L) * K))
        B_post = np.zeros((sum(L) * K, sum(L) * T))
        for order_j, j in enumerate(list(reversed(ordered_vertices))):
            mu_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] = mu[i, sum(L[:j]) * K:sum(L[:j + 1]) * K]
            A_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] = A[i, sum(L[:j]) * K:sum(L[:j + 1]) * K]
            B_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] = B[i, sum(L[:j]) * K:sum(L[:j + 1]) * K]
            for order_k, k in enumerate(list(reversed(ordered_vertices))):
                cov_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K, sum(Ls[:order_k]) * K:sum(Ls[:order_k + 1]) * K] \
                    = Sigma[i, sum(L[:j]) * K:sum(L[:j + 1]) * K, sum(L[:k]) * K:sum(L[:k + 1]) * K]

        for order_j, j in enumerate(list(reversed(ordered_vertices))):
            mu_post[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                = mu_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                  + cov_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K, :sum(Ls[:order_j]) * K] \
                  @ np.linalg.inv(cov_pri[:sum(Ls[:order_j]) * K, :sum(Ls[:order_j]) * K]) \
                  @ (mu_post[:sum(Ls[:order_j]) * K] - mu_pri[:sum(Ls[:order_j]) * K])
            A_post[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                = A_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                  - cov_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K, :sum(Ls[:order_j]) * K] \
                  @ np.linalg.inv(cov_pri[:sum(Ls[:order_j]) * K, :sum(Ls[:order_j]) * K]) \
                  @ (A_pri[:sum(Ls[:order_j]) * K] - A_post[:sum(Ls[:order_j]) * K])
            B_post[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                = B_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K] \
                  - cov_pri[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K, :sum(Ls[:order_j]) * K] \
                  @ np.linalg.inv(cov_pri[:sum(Ls[:order_j]) * K, :sum(Ls[:order_j]) * K]) \
                  @ (B_pri[:sum(Ls[:order_j]) * K] - B_post[:sum(Ls[:order_j]) * K])

        Sigma_post = sigma ** 2 * A_post @ A_post.T + omega ** 2 * B_post @ B_post.T
        for order_j, j in enumerate(list(reversed(ordered_vertices))):
            mu_smooth[i, sum(L[:j]) * K:sum(L[:j + 1]) * K] = mu_post[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K]
            for order_k, k in enumerate(list(reversed(ordered_vertices))):
                Sigma_smooth[i, sum(L[:j]) * K:sum(L[:j + 1]) * K, sum(L[:k]) * K:sum(L[:k + 1]) * K] \
                    = Sigma_post[sum(Ls[:order_j]) * K:sum(Ls[:order_j + 1]) * K, sum(Ls[:order_k]) * K:sum(Ls[:order_k + 1]) * K]
    return mu.reshape((N, sum(L), K)), Sigma


def update_U(Y, X, L, init_U=None):
    [N, _, K] = X.shape
    [N, _, T] = Y.shape
    P = len(L)
    U = np.zeros((P, K, T))
    loss = 0
    for p in range(P):
        svd_Y = Y[:, sum(L[:p]):sum(L[:p + 1]), :].reshape((N * L[p], T))
        svd_X = X[:, sum(L[:p]):sum(L[:p + 1]), :].reshape((N * L[p], K))
        V, S, W = np.linalg.svd(svd_Y.T @ svd_X, full_matrices=False)
        Psi = (V @ W).T * np.sqrt(T)
        U[p, :, :] = Psi
        R = Y[:, sum(L[:p]):sum(L[:p + 1]), :] - np.einsum('ijk,kl->ijl', X[:, sum(L[:p]):sum(L[:p + 1]), :], Psi)
        loss = loss + np.sum(R ** 2)
    return U, np.sqrt(loss / N / sum(L) / T)


def CMFGC(Y, L, K, max_iter=3, tol=1e-2, init_U=None, init_X=None, init_C=None, E_true=None, W_true=None):
    [N, _, T] = Y.shape
    P = len(L)
    sum_L = sum(L)
    U = np.zeros((P, K, T))
    X = np.zeros((N, sum_L, K))
    C = np.zeros((sum_L * K, sum_L * K))
    if init_U is not None:
        U = init_U
    if init_X is not None:
        X = init_X
    if init_C is not None:
        C = init_C
    E_est = np.zeros((P, P))
    omega, sigma = 1, 1
    for _ in range(max_iter):
        C_old = C.copy()
        X, Cov = update_X(Y, U, C, sigma, omega, E_est, K, L)
        C, sigma, E_est, CK, CL = update_C(X, L, lambda1=0.1, Cov=Cov)
        print(np.sum((C - C_old) ** 2))
        if np.sum((C - C_old) ** 2) <= tol:
            break
        print("sigma:%f" % sigma)
        U, omega = update_U(Y, X, L, init_U)

        print("omega:%f" % omega)
    # rotation
    def transform(basis_pc, basis_true):
        [K, T] = basis_pc.shape
        B = FDataGrid(basis_pc, np.arange(0, 1, 1 / T)).to_basis(basis_true).coefficients[:, :K]
        return B
    C_rotate = np.zeros((sum(L) * K, sum(L) * K))
    for i in range(P):
        for j in range(P):
            CL_ij = CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])].copy()  # a Li * Lj matrix
            CK_ij = CK[i * K:(i + 1) * K, j * K:(j + 1) * K].copy()
            B_i = transform(U[i, :, :], Fourier((0, 1), n_basis=K, period=1))
            B_j = transform(U[j, :, :], Fourier((0, 1), n_basis=K, period=1))
            CK_ij = B_i.T @ CK_ij @ B_j
            C_rotate[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = np.kron(CL_ij, CK_ij)
    return U, X, C, E_est, C_rotate


def est_function(a_est, N, L, K, T, v, node):
    a_est = a_est.reshape((N, sum(L), K))
    Y_est = np.zeros((N, sum(L), T))
    for j in range(sum(L)):
        Y_est[:, j, :] = a_est[:, j, :] @ v[node[j], :, :]
    return Y_est


def single_test(N, T, K, P, L, sigma=None, s0=None, h=None, E_true=None, W_true=None):
    if h is None:
        E_true, W_true, g, h, true_a = sim_setup.generate_DAG(N=N, T=T, K=K, P=P, L=L, s0=14, sigma=sigma, seed=233)
    C_true = np.zeros((sum(L) * K, sum(L) * K))
    CK = np.zeros((P * K, P * K))
    CL = np.zeros((sum(L), sum(L)))
    for i in range(P):
        for j in range(P):
            CK[i * K:(i + 1) * K, j * K:(j + 1) * K] = np.identity(K)
            CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])] = np.full((L[i], L[j]), W_true[i, j])
            C_true[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] \
                = np.kron(CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])], CK[i * K:(i + 1) * K, j * K:(j + 1) * K])
    Y = sim_setup.matrix_form(h)
    init_X, _, init_U = net_fpca(Y, L, K)
    U, X, C_est, E_est, C_rotate = CMFGC(Y, L, K=K, init_X=init_X, init_U=init_U, E_true=E_true, W_true=W_true)
    acc = utils.count_accuracy(E_true, E_est != 0)
    node = {}
    for p in range(P):
        for l in range(L[p]):
            node[l + sum(L[:p])] = p
    fourier_basis = Fourier((0, 1), n_basis=K, period=1)
    F_true = fourier_basis(np.arange(0, 1, 1 / T))
    U_true = np.zeros((P, K, T))
    for i in range(P):
        for k in range(K):
            U_true[i, k, :] = F_true[k, :, 0]
    Y_est = est_function(X.reshape(N, sum(L) * K) @ C_est, N, L, K, T, U, node)
    Y_true = est_function(true_a.reshape(N, sum(L) * K) @ C_true, N, L, K, T, U_true, node)
    mse = np.sum((Y_est - Y) ** 2) / N / T / sum(L)
    mse_true = np.sum((Y_true - Y) ** 2) / N / T / sum(L)
    acc['mse'] = mse
    acc['mse_true'] = mse_true
    acc['delta_C'] = np.sum((C_rotate - C_true) ** 2)
    print(C_est)
    print(acc)
    return acc


if __name__ == '__main__':
    N = 20
    T = 100
    P = 10
    K = 2
    L = np.full(P, 2)
    acc = single_test(N, T, K, P, L, 0.1)
    acc_pd = pd.DataFrame.from_dict(acc, orient='index').T
    acc_pd.to_csv('save.csv')
    # acc = single_test(N, T, K, P, L, 0.1)
    # acc_pd = pd.DataFrame.from_dict(acc, orient='index').T
    # acc_pd.to_csv('20_0.csv')
