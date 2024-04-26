import numpy as np
import pandas as pd
import os
from optimize import Adam
from sim import sim_setup
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier
from time import time
import scipy.linalg as slin
import scipy.optimize as sopt
from sim import utils

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
        U = X.T @ R
        loss = 0.5 / X.shape[0] * np.sum(R ** 2)
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
        for i in range(sum(L)):
            for j in range(sum(L)):
                if node[i] == node[j] or (node[i] == 0 and node[j] == 1):
                    obj += 100000 * CL[i, j] ** 2
                    G_CLsmooth[i, j] += 200000 * CL[i, j]
        for i in range(P * K):
            for j in range(P * K):
                if int(i / K) == int(j / K):
                    obj += 100000 * CK[i, j] ** 2
                    G_CKsmooth[i, j] += 200000 * CK[i, j]
        g_obj = np.concatenate((G_CKsmooth, G_CLsmooth), axis=None)
        return obj, g_obj

    def _train(optimizer):
        for _ in range(1000):
            _, _grad = _func(optimizer.params)
            optimizer.update(_grad)
        return optimizer.params
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
            # optimizer = Adam(c_est)
            # _train(optimizer)
            # c_new = optimizer.params
            # #optimizer = Adam(c_est)
            # c_new = train(optimizer)
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


def transform(basis_pc, basis_true):
    [K, T] = basis_pc.shape
    B = FDataGrid(basis_pc, np.arange(0, 1, 1 / T)).to_basis(basis_true).coefficients[:, :K]
    return B


def est_function(a_est, N, L, K, T, v, node):
    a_est = a_est.reshape((N, sum(L), K))
    Y_est = np.zeros((N, sum(L), T))
    for j in range(sum(L)):
        Y_est[:, j, :] = a_est[:, j, :] @ v[node[j], :, :]
    return Y_est


def single_test(N, T, K, P, L, sigma=None, s0=None, h=None, E_true=None, W_true=None):
    # rho = np.full(P, 0)
    # L[0] = 5
    # L[1] = 1
    # scale = [1, 0.5]
    if h is None:
        E_true, W_true, g, h, true_a = sim_setup.generate_DAG(N=N, T=T, K=K, P=P, L=L, s0=14, sigma=sigma, seed=233)
    CK_true = np.zeros((P * K, P * K))
    CL_true = np.zeros((sum(L), sum(L)))
    C_true = np.zeros((sum(L) * K, sum(L) * K))
    for i in range(P):
        for j in range(P):
            CK_true[i * K:(i + 1) * K, j * K:(j + 1) * K] = np.identity(K)
            CL_true[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])] = np.full((L[i], L[j]), W_true[i, j])
            C_true[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] \
                = np.kron(CL_true[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])], CK_true[i * K:(i + 1) * K, j * K:(j + 1) * K])
    Y = sim_setup.matrix_form(h)
    a, c, v = net_fpca(Y, L, K=K)
    CK, CL = notears_multifunctional(a, L, lambda1=0.1, c_threshold=0, h_tol=1e-8, rho_max=1e+16)
    C = np.zeros((sum(L) * K, sum(L) * K))
    C_est = np.zeros((sum(L) * K, sum(L) * K))
    E_est = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            CL_ij = CL[sum(L[:i]):sum(L[:i + 1]), sum(L[:j]):sum(L[:j + 1])].copy()  # a Li * Lj matrix
            CK_ij = CK[i * K:(i + 1) * K, j * K:(j + 1) * K].copy()
            C_est[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = np.kron(CL_ij, CK_ij)
            B_i = transform(v[i, :, :], Fourier((0, 1), n_basis=K, period=1))
            B_j = transform(v[j, :, :], Fourier((0, 1), n_basis=K, period=1))
            CK_ij = B_i.T @ CK_ij @ B_j
            C[sum(L[:i]) * K:sum(L[:i + 1]) * K, sum(L[:j]) * K:sum(L[:j + 1]) * K] = np.kron(CL_ij, CK_ij)
            E_est[i, j] = np.sqrt(np.sum(CK_ij ** 2) * np.sum(CL_ij ** 2) / L[i] / L[j] / K)
    np.savetxt('MFGM-C30.csv', C)
    return 0
    E_est[np.abs(E_est) < 0.3] = 0
    node = {}
    for p in range(P):
        for l in range(L[p]):
            node[l + sum(L[:p])] = p
    for i in range(sum(L) * K):
        for j in range(sum(L) * K):
            C_est[i, j] *= (E_est[node[int(i / K)], node[int(j / K)]] > 0)
    acc = utils.count_accuracy(E_true, E_est != 0)
    Y_est = est_function(a.reshape(N, sum(L) * K) @ C_est, N, L, K, T, v, node)
    mse = np.sum((Y_est - Y) ** 2) / N / T / sum(L)
    # print(np.sqrt(np.sum((C - C_true) ** 2)))
    # print(acc, mse)
    # X = true_a.reshape((N, sum(L) * K))
    # R = X - X @ C
    # Phi = np.cov(R.T)
    # Sigma = np.cov(X.T)
    # acc = pd.DataFrame.from_dict(acc, orient='index')
    # savepath = '../output/sigma_' + str(sigma)
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # savefile = savepath + '/' + str(test_id) + '.csv'
    # acc.to_csv(savefile)
    acc['mse'] = mse
    print(acc)
    return acc


if __name__ == '__main__':
    P = 10
    L = np.full(P, 2)
    single_test(N=30, T=100, K=2, P=P, s0=14, L=L, sigma=0.1)
