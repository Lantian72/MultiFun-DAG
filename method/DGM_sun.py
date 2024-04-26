import numpy as np
from sim.sim_setup import generate_DAG
from sim.sim_setup import matrix_form
from sim import utils
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
import scipy.optimize as sopt
import igraph


def net_fpca(Y, L, K):
    P = len(L)
    (N, sum_L, T) = Y.shape
    fdata = FDataGrid(Y.reshape((N * sum_L, T)), np.arange(0, 1, 1 / T))
    a = np.zeros((N, sum_L, K))
    fpca_grid = FPCA(K, centering=True)
    fpca_grid.fit(fdata)
    v = fpca_grid.components_.data_matrix.reshape((K, T))
    fpca_grid = FPCA(K, centering=True)
    a = fpca_grid.fit_transform(fdata).reshape((N, sum_L, K))
    return a, v


def learning_DGM(Y, a, v, L, cpa, lambda1, max_iter=100, tol=1e-8):
    [N, sum_L, K] = a.shape
    T = Y.shape[2]
    P = len(L)
    node = {}
    B = np.zeros((sum_L, sum_L, K))
    for p in range(P):
        for l in range(L[p] * K):
            node[l + sum(L[:p]) * K] = p
    # for k in range(sum_L):
    #     L_old = np.inf
    #     L_new = np.inf
    #     b_k = np.zeros((sum_L, K))
    #     for _ in range(max_iter):
    #         for j in range(sum_L):
    #             if node[j] not in cpa[node[k]]:
    #                 continue
    #             r_kj = Y[:, k, :].copy()
    #             b_k[j] = np.zeros(K)
    #             b_kold = np.zeros(K)
    #             for l in range(sum_L):
    #                 if l != j:
    #                     for i in range(N):
    #                         r_kj[i] -= Y[i, l, :] * (b_k[l] @ v)
    #
    #             t, s = 0, 1
    #
    #             def _func(b):
    #                 y = r_kj.copy()
    #                 for i in range(N):
    #                     y[i] = y[i] - Y[i, j, :] * (b @ v)
    #                 return np.sum(y ** 2)
    #
    #             print(k, j, _func(b_k[j]))
    #             while t < 1000:
    #                 grad = np.zeros(K)
    #                 for i in range(N):
    #                     grad += (Y[i, j, :] * (r_kj[i] - Y[i, j, :] * (b_kold @ v))) @ v.T
    #                 grad += lambda1 * np.sign(b_k[j])
    #                 grad = grad / N / 100
    #                 b_k[j] = b_kold + grad * s
    #                 while _func(b_k[j]) > _func(b_kold):
    #                     s = s * 0.8
    #                     b_k[j] = b_kold + grad * s
    #                     if s < 1e-15:
    #                         break
    #                 t += 1
    #                 if _func(b_kold) < _func(b_k[j]) + tol:
    #                     break
    #                 b_kold = b_k[j].copy()
    #                 if s < 1e-15:
    #                     break
    #             print(k, j, _func(b_k[j]), s)
    #         r_k = Y[:, k, :].copy()
    #         for l in range(sum_L):
    #             for i in range(N):
    #                 r_k[i] -= Y[i, l, :] * (b_k[l] @ v)
    #         L_new = np.sum(r_k ** 2)
    #         if np.abs(L_new - L_old) < tol * N * 100:
    #             break
    #         L_old = L_new
    #     B[k, :, :] = b_k
    # sum_Y = np.zeros((sum_L, K))
    # scov_Y = np.zeros((sum_L, sum_L, K))
    # for l in range(sum_L):
    #     for t in range(T):
    #         sum_Y[t] = np.sum(Y[:, l, t])
    #     for j in range(sum_L):
    #         for t in range(T):
    #             scov_Y[l, j, t] = np.sum(Y[:, j, t] * Y[:, l, t])
    for k in range(sum_L):
        b_k = np.zeros(sum_L * K)
        def _func(b):
            b = b.reshape((sum_L, K)).copy()
            g_b = np.zeros((sum_L, K))
            r_k = Y[:, k, :].copy()
            for j in range(sum_L):
                if node[j] not in cpa[node[k]]:
                    continue
                r_kj = Y[:, k, :].copy()
                # for l in range(sum_L):
                #     if l != j:
                #         for i in range(N):
                #             r_kj[i] -= Y[i, l, :] * (b[l] @ v)
                # for i in range(N):
                #     r_k[i] -= Y[i, j, :] * (b[j] @ v)
                #     g_b[j] += (Y[i, j, :] * (r_kj[i] - Y[i, j, :] * (b[j] @ v))) @ v.T

                for l in range(sum_L):
                    if l != j:
                        r_kj -= np.einsum('ij,j->ij', Y[:, l, :], (b[l] @ v).reshape(T))
                m_k = np.einsum('ij,j->ij', Y[:, j, :], (b[j] @ v).reshape(T))
                r_k -= m_k
                g_b[j] += ((Y[:, j, :] * (r_kj - m_k)) @ v.T).sum(axis=0)

                g_b[j] += lambda1 * np.sign(b[j])
                g_b[j] = g_b[j] / N / Y.shape[2]
            # print(np.sum(r_k ** 2) / N / Y.shape[2])
            return np.sum(r_k ** 2) / N / Y.shape[2], -g_b.reshape(sum_L * K)
        sol = sopt.minimize(_func, b_k, method='l-bfgs-b', jac=True)
        b_k = sol.x
        print(k, sol.fun)
        B[:, k, :] = b_k.reshape((sum_L, K))
    return B


def est_function(B_est, N, P, L, K, T, v, Y):
    Y_est = np.zeros((N, sum(L), T))
    for i in range(N):
        for j in range(sum(L)):
            for k in range(sum(L)):
                Y_est[i, j, :] += Y[i, k, :] * (B_est[k, j, :] @ v)
    return Y_est


def single_test(N, T, K, P, L, sigma=None, s0=None, h=None, E_true=None, W_true=None):
    if h is None:
        E_true, W_true, g, h, true_a = generate_DAG(N=N, T=T, K=K, P=P, L=L, s0=s0, sigma=sigma)
    Y = matrix_form(h)
    a, v = net_fpca(Y, L, K)
    G = igraph.Graph.Adjacency(E_true)
    ordered_vertices = G.topological_sorting()
    cpa = {}
    for p in range(len(ordered_vertices)):
        cpa[ordered_vertices[p]] = []
        for _p in range(p):
            cpa[ordered_vertices[p]].append(ordered_vertices[_p])
    B_est = learning_DGM(Y, a, v, L, cpa, lambda1=0.01)
    # E_est[abs(E_est) < 0.3] = 0
    # E_est[abs(E_est) > 0.3] = 1
    node = {}
    for p in range(P):
        for l in range(L[p]):
            node[l + sum(L[:p])] = p
    E_est = np.zeros((P, P))
    for i in range(sum(L)):
        for j in range(sum(L)):
            for k in range(K):
                E_est[node[i], node[j]] += abs(B_est[i, j, k]) / L[node[i]] / L[node[j]] / K
    E_est[E_est < 0.3] = 0
    E_est[E_est > 0] = 1
    for i in range(sum(L)):
        for j in range(sum(L)):
            for k in range(K):
                B_est[i, j, k] *= E_est[node[i], node[j]]
    acc = utils.count_accuracy(E_true, E_est)
    Y_est = est_function(B_est, N, P, L, K, T, v, Y)
    # Y_est = Y @ B_est
    mse = np.sum((Y_est - Y) ** 2) / N / T / sum(L)
    acc['mse'] = mse
    return acc


if __name__ == '__main__':
    P = 10
    L = np.full(P, 1)
    single_test(N=1000, T=100, K=2, P=P, s0=20, L=L, sigma=0.1)
