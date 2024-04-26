import numpy as np
from sim.sim_setup import generate_DAG
from sim.sim_setup import matrix_form
from sim import utils
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
import igraph


def net_fpca(Y, L, K):
    P = len(L)
    (N, sum_L, T) = Y.shape
    a = np.zeros((N, P, K))
    v = {}
    for j in range(P):
        v[j] = np.zeros((K, T * L[j]))
        fdata_ij = FDataGrid(Y[:, sum(L[:j]):sum(L[:j + 1]), :].reshape((N, T * L[j])), np.arange(0, 1, 1 / T / L[j]))
        fpca_grid = FPCA(K, centering=True)
        fpca_grid.fit(fdata_ij)
        v[j] = fpca_grid.components_.data_matrix.reshape((K, T * L[j]))
        fpca_grid = FPCA(K, centering=True)
        a[:, j, :] = fpca_grid.fit_transform(fdata_ij)
    return a, v


def learning_DGM(X, cpa, lambda1=0.0, gamma=0.0, max_iter=100, tol=1e-8):
    [N, P, K] = X.shape
    B = np.zeros((P, K, P, K))
    for k in range(P):
        L_new, L_est = 2e+9, 1e+9
        b_k = np.zeros((P, K, K))
        for i in range(max_iter):
            for j in cpa[k]:
                r_kj = X[:, k, :]
                for l in cpa[k]:
                    if l != j:
                        r_kj = r_kj - X[:, l, :] @ b_k[l]
                b_kold = np.zeros((K, K))
                # b_k[j] = np.zeros((K, K))
                eta = np.zeros((K, K))
                s, t = 1, 1
                # if i == 1:
                #     print(np.sum(r_kj ** 2))
                while t < 1000:
                    z = eta + s * (X[:, j, :].T @ (r_kj - X[:, j, :] @ eta) - lambda1 * eta)
                    b_k[j] = max(0, 1 - s * gamma * K / np.linalg.norm(z, ord=2)) * z
                    while np.sum((r_kj - X[:, j, :] @ b_k[j]) ** 2) + 0.5 * lambda1 * np.sum(b_k[j] ** 2) + gamma * K * np.sqrt(np.sum(b_k[j] ** 2))\
                            >= np.sum((r_kj - X[:, j, :] @ b_kold) ** 2) + 0.5 * lambda1 * np.sum(b_kold ** 2) + gamma * K * np.sqrt(np.sum(b_kold ** 2)):
                        s = s * 0.8
                        z = eta + s * (X[:, j, :].T @ (r_kj - X[:, j, :] @ eta) - lambda1 * eta)
                        b_k[j] = max(0, 1 - s * gamma * K / np.linalg.norm(z, ord=2)) * z
                        if s < 1e-15:
                            break
                    if np.sum(abs(b_k[j] - b_kold)) < tol:
                        break
                    eta = b_k[j] # + t / (t + 3) * (b_k[j] - b_kold)
                    b_kold = b_k[j].copy()
                    t += 1
                # print(k, j, np.sum((r_kj - X[:, j, :] @ b_k[j]) ** 2))
            x_k = X[:, k, :]
            for j in cpa[k]:
                x_k = x_k - X[:, j, :] @ b_k[j]
            L_new = 0.5 * np.sum(x_k ** 2) + 0.5 * lambda1 * np.sum(b_k ** 2) + gamma * K * np.sqrt(np.sum(b_k ** 2))
            # print(L_new, L_est)
            # if L_new > 100000:
            #     print(L_est, L_new)
            if abs(L_new - L_est) < tol:
                break
            L_est = L_new
        B[:, :, k, :] = b_k.copy()
        print(L_new)
    E_est = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            E_est[i, j] = np.sqrt(np.sum(B[i, :, j, :] ** 2)) / np.sqrt(K)
    return B, E_est


def est_function(a_est, N, P, L, K, T, v):
    a_est = a_est.reshape((N, P, K))
    Y_est = np.zeros((N, sum(L), T))
    for j in range(P):
        Y_compress = a_est[:, j, :] @ v[j]
        for l in range(L[j]):
            Y_est[:, sum(L[:j]):sum(L[:j + 1]), :] = Y_compress.reshape((N, L[j], T))
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
    B_est, E_est = learning_DGM(a, cpa, lambda1=0.001 * N, gamma=0.001 * N)
    E_est[abs(E_est) < 0.3] = 0
    E_est[abs(E_est) > 0.3] = 1
    for i in range(P):
        for k in range(K):
            for j in range(P):
                for _k in range(K):
                    B_est[i, k, j, _k] *= E_est[i, j]
    acc = utils.count_accuracy(E_true, E_est)
    Y_est = est_function(a.reshape(N, P * K) @ B_est.reshape(P * K, P * K), N, P, L, K, T, v)
    mse = np.sum((Y_est - Y) ** 2) / N / T / sum(L)
    acc['mse'] = mse
    return acc


if __name__ == '__main__':
    P = 10
    L = np.full(P, 2)
    single_test(N=1000, T=100, K=2, P=P, s0=20, L=L, sigma=0.1)