import random

import numpy as np
from skfda.representation.basis import Fourier
from sim import utils
import igraph


def generate_DAG(N=100, P=20, T=100, K=5, s0=20, L=None, sigma=1, seed=233, rho=None, scale=None, sig_left=1, sig_right=1):
    """
    generate a multiple functional DAGs simulation data
    :param N: the number of observations
    :param P: the number of nodes
    :param T: the number of timesteps
    :param K: the number of basis(function)
    :param L: a 1*p array, L_i is the number of functions in i-th node
    :return:
    E_true: a adjacency matrix represents the true DAGs
    W_true: the weights of edges in E_true
    g: a list, where g[i] is a dict where g[i][p] is a L[p] * T array represents the true multiple functions of i-th node
    h: a dict, where h[i] is a dict where h[i][p] is a L[p] * T array represents the multiple functional data of i-th node
    """
    if scale is None:
        scale = np.full(P, 1)
    if rho is None:
        rho = np.zeros(P)
    utils.set_random_seed(seed)
    graph_type = 'ER'
    E_true = utils.simulate_dag(P, s0, graph_type)
    W_true = utils.simulate_parameter(E_true)
    fourier_basis = Fourier((0, 1), n_basis=K, period=1)
    s = fourier_basis(np.arange(0, 1, 1 / T))
    G = igraph.Graph.Adjacency(E_true)
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == P
    g = []
    h = []
    true_a = np.zeros((N, sum(L), K))
    for i in range(N):
        delta_i = {}
        for p in ordered_vertices:
            parents = G.neighbors(p, mode=igraph.IN)
            delta_i[p] = np.zeros((L[p], K))
            # cov_pl = np.random.multivariate_normal(mean=np.zeros(L[p]),
            #                                        cov=scale[p] * (np.identity(L[p]) * (1 - rho[p]) + np.full((L[p], L[p]), rho[p])))
            # rK = np.random.multivariate_normal(mean=np.zeros(K), cov=np.identity(K))
            for l in range(L[p]):
                mean_pl = np.zeros(K)
                for _p in parents:
                    for _l in range(L[_p]):
                        mean_pl = mean_pl + (delta_i[_p][_l, :] * W_true[_p, p]).reshape(K)
                # delta_i[p][l, :] = mean_pl + cov_pl[l] * np.full(K, 1)
                delta_i[p][l, :] = np.random.multivariate_normal(mean=mean_pl, cov=np.identity(K) * random.uniform(sig_left, sig_right))
                true_a[i, sum(L[:p]) + l, :] = delta_i[p][l, :]
        g_i = {}
        h_i = {}
        for p in range(P):
            g_i[p] = np.zeros((L[p], T))
            h_i[p] = np.zeros((L[p], T))
            for l in range(L[p]):
                for k in range(K):
                    g_i[p][l, :] += (delta_i[p][l, k] * s[k, :]).reshape((T,))
                for t in range(T):
                    h_i[p][l, t] = np.random.normal(loc=g_i[p][l, t], scale=sigma)
        g.append(g_i)
        h.append(h_i)
    return E_true, W_true, g, h, true_a


def matrix_form(h):
    """
    :param h: a list, h[i] is a dict which h[i][p] is a L[i] * T array represents the i-th observation data
    in p-th nodes.
    :return: a N * sum(L[i]) * T 3d-array
    """
    # get the shape of h (N, P, L[i], T)
    N = len(h)
    P = len(h[1])
    L = np.zeros(P, dtype=np.int)
    T = 0
    for p in range(P):
        (L[p], T) = h[0][p].shape
    Y = np.zeros((N, sum(L), T))
    for i in range(N):
        for p in range(P):
            loc = sum(L[:p])
            for l in range(L[p]):
                Y[i, loc + l, :] = h[i][p][l, :]
    return Y
