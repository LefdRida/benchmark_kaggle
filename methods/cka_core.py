import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment



def centering(K, device):
    n = K.shape[0]
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    return H @ K @ H




class LinearCKA:
    def __init__(self, device):
        self.device = device

    def calculate(self, X, Y):
        """
        X: (n, d)
        Y: (n, d)
        """

        # Linear kernels
        K = X @ X.T
        L = Y @ Y.T

        # Center
        Kc = centering(K, self.device)
        Lc = centering(L, self.device)

        # HSIC
        hsic = torch.sum(Kc * Lc)

        # Normalization
        var1 = torch.sqrt(torch.sum(Kc * Kc))
        var2 = torch.sqrt(torch.sum(Lc * Lc))

        return hsic / (var1 * var2 + 1e-8)



def rbf(X, sigma=None):
    GX = X @ X.T
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T

    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)

    KX *= -0.5 / (sigma * sigma)
    return torch.exp(KX)


class KernelCKA:
    def __init__(self, device):
        self.device = device

    def calculate(self, X, Y, sigma=None):

        K = rbf(X, sigma)
        L = rbf(Y, sigma)

        Kc = centering(K, self.device)
        Lc = centering(L, self.device)

        hsic = torch.sum(Kc * Lc)

        var1 = torch.sqrt(torch.sum(Kc * Kc))
        var2 = torch.sqrt(torch.sum(Lc * Lc))

        return hsic / (var1 * var2 + 1e-8)




def linear_local_CKA(source_base, target_base,
                     source_query, target_query,
                     device):

    # Move everything to device
    source_base = source_base.to(device)
    target_base = target_base.to(device)
    source_query = source_query.to(device)
    target_query = target_query.to(device)

    cka = LinearCKA(device)
    graph = []

    with torch.no_grad():

        for i in tqdm(range(source_query.shape[0])):

            source = source_query[i:i+1]
            row = []

            for j in range(target_query.shape[0]):

                target = target_query[j:j+1]

                Z = torch.cat((source_base, source), dim=0)
                H = torch.cat((target_base, target), dim=0)

                score = cka.calculate(Z, H)

                row.append(score.item())

            graph.append(row)

    return torch.tensor(graph, device=device, dtype=torch.float32)


def kernel_local_CKA(source_base, target_base,
                     source_query, target_query,
                     device):

    source_base = source_base.to(device)
    target_base = target_base.to(device)
    source_query = source_query.to(device)
    target_query = target_query.to(device)

    cka = KernelCKA(device)
    graph = []

    with torch.no_grad():

        for i in tqdm(range(source_query.shape[0])):

            source = source_query[i:i+1]
            row = []

            for j in range(target_query.shape[0]):

                target = target_query[j:j+1]

                Z = torch.cat((source_base, source), dim=0)
                H = torch.cat((target_base, target), dim=0)

                score = cka.calculate(Z, H)

                row.append(score.item())

            graph.append(row)

    return torch.tensor(graph, device=device, dtype=torch.float32)


def get_retrieval(graph):

    graph = graph.detach().cpu().numpy()

    top_1 = 0
    top_5 = 0
    top_10 = 0
    N = graph.shape[0]

    for i in range(N):

        row = graph[i]
        sorted_idx = np.argsort(-row)

        if i in sorted_idx[:1]:
            top_1 += 1

        if i in sorted_idx[:5]:
            top_5 += 1

        if i in sorted_idx[:10]:
            top_10 += 1

    return top_1/N, top_5/N, top_10/N



def linear_matching(graph):

    graph_np = graph.detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(graph_np, maximize=True)

    return np.mean(row_ind == col_ind)