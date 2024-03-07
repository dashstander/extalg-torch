
import math
import torch



def get_all_minors(X, k: int):
    """ Given an m by n matrix X, compute all C(m, k) x C(n, k) minors of X.
    """
    m, n = X.shape
    device = X.device
    
    row_combinations = torch.combinations(torch.arange(m, device=device), r=k)
    col_combinations = torch.combinations(torch.arange(n,device=device), r=k)
    
    row_indices = row_combinations.view(-1)
    col_indices = col_combinations.view(-1)
    
    index_combinations = torch.cartesian_prod(row_indices, col_indices)
    
    minors = X[
        index_combinations[:, 0].view(-1, k)[:, :, None],
        index_combinations[:, 1].view(-1, k)[:, :, None]
    ]
    
    return minors.reshape(-1, k, k)


def compound_matrix_k(X, k: int):
    """ Given an m by n matrix X, compute the kth compound matrix of X, which is in the kth exterior power of X.
    Consider a linear map X: W -> V. W and V are an n and m dimensional vector spaces, respectively. The exterior algebra (Λk)W is the set of k-dimensional subspaces of W. (Λk)X is represents how X maps k-dimensional subspaces of W to k-dimensional subspaces of V. 
    """
    m, n = X.shape
    nrow = math.comb(m, k)
    ncol = math.comb(n, k)
    return torch.vmap(torch.linalg.det)(get_all_minors(X, k)).reshape(nrow, ncol)
