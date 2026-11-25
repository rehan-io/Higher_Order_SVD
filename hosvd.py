"""HOSVD implemented from scratch. Only numpy's matrix SVD is used."""
import numpy as np


def unfold(T, n):
    """Flatten tensor T into a matrix whose rows run along dimension n."""
    return np.moveaxis(T, n, 0).reshape(T.shape[n], -1)


def fold(M, n, shape):
    """Turn a matrix made by unfold() back into a tensor of the given shape."""
    other_dims = [s for k, s in enumerate(shape) if k != n]
    return np.moveaxis(M.reshape([shape[n]] + other_dims), 0, n)


def mode_n_product(T, M, n):
    """Multiply tensor T by matrix M along dimension n."""
    new_shape = list(T.shape)
    new_shape[n] = M.shape[0]
    return fold(M @ unfold(T, n), n, new_shape)


def hosvd(T):
    """Split tensor T into a core tensor S and one orthogonal matrix per dimension."""
    U = []
    for n in range(T.ndim):
        T_n = unfold(T, n)
        # full_matrices only when needed, so every U[n] comes out square
        U_n, _, _ = np.linalg.svd(T_n, full_matrices=T_n.shape[0] > T_n.shape[1])
        U.append(U_n)

    S = T
    for n in range(T.ndim):
        S = mode_n_product(S, U[n].T, n)
    return S, U


def reconstruct(S, U):
    """Rebuild the tensor from the core S and the matrices U."""
    T = S
    for n in range(S.ndim):
        T = mode_n_product(T, U[n], n)
    return T
