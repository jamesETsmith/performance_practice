import numpy as np
import opt_einsum as oe
from opt_einsum.testing import build_views
from pprint import pprint


def make_size_dict(shapes, einsum_str):
    size_dict = dict()
    einsum_str_stripped = einsum_str.replace(",", "").replace("-", "").replace(">", "")
    final_idx = set(einsum_str.split("->")[1])

    idx_to_contract = set(einsum_str_stripped) - final_idx
    print(idx_to_contract)

    t_i_pairs_to_contract = []

    str_idx = 0
    for ti, shape in enumerate(shapes):
        for i in range(len(shape)):
            size_dict[einsum_str_stripped[str_idx]] = shape[i]
            str_idx += 1

    return size_dict


tensor_shape = [(10, 30), (30, 5), (5, 60)]
einsum_str = "ij,jk,kl->il"

size_dict = make_size_dict(tensor_shape, einsum_str)
pprint(size_dict)
views = build_views(einsum_str, size_dict)

path, path_info = oe.contract_path(einsum_str, *views)
print(path)
print(path_info)


unique_shapes = [10, 30, 5, 60]

cache = {}


def min_cost(i: int, j: int, unique_shapes: list[int]):
    """Calculates the minimum cost of contracting a range of tensors.

    Parameters
    ----------
    i : int
        The index of the first tensor to contract
    j : int
        The index of the last tensor to contract
    unique_shapes : list[int]
        The shapes of the tensors to contract (N+1) where N is the number of tensors

    Returns
    -------
    int
        The minimum cost of contracting the range of tensors
    """
    # Base case when we're "contracting" a single tensor
    if i == j:
        return 0

    if (i, j) in cache:
        return cache[(i, j)]

    res = np.inf

    for k in range(i, j):
        cost = (
            # Const of contacting A_i...A_k
            min_cost(i, k, unique_shapes)
            # Cost of contracting A_k+1...A_j
            + min_cost(k + 1, j, unique_shapes)
            # Cost of contracting A_i...A_k and A_k+1...A_j
            + unique_shapes[i - 1] * unique_shapes[k] * unique_shapes[j] * 2
        )
        res = min(res, cost)

    cache[(i, j)] = res

    return res


print(min_cost(1, len(unique_shapes) - 1, unique_shapes))

cache = {}
unique_shapes = [40, 20, 30, 10, 30]
print(min_cost(1, len(unique_shapes) - 1, unique_shapes))

shapes_to_contract = np.array(unique_shapes[1:-1])
idx_sorted = np.argsort(shapes_to_contract)[::-1] + 1

# print(shapes_to_contract[idx_sorted])


def cost_of_contraction(idx_sorted, shapes):
    print(shapes)
    cost = 0
    for i in idx_sorted:
        print(i - 1, i, i + 1)
        cost += shapes[i - 1] * shapes[i] * shapes[i + 1] * 2

    return cost


print(cost_of_contraction(idx_sorted, unique_shapes))
