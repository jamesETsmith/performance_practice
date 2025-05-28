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


def find_contractions(shapes, einsum_str):
    size_dict = dict()
    einsum_str_stripped = einsum_str.replace(",", "").replace("-", "").replace(">", "")
    final_idx = set(einsum_str.split("->")[1])
    print(f"output indices: {final_idx}")
    t_str_idx = einsum_str.split("->")[0].split(",")
    ten_idx_to_str = [
        {ci: c for ci, c in enumerate(t_str_idx[i])} for i in range(len(t_str_idx))
    ]
    print(f"tensor indices: {ten_idx_to_str}")

    idx_to_contract = set(einsum_str_stripped) - final_idx
    print(f"indices to contract: {idx_to_contract}")

    contractions = {k: [] for k in idx_to_contract}

    str_idx = 0
    for ti, shape in enumerate(shapes):
        for i in range(len(shape)):
            # Keep track of the size of each index
            size_dict[einsum_str_stripped[str_idx]] = shape[i]

            # If the index is in the set of indices to contract, add the tensor and index pair to the list of contractions for that index
            if einsum_str_stripped[str_idx] in idx_to_contract:
                contractions[einsum_str_stripped[str_idx]].append((ti, i))

            str_idx += 1

    print(f"contractions: {contractions}")

    return size_dict, contractions, ten_idx_to_str, t_str_idx


def compute_size_by_dict(indices, idx_dict) -> int:
    """Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes

    Returns:
    -------
    ret : int
        The resulting product.

    Examples:
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:  # lgtm [py/iteration-string-and-sequence]
        ret *= idx_dict[i]
    return ret


def flop_count(
    idx_contraction: str,
    inner: bool,
    num_terms: int,
    size_dictionary: dict[str, int],
) -> int:
    """Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction (e.g. 'ijk' in ij,jk->ik)
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns:
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples:
    --------
    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


tensor_shape = [(2, 4), (4, 3), (3, 5)]
einsum_str = "ij,jk,kl->il"

# tensor_shape = [(2, 4), (4, 3)]
# einsum_str = "ij,jk->ik"

# tensor_shape = [(4, 3), (3, 5)]
# einsum_str = "ij,jk->ik"

size_dict = make_size_dict(tensor_shape, einsum_str)
pprint(size_dict)
views = build_views(einsum_str, size_dict)

path, path_info = oe.contract_path(einsum_str, *views)
print(path)
print(path_info)


size_dict, contractions, ten_idx_to_str, t_str_idx = find_contractions(
    tensor_shape, einsum_str
)

for k, v in contractions.items():
    print(f"{k}: {v}")
    idx_involved = set()
    for t, i in v:
        idx_involved.update(set(t_str_idx[t]))
    print(f"Inidces involved in contraction: {idx_involved}")
    print(f"Size of contraction: {flop_count(idx_involved, True,2, size_dict)}")
