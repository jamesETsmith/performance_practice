import numpy as np
import opt_einsum as oe
from opt_einsum.testing import build_views
from pprint import pprint
from itertools import permutations


def make_size_dict(shapes, einsum_str):
    size_dict = dict()
    einsum_str_split = einsum_str.split("->")[0].split(",")

    for i, shape in enumerate(shapes):
        for j, size in enumerate(shape):
            size_dict[einsum_str_split[i][j]] = size
    return size_dict


einsum_str = "ij,jk,kl->il"
shapes = [(2, 4), (4, 3), (3, 5)]
size_dict = make_size_dict(shapes, einsum_str)
pprint(size_dict)

views = build_views(einsum_str, size_dict)
pprint(views)

path, path_info = oe.contract_path(einsum_str, *views)
print(path)
print(path_info)


# if we contract the first two tensors, what happens?
einsum_str_split = einsum_str.split("->")[0].split(",")
# transform str

# get tensors involved
# idx_to_contract = "j"
# tensors_involved = [
#     i for i, str_idx in enumerate(einsum_str_split) if idx_to_contract in str_idx
# ]
# print(tensors_involved)

# new_str_idx = "".join([einsum_str_split[i] for i in tensors_involved]).replace(
#     idx_to_contract, ""
# )
# print(new_str_idx)

# new_shapes = [tuple([size_dict[i] for i in new_str_idx])] + [
#     s for i, s in enumerate(shapes) if i not in tensors_involved
# ]
# print(new_shapes)


def naive_flopcount(einsum_str, shapes):
    size_dict = make_size_dict(shapes, einsum_str)

    lhs_indices, rhs_indices = einsum_str.replace(",", "").split("->")
    print(lhs_indices, rhs_indices)
    indices_to_contract = set(lhs_indices) - set(rhs_indices)

    print(indices_to_contract)

    n_terms_to_contract = len(shapes)
    n_mul = n_terms_to_contract - 1
    n_add = 1
    flops_per_inner_loop = n_mul + n_add
    print(flops_per_inner_loop)

    n_loop_iter = np.prod(
        [
            size_dict[i]
            for i in set(einsum_str.replace(",", "").replace("->", "").replace("-", ""))
        ]
    )
    print(n_loop_iter)

    return n_loop_iter * flops_per_inner_loop


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
    idx_contraction,
    inner: bool,
    num_terms: int,
    size_dictionary,
) -> int:
    """Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
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


print(shapes)
print(naive_flopcount(einsum_str, shapes))
print(flop_count("kj", True, 3, size_dict) * shapes[0][0] * shapes[-1][-1])


lhs_indices, rhs_indices = einsum_str.replace(",", "").split("->")
indices_to_contract = set(lhs_indices) - set(rhs_indices)
print(indices_to_contract)


contraction_orderings = list(permutations(indices_to_contract))
print(contraction_orderings)
