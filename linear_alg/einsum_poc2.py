import numpy as np
import opt_einsum as oe
from opt_einsum.testing import build_views
from pprint import pprint
from collections import defaultdict


class HGraph:
    def __init__(self, einsum_str, shapes):
        self.einsum_str = einsum_str
        self.shapes = shapes

        self.size_dict = dict()
        self.vertices = {i for i in range(len(shapes))}
        self.hyper_edges = defaultdict(list)
        # for each hyper edge, we need to collect the vertex index, the index of the tensor

        output_indices = set(einsum_str.split("->")[1])
        input_indices = set(einsum_str.split("->")[0])
        indices_to_contract = input_indices - output_indices

        # example str
        # einsum_str = "ij,jk,kl->il"
        input_str_split = einsum_str.split("->")[0].split(",")
        for tensor_idx, tensor_str in enumerate(input_str_split):
            for t_i, t_i_str in enumerate(tensor_str):
                self.size_dict[t_i_str] = self.shapes[tensor_idx][t_i]
                if t_i_str in indices_to_contract:
                    self.hyper_edges[t_i_str].append((tensor_idx, t_i))

        pprint(self.hyper_edges)

    def contract_hyper_edge(self, hyper_edge_idx):
        tensors_involved = [
            tensor_idx for tensor_idx, _ in self.hyper_edges[hyper_edge_idx]
        ]
        print(tensors_involved)

        # fuse tensors

        # fuse einsum_str
        input_str_split = self.einsum_str.split("->")[0].split(",")

        indices_involved = set()
        for tensor_idx in tensors_involved:
            tensor_str = input_str_split[tensor_idx]
            for i in tensor_str:
                indices_involved.add(i)

        result_indices = indices_involved - {hyper_edge_idx}

        print(indices_involved)
        print(result_indices)
        # fuse shapes

    def print(self):
        print("HGraph:")
        for k, v in self.__dict__.items():
            print(f"{k}:\n\t{v}")


hg = HGraph("ij,jk,kl->il", [(2, 4), (4, 3), (3, 5)])
hg.print()
hg.contract_hyper_edge("j")
