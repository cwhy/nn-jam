from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Optional, FrozenSet, Any, Dict, Set

from variable_protocols.protocols import Variable, VariableGroup, struct_hash, VariableTensor


# noinspection PyTypeChecker
# because pycharm sucks
class HashedTree:
    def __init__(self, var: Variable,
                 ignore_names: bool = True,
                 is_root: bool = True) -> None:
        self.hash = struct_hash(var, ignore_names)
        self.is_root = is_root
        self.id: Optional[int] = None

        if var.type == 'VariableTensor':
            assert isinstance(var, VariableTensor)
            if var.var.type == 'BaseVariable':
                self.children: List[HashedTree] = []
                self.nodes: List[HashedTree] = [self]
            else:
                assert isinstance(var.var, VariableGroup)
                subtree = HashedTree(var.var)
                self.children = [subtree]
                self.nodes = self.children + subtree.nodes

        elif var.type == 'VariableGroup':
            assert isinstance(var, VariableGroup)
            self.children = list(sorted(
                (HashedTree(var, ignore_names, is_root=False) for var in var.vars),
                key=lambda x: x.hash
            ))
            self.nodes = sum([c.nodes for c in self.children], [])
        else:
            raise Exception(f"Unexpected Variable type {var.type}")

        if is_root:
            self.node_idx: Optional[Dict[HashedTree, int]] = {node: i for i, node in enumerate(self.nodes)}
            self.hash_dict: Optional[Dict[str, Set[HashedTree]]] = defaultdict(set)
            for n in self.nodes:
                self.hash_dict[n.hash].add(n)
        else:
            self.hash_dict = None
            self.node_idx = None


def check_tr(t1: HashedTree, t2: HashedTree) -> bool:
    return t1.hash == t2.hash


DiffNode = HashedTree


def diff(t1: HashedTree, t2: HashedTree) -> Tuple[Dict[str, int], Dict[str, int]]:
    if t1.hash_dict is None or t2.hash_dict is None:
        raise ValueError("Either tree entered for compare is not root")
    else:
        d1, d2 = t1.hash_dict, t2.hash_dict
        common_keys = set(d1).intersection(set(d2))
        t1_special = {k: len(d1[k]) for k in set(d1) - common_keys}
        t2_special = {k: len(d2[k]) for k in set(d2) - common_keys}
        for k in common_keys:
            if len(d1[k]) > len(d2[k]):
                t1_special[k] = len(d1[k]) - len(d2[k])
            elif len(d1[k]) < len(d2[k]):
                t2_special[k] = len(d2[k]) - len(d1[k])
        return t1_special, t2_special


DiffResultTr = List[Tuple[FrozenSet[DiffNode], FrozenSet[DiffNode]]]


def diff_tr(t1: HashedTree, t2: HashedTree) -> DiffResultTr:
    assert t1.is_root
    assert t2.is_root
    return diff_helper(t1, t2)


def diff_helper(t1: HashedTree, t2: HashedTree) -> DiffResultTr:
    if t1.hash == t2.hash:
        return []
    elif t1.children is None:
        if t2.children is not None:
            return [(frozenset([t1]), frozenset([t2]))]
        return []
    elif t2.children is None:
        return [(frozenset([t1]), frozenset([t2]))]
    else:
        tc1 = set(t1.children)
        tc2 = set(t2.children)
        tc1_hd = {n.hash: n for n in t1.children}
        tc2_hd = {n.hash: n for n in t2.children}
        g1, g2 = set(), set()
        for n1 in t1.children:
            if n1.hash in tc2_hd:
                print(n1.hash, tc2_hd[n1.hash].hash)
                print(n1, tc2_hd[n1.hash])
                tc2.remove(tc2_hd[n1.hash])
                tc1.remove(n1)
            else:
                g1.add(n1)
        for n2 in t2.children:
            if n2.hash in tc1_hd:
                tc1.remove(tc1_hd[n2.hash])
                tc2.remove(n2)
            else:
                g2.add(n2)
        if len(tc1) == 0 and len(tc2) == 0:
            raise Exception("This should not happen")
        return [(frozenset(tc1), frozenset(tc2))]
