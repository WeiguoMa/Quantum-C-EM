"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
import torch as tc
from tensornetwork.visualization.graphviz import to_graphviz
from basic_gates import TensorGate
import re

def rename_edgeAxis(_node, _oqs: list[int]):
    """
    Rename the edge axis of a node after a double-qubit gate was applied
        Convert the index name of a tensor to 'physics' if it is like 'physics_{}'.format(int).
    Args:
        _node: the node after the gate was applied;
        _oqs: operating_qubits, the qubits that the gate was applied.
    Returns:
        _node: the node after the edge axis was renamed.
    """
    def remove_numFromPhysics(_name: list or [str]) -> list[str]:
        """
        :param _name: list of str
        :return: list of str
        """
        def has_s_int(string):
            match = re.search(r's_\d+', string)
            return bool(match)

        if isinstance(_name, list) is False:
            raise TypeError('Input must be a list.')

        for __i, __element in enumerate(_name):
            if has_s_int(__element) is True:
                _name[__i] = 'physics'
        return _name

    for _i in _oqs:
        _node[_i].axis_names = remove_numFromPhysics(_node[_i].axis_names)
        for _j in range(_node[_i].get_rank()):
            if _node[_i][_j].is_dangling() is True:
                _node[_i]['physics'].name = 'physics'
    return _node

def sort_edges4svd(_node: tn.Node, _op_idx: list):
    """
    Sort the bond and physics edges of a node after svd.

    Args:
        _node: the node to be sorted
        _op_idx: the index of the operator in the node
    Returns:
        _left: the left edges of the node after svd
        _right: the right edges of the node after svd
    """

    def _cluster_name(_op_index: list, *args):
        """
        Cluster the edges of a node according to the operator index.

        Args:
            _op_index: the index of the operator in the node
            *args: the edges of the node
        Returns:
            _left: the names of svd left side;
            _right: the names of svd right side.
        """
        def leftORight(_op_index_: list, _bond_idx: str):
            # Generally, left bond_name is smaller than right bond_name in integer form, like 'bond_01' < 'bond_23'
            # and 'bond_910' < 'bond_1011', 'bond_1011' < 'bond_1012'
            if int(_bond_idx.split('_')[1]) < int(str(_op_index[0]) + str(_op_index[1])):
                return 'left'
            else:
                return 'right'

        _left_, _right_ = [], []
        _bond_, _inner_, _physics_ = [], [], []
        for _i in range(len(args)):
            for _j in range(len(args[_i])):
                if args[_i][_j].startswith('bond'):
                    _bond_.append(args[_i][_j])
                elif args[_i][_j].startswith('inner'):
                    _inner_.append(args[_i][_j])
                elif args[_i][_j].startswith('physics'):
                    _physics_.append(args[_i][_j])

        _bond_.sort(), _inner_.sort(), _physics_.sort()
        if len(_bond_) == 0:
            _bond_ = [[], []]
        elif len(_bond_) == 1:
            if leftORight(_op_index, _bond_[0]) == 'left':
                _bond_ = [_bond_[0], []]
            else:
                _bond_ = [[], _bond_[0]]
        if _inner_:
            _left_ = [_bond_[0]] + [_physics_[0]] + [_inner_[0]]
            _right_ = [_inner_[1]] + [_physics_[1]] + [_bond_[1]]
        else:
            _left_ = [_bond_[0]] + [_physics_[0]]
            _right_ = [_physics_[1]] + [_bond_[1]]

        # Remove empty list which is nested in list
        _left_ = [_x for _x in _left_ if _x]
        _right_ = [_x for _x in _right_ if _x]
        return _left_, _right_

    _left, _right = _cluster_name(_op_idx, _node.axis_names)
    _left = [_node[name] for name in _left]
    _right = [_node[name] for name in _right]
    return _left, _right

def is_nested(lst: list) -> bool:
    """
    Check if a list is nested
    :param lst: list to be checked;
    :return: return True if the list is nested, otherwise return False
    """
    return any(isinstance(i, list) for i in lst)

def ket0():
    """
    :return: Return the state |0>
    """
    return tc.tensor([1. + 0.j, 0. + 0.j], dtype=tc.complex128)

def get_spilt(_node, _oqs: list[int]):
    """
    Split node while a double-qubit gate was applied

    Args:
        _node: the node after the gate was applied;
        _oqs: operating_qubits, the qubits that the gate was applied.
    Returns:
        _left: the left node after split;
        _right: the right node after split;
        _: the middle node after split.
    """
    _left_edges, _right_edges = sort_edges4svd(_node, _oqs)
    _left, _right, _ = tn.split_node(_node,
                                     left_edges=_left_edges,
                                     right_edges=_right_edges,
                                     left_name='qubit_' + str(_oqs[0]),
                                     right_name='qubit_' + str(_oqs[1]),
                                     edge_name='bond_{}{}'.format(_oqs[0], _oqs[1]))
    return _left, _right, _

def qr_left2right(_qubits: list):
    """
    QR decomposition from left to right.
    :param _qubits: List of qubits (as nodes);
    :return: Nodes after QR decomposition.
    """
    # left-most rank-2 node
    _q, _r = tn.split_node_qr(_qubits[0],
                              left_edges=[_qubits[0]['physics']],
                              right_edges=[_qubits[0]['bond_{}{}'.format(0, 1)]],
                              left_name=_qubits[0].name)
    _r = _r @ _qubits[1]
    _r.name = 'qubit_' + str(0 + 1)
    _qubits[0], _qubits[1] = _q, _r
    # left to right rank-3 node
    for ii in range(1, len(_qubits)- 1):
        _q, _r = tn.split_node_qr(_qubits[ii],
                                  left_edges=[_qubits[ii][0], _qubits[ii][1]],
                                  right_edges=[_qubits[ii][2]],
                                  left_name=_qubits[ii].name)
        _r = _r @ _qubits[ii+1]
        _r.name = 'qubit_' + str(ii+1)
        _qubits[ii], _qubits[ii+1] = _q, _r

def svd_right2left(_qubits, chi: int = None):
    """
    SVD decomposition from right to left.
    :param _qubits: List of qubits (as nodes);
    :param chi: Maximum bond dimension to be saved in SVD;
    :return: Nodes after SVD decomposition.
    """
    if chi is None:
        # A number who is big enough to keep all the information
        chi = 2 ** len(_qubits)
    # right-most rank-2 node
    idx = len(_qubits) - 1
    contracted_two_nodes = tn.contract_between(_qubits[idx - 1], _qubits[idx])
    _left, _right, _ = tn.split_node(contracted_two_nodes,
                                     left_edges=[contracted_two_nodes[0], contracted_two_nodes[1]],
                                     right_edges=[contracted_two_nodes[2]],
                                     left_name='qubit_' + str(idx - 1),
                                     right_name='qubit_' + str(idx), max_singular_values=chi)
    _qubits[idx - 1], _qubits[idx] = _left, _right
    # right to left rank-4 node
    for ii in range(len(_qubits) - 2, 1, -1):
        contracted_two_nodes = tn.contract_between(_qubits[ii - 1], _qubits[ii])
        _left, _right, _ = tn.split_node(contracted_two_nodes,
                                         left_edges=[contracted_two_nodes[0], contracted_two_nodes[1]],
                                         right_edges=[contracted_two_nodes[2], contracted_two_nodes[3]],
                                         left_name='qubit_' + str(ii - 1),
                                         right_name='qubit_' + str(ii), max_singular_values=chi)
        _qubits[ii - 1], _qubits[ii] = _left, _right
    # left-most rank-2 node
    contracted_two_nodes = tn.contract_between(_qubits[0], _qubits[1])
    _left, _right, _ = tn.split_node(contracted_two_nodes,
                                     left_edges=[contracted_two_nodes[0]],
                                     right_edges=[contracted_two_nodes[1], contracted_two_nodes[2]],
                                     left_name='qubit_' + str(0),
                                        right_name='qubit_' + str(1),
                                        max_singular_values=chi)
    _qubits[0], _qubits[1] = _left, _right

def create_init_mpo(number: int, phys_dimension=None, bond_dimension=None, conj_conn_dim=None) -> list:
    """
    create initial mpo
    :param number: the number of qubits;
    :param phys_dimension: the dimension of each site, which exactly is physical dimension;
    :param bond_dimension: the dimension of bond;
    :param conj_conn_dim: link dimension between two mpo;
    :return:
    """
    _mps = [
        tn.Node(ket0(), name='qubit_{}'.format(ii), axis_names=['physics']) for ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def add_gate(qubit_edges, gate, operating_qubits: list):
    """
    Add quantum Gate to tensor network
    :param qubit_edges: edges which represent qubits;
    :param gate: gate to be added;
    :param operating_qubits: operating qubits;
    :return: added gate tensor network.
    """
    single = gate.single
    if single is False:
        if is_nested(operating_qubits) is True:
            raise NotImplementedError('Couple of CNOT gates are not supported yet.')
        else:
            _gate = tn.Node(gate.tensor, name=gate.name)
            for i, bit in enumerate(operating_qubits):
                tn.connect(qubit_edges[bit], _gate[i])
                qubit_edges[bit] = _gate[i + len(operating_qubits)]
    else:
        _gate_list = [tn.Node(gate.tensor, name=gate.name) for _ in range(len(operating_qubits))]
        for i, bit in enumerate(operating_qubits):
            tn.connect(qubit_edges[bit], _gate_list[i][0])
            qubit_edges[bit] = _gate_list[i][1]

def add_gate_truncate(_qubits: list, _gate: TensorGate, operating_qubits: list):
    if isinstance(_qubits, list) is False:
        raise TypeError('Qubit must be a list.')
    if isinstance(_gate, TensorGate) is False:
        raise TypeError('Gate must be a TensorGate.')
    if isinstance(operating_qubits, list) is False:
        raise TypeError('Operating qubits must be a list.')
    if len(operating_qubits) > 2:
        raise NotImplementedError('Only two-qubit gates are supported currently.')

    single = _gate.single
    if single is False:
        if is_nested(operating_qubits) is True:
            raise TypeError('Series CNOT gates are not supported yet.')
        else:
            _edges = []
            _gate = tn.Node(_gate.tensor, name=_gate.name, axis_names=_gate.axis_name)
            for _i, _bit in enumerate(operating_qubits):
                _edges.append(tn.connect(_qubits[_bit]['physics'], _gate['inner_' + str(_i)]))
            # contract the connected edge and inherit the name of the pre-qubit
            for _j, _edge in enumerate(_edges):
                _gate = tn.contract(_edges[_j], name=_gate.name)
            # In tensornetwork package, axis_name is not equal to _name_of_edge_. While calculating, to ensure that
            # we are using right order of axis_names, we need to set axis_names of _gate according to its edges' name.
            axis_names = []
            for _edge in [_gate[_i] for _i in range(_gate.get_rank())]:
                axis_names.append(_edge.name)
            _gate.axis_names = axis_names
            #
            # # Codes above are fitted for multi-qubits gate, like q>=3, but the following codes are only for
            # two-qubits gate with the consideration of simple SVD truncation. # SVD truncation back to two qubits
            _qubits[operating_qubits[0]], _qubits[operating_qubits[1]], _ = get_spilt(_gate, operating_qubits)
            _qubits = rename_edgeAxis(_qubits, operating_qubits)
    else:
        _gate_list = [tn.Node(_gate.tensor, name=_gate.name) for _ in range(len(operating_qubits))]
        for _i, _bit in enumerate(operating_qubits):
            tn.connect(_qubits[_bit][0], _gate_list[_i][0])
            # contract the connected edge and inherit the name of the pre-qubit
            _qubits[_bit] = _qubits[_bit] @ _gate_list[_i]
            _qubits[_bit].name = 'qubit_' + str(_bit)
            _qubits[_bit].axis_names = ['physics']
            _qubits[_bit][0].name = 'physics'

def contract_mps(_qubits):
    """
    Contract all qubits to a single node.
    :param _qubits: List of qubits (nodes);
    :return: Merged node.
    """
    op = _qubits[0]
    for i in range(1, len(_qubits)):
        op = tn.contract_between(op, _qubits[i])
    return op

def plot_nodes(nodes):
    """
    Plot tensor network nodes.
    :param nodes: tensor network nodes;
    :return:
    """
    raise NotImplementedError('Plotting is not supported yet.')
    # for node in nodes:
    #     print(node)
    #     print(to_graphviz(node))