"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
from tensornetwork.visualization.graphviz import to_graphviz
from basic_gates import TensorGate
import re
import torch as tc


def leftORight(_op_index_: list, _bond_idx: str):
    if int(_bond_idx.split('_')[1]) < _op_index_[0]:
        return 'left'
    else:
        return 'right'

def rename_edgeAxis(_node, _oqs: list[int]):
    r"""
    Rename the edge axis of a node after a double-qubit gate was applied
        Convert the index name of a tensor to 'physics_{}'.format{qnumber} if it is like 'physics_{}'.format(int).
    Args:
        _node: the node after the gate was applied;
        _oqs: operating_qubits, the qubits that the gate was applied.
    Returns:
        _node: the node after the edge axis was renamed.
    """
    def reNum_FromPhysics(_name: list or [str], __oq) -> list[str]:
        r"""
        Rename the edge axis of a node after a double-qubit gate was applied

        Args:
            _name: the name of the edge axis;
            __oq: operating_qubit, the qubit that the gate was applied.

        Returns:
            _name: the name of the edge axis after rename.
        """
        def has_s_int(string):
            # detect form 's_num'
            match = re.search(r's_\d+', string)
            return bool(match)

        if isinstance(_name, list) is False:
            raise TypeError('Input must be a list.')

        for __i, __element in enumerate(_name):
            if has_s_int(__element) is True:
                _name[__i] = 'physics_{}'.format(__oq)
        return _name

    for _i in _oqs:
        _node[_i].axis_names = reNum_FromPhysics(_node[_i].axis_names, _i)
        for _j in range(_node[_i].get_rank()):
            if _node[_i][_j].is_dangling() is True:
                _node[_i]['physics_{}'.format(_i)].name = 'physics_{}'.format(_i)
    return _node

def cluster_name4svd(_op_index: list, *args):
    r"""
    Cluster the edges of a node according to the operator index.

    Args:
        _op_index: the index of the operator in the node
        *args: the edges of the node
    Returns:
        _left: the names of svd left side;
        _right: the names of svd right side.

    Additional information:
        !!! IMPORTANT
        This cluster method may cause error when the inner and physics edges are complex, we made new cluster method,
            which is shown in Function(sort_edges4dep), which records the edges-(axis)name can make it self-consistent.
    """
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

    # May occur error when the number of inner and physics edges for one node is larger than 10.
    if len(_inner_) > 9 or len(_physics_) > 9:
        raise NotImplementedError('Currently, the number of inner and physics edges for one node is limited to 10.')

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

def qr_cluster(_axis_names: list, _op_idx: list[int]):
    r"""
    Cluster the edges of a node according to the operator index in qr method.

    Args:
        _axis_names: the names of the edges of a node;
        _op_idx: the index of the operator in the node.
    Returns:
        _left: the names of svd left side.
    """
    if len(_op_idx) != 1:
        raise ValueError('QR decomposition from left to right working with each qubit')

    _left, _right = [], []
    _physics_, _I_, _bond_ = [], [], []
    for name in _axis_names:
        if name.startswith('physics'):
            _physics_.append(name)
        elif name.startswith('I'):
            _I_.append(name)
        elif name.startswith('bond') or name.startswith('qrbond'):
            _bond_.append(name)
        else:
            raise ValueError('Invalid edge name.')
    _physics_.sort(), _I_.sort(), _bond_.sort()

    if len(_bond_) == 1:
        if _op_idx[0] != int(_bond_[0].split('_')[1]):
            raise ValueError('Bond is not compatible with the operator.')
        _bond_ = [_bond_[0], []]
    elif len(_bond_) == 2:
        if _op_idx[0] != int(_bond_[0].split('_')[1]) and _op_idx[0] != int(_bond_[0].split('_')[-1]):
            raise ValueError('Bond is not compatible with the operator.')
        if _op_idx[0] != int(_bond_[1].split('_')[1]) and _op_idx[0] != int(_bond_[1].split('_')[-1]):
            raise ValueError('Bond is not compatible with the operator.')

    if leftORight(_op_idx, _bond_[0]) == 'left':
        _left = [_bond_[0]] + _physics_ + _I_
        _right = [_bond_[1]]
    else:
        _left = [_bond_[1]] + _physics_ + _I_
        _right = [_bond_[0]]

    _left = [_x for _x in _left if _x]
    _right = [_x for _x in _right if _x]
    return _left, _right

def sort_edges4dep(_node: tn.Node,
                   _left_AxisName: list[str] = None,
                   _right_AxisName: list[str] = None,
                   _op_idx: list = None,
                   _purpose: str = 'svd'):
    r"""
    Sort the bond and physics edges of a node after svd.

    Args:
        _node: the node to be sorted
        _left_AxisName: the left edges of the node before svd
        _right_AxisName: the right edges of the node before svd
        _op_idx: the index of the operator in the node
        _purpose: the purpose of the sorting, 'svd' or 'qr'

    Returns:
        _left: the left edges of the node after svd
        _right: the right edges of the node after svd

    Attention:
        Some parts of program stop calling the method.
    """
    if _left_AxisName is None and _right_AxisName is None and _op_idx is None:
        raise ValueError('There is no information about the edges of the node.')

    if _purpose == 'svd':
        if _op_idx:
            _left_AxisName, _right_AxisName = cluster_name4svd(_op_idx, _node.axis_names)
        # Repetition
        _left = [_node[name] for name in _left_AxisName]
        _right = [_node[name] for name in _right_AxisName]
        return _left, _right
    elif _purpose == 'qr':
        _left, _right = qr_cluster(_node.axis_names, _op_idx)
        # Repetition
        _left = [_node[name] for name in _left]
        _right = [_node[name] for name in _right]
        return _left, _right

def is_nested(_lst: list) -> bool:
    r"""
    Check if a list is nested

    Args:
        _lst: the list to be checked

    Returns:
        True if the list is nested, False otherwise
    """
    return any(isinstance(_i, list) for _i in _lst)

def get_spilt(_node, _oqs: list[int], _left_AxisName: list[str] = None, _right_AxisName: list[str] = None):
    r"""
    Split node while a double-qubit gate was applied

    Args:
        _node: the node after the gate was applied;
        _oqs: operating_qubits, the qubits that the gate was applied;
        _left_AxisName: the axis_names of the left node after split;
        _right_AxisName: the axis_names of the right node after split.
    Returns:
        _left: the left node after split;
        _right: the right node after split;
        _: the middle node after split.
    """
    if _left_AxisName is None or _right_AxisName is None:
        _left_edges, _right_edges = sort_edges4dep(_node, _op_idx=_oqs)
    else:
        _left_edges, _right_edges = sort_edges4dep(_node,
                                                   _left_AxisName=_left_AxisName,
                                                   _right_AxisName=_right_AxisName)
    _left, _right, _ = tn.split_node(_node,
                                     left_edges=_left_edges,
                                     right_edges=_right_edges,
                                     left_name='qubit_' + str(_oqs[0]),
                                     right_name='qubit_' + str(_oqs[1]),
                                     edge_name='bond_{}_{}'.format(_oqs[0], _oqs[1]))
    return _left, _right, _

def EdgeName2AxisName(_nodes: list[tn.Node] or list[tn.AbstractNode]):
    r"""
    ProcessFunction -->
        In tensornetwork package, axis_name is not equal to _name_of_edge_. While calculating, to ensure that
                we are using right order of axis_names, we need to set axis_names of _gate according to its edges' name.

    Args:
        _nodes: the node to be set axis_names.

    Returns:
        None, but the axis_names of _nodes will be set in memory.
    """
    if not isinstance(_nodes, list):
        if not isinstance(_nodes, tn.Node) and not isinstance(_nodes, tn.AbstractNode):
            raise ValueError('The input should be a list of nodes.')
        _nodes = [_nodes]

    for _node in _nodes:
        _axis_names = []
        for _edge in [_node[i] for i in range(_node.get_rank())]:
            # hardcode, which is relating to code design from weiguo
            if 'qr' in _edge.name:
                _edge.name = _edge.name.replace('qr', '')
            _axis_names.append(_edge.name)
        _node.axis_names = _axis_names

def ket0():
    r"""
    Return: Return the state |0>
    """
    return tc.tensor([1. + 0.j, 0. + 0.j], dtype=tc.complex128)

def ket1():
    r"""
    Return: Return the state |1>
    """
    return tc.tensor([0. + 0.j, 1. + 0.j], dtype=tc.complex128)

def ket_hadamard():
    r"""
    Return: Return the state |+>
    """
    return tc.tensor([1. / tc.sqrt(tc.tensor(2.)), 1. / tc.sqrt(tc.tensor(2.))], dtype=tc.complex128)

def create_ket0Series(_number: int) -> list:
    r"""
    create initial qubits

    Args:
        _number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |0> * _number
    """

    _mps = [
        tn.Node(ket0(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(_number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket1Series(_number: int) -> list:
    r"""
    create initial qubits

    Args:
        _number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |1> * _number
    """

    _mps = [
        tn.Node(ket1(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(_number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket_hadamardSeries(_number: int) -> list:
    r"""
    create initial qubits

    Args:
        _number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_hadamard(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(_number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def add_gate(qubit_edges, gate, operating_qubits: list):
    r"""
    Add quantum Gate to tensor network

    Args:
        qubit_edges: the edges of qubits;
        gate: the gate to be added;
        operating_qubits: the qubits that the gate was applied.

    Returns:
        qubit_edges: the edges of qubits after the gate was added.
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

def add_gate_truncate(_qubits: list, _gate: TensorGate, _oqs: list):
    r"""
    Add quantum Gate to tensornetwork

    Args:
        _qubits: qubits;
        _gate: gate to be added;
        _oqs: operating qubits.

    Returns:
        _qubits: tensornetwork after adding gate.
    """
    if isinstance(_qubits, list) is False:
        raise TypeError('Qubit must be a list.')
    if isinstance(_gate, TensorGate) is False:
        raise TypeError('Gate must be a TensorGate.')
    if isinstance(_oqs, list) is False:
        raise TypeError('Operating qubits must be a list.')

    single = _gate.single
    if single is False:
        if len(_oqs) > 2:
            raise NotImplementedError('Only two-qubit gates are supported currently.')
        if _oqs[0] == _oqs[1]:
            raise ValueError('Operating qubits must be different.')
        if is_nested(_oqs) is True:
            raise NotImplementedError('Series CNOT gates are not supported yet.')
        else:
            _edges = []
            _gate = tn.Node(_gate.tensor, name=_gate.name, axis_names=_gate.axis_name)

            _contract_qubits = tn.contract_between(_qubits[_oqs[0]], _qubits[_oqs[1]],
                                                   name='q_{}_{}'.format(_oqs[0], _oqs[1]),
                                                   allow_outer_product=True)
            EdgeName2AxisName([_contract_qubits])
            for _i, _bit in enumerate(_oqs):
                _edges.append(tn.connect(_contract_qubits['physics_{}'.format(_bit)], _gate['inner_' + str(_i)]))

            # contract the connected edge and inherit the name of the pre-qubit
            _gate = tn.contract_between(_contract_qubits, _gate, name=_gate.name)

            # ProcessFunction, for details, see the function definition.
            EdgeName2AxisName([_gate])

            # # Codes above are fitted for multi-qubits gate, like q>=3, but the following codes are only for
            # two-qubits gate with the consideration of simple SVD truncation. # SVD truncation back to two qubits
            _left_AxisName, _right_AxisName = _qubits[_oqs[0]].axis_names, _qubits[_oqs[1]].axis_names
            for _element in range(len(_left_AxisName)):
                if 'physics' in _left_AxisName[_element]:
                    _left_AxisName[_element] = 'physics_0'
            for _element in range(len(_right_AxisName)):
                if 'physics' in _right_AxisName[_element]:
                    _right_AxisName[_element] = 'physics_1'
            for _l_name in _left_AxisName:
                for _r_name in _right_AxisName:
                    if _l_name == _r_name:
                        _left_AxisName.remove(_l_name), _right_AxisName.remove(_r_name)

            _qubits[_oqs[0]], _qubits[_oqs[1]], _ = get_spilt(_gate, _oqs, _left_AxisName, _right_AxisName)
            _qubits = rename_edgeAxis(_qubits, _oqs)
            # /|\ May cost time for a whole check of qubits, but it's easy to generalize to q>=3.
    else:
        _gate_list = [tn.Node(_gate.tensor, name=_gate.name, axis_names=_gate.axis_name) for _ in range(len(_oqs))]
        for _i, _bit in enumerate(_oqs):
            tn.connect(_qubits[_bit]['physics_{}'.format(_bit)], _gate_list[_i]['inner'])
            # contract the connected edge and inherit the name of the pre-qubit
            _qubits[_bit] = tn.contract_between(_qubits[_bit], _gate_list[_i], name=_qubits[_bit].name)

            for __i in range(len(_qubits[_bit].axis_names)):
                if 'physics' in _qubits[_bit][__i].name:
                    _qubits[_bit][__i].name = 'physics_{}'.format(_bit)

            EdgeName2AxisName([_qubits[_bit]])


def contract_mps(_qubits):
    r"""
    Contract all qubits to a single node.

    Args:
        _qubits: qubits.
    Returns:
        op: the contracted node.
    """
    op = _qubits[0]
    for i in range(1, len(_qubits)):
        op = tn.contract_between(op, _qubits[i], allow_outer_product=True)
    return op

def plot_nodes(_nodes):
    r"""
    Plot tensor network nodes.

    Args:
        _nodes: nodes to be plotted.

    Returns:
        None
    """
    raise NotImplementedError('Plotting is not supported yet.')
    # for node in nodes:
    #     print(_node)
    #     print(to_graphviz(_node))