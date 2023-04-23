"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy
import tensornetwork as tn
from tensornetwork.visualization.graphviz import to_graphviz
from basic_gates import TensorGate
import torch as tc


def is_nested(_lst: list) -> bool:
    r"""
    Check if a list is nested

    Args:
        _lst: the list to be checked

    Returns:
        True if the list is nested, False otherwise
    """
    return any(isinstance(_i, list) for _i in _lst)

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

def create_ket0Series(number: int) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |0> * _number
    """

    _mps = [
        tn.Node(ket0(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket1Series(number: int) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |1> * _number
    """

    _mps = [
        tn.Node(ket1(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def create_ket_hadamardSeries(number: int) -> list:
    r"""
    create initial qubits

    Args:
        number: the number of qubits.

    Returns:
        _mps: the initial mps with the state |+> * _number
    """

    _mps = [
        tn.Node(ket_hadamard(), name='qubit_{}'.format(_ii),
                axis_names=['physics_{}'.format(_ii)]) for _ii in range(number)
    ]
    # Initial nodes has no edges need to be connected, which exactly cannot be saying as a MPO.
    return _mps

def add_gate(_qubits: list, gate: TensorGate, _oqs: list):
    r"""
    Add quantum Gate to tensornetwork

    Args:
        _qubits: qubits;
        gate: gate to be added;
        _oqs: operating qubits.

    Returns:
        _qubits: tensornetwork after adding gate.
    """
    if isinstance(_qubits, list) is False:
        raise TypeError('Qubit must be a list.')
    if isinstance(gate, TensorGate) is False:
        raise TypeError('Gate must be a TensorGate.')
    if isinstance(_oqs, list) is False:
        raise TypeError('Operating qubits must be a list.')

    single = gate.single
    if single is False:     # Two-qubit gate
        if len(_oqs) > 2:
            raise NotImplementedError('Only two-qubit gates are supported currently.')
        if _oqs[0] == _oqs[1]:
            raise ValueError('Operating qubits must be different.')
        if is_nested(_oqs) is True:
            raise NotImplementedError('Series CNOT gates are not supported yet.')
        else:
            _edges = []
            gate = tn.Node(gate.tensor, name=gate.name,
                            axis_names=[f'inner_{_oqs[0]}', f'inner_{_oqs[1]}', f'physics_{_oqs[0]}', f'physics_{_oqs[1]}'])
            # Created a new node in memory
            _contract_qubits = tn.contract_between(_qubits[_oqs[0]], _qubits[_oqs[1]],
                                                   name='q_{}_{}'.format(_oqs[0], _oqs[1]),
                                                   allow_outer_product=True)
            EdgeName2AxisName([_contract_qubits])
            for _i, _bit in enumerate(_oqs):
                _edges.append(tn.connect(_contract_qubits['physics_{}'.format(_bit)], gate[f'inner_{_bit}']))

            # contract the connected edge and inherit the name of the pre-qubit
            gate = tn.contract_between(_contract_qubits, gate, name=gate.name)

            # ProcessFunction, for details, see the function definition.
            EdgeName2AxisName([gate])

            # Split back to two qubits
            _left_AxisName, _right_AxisName = _qubits[_oqs[0]].axis_names, _qubits[_oqs[1]].axis_names
            _left_edges, _right_edges = [gate[name] for name in _left_AxisName],\
                                        [gate[name] for name in _right_AxisName]

            _qubits[_oqs[0]], _qubits[_oqs[1]], _ = tn.split_node(gate,
                                                                  left_edges=_left_edges,
                                                                  right_edges=_right_edges,
                                                                  left_name=f'qubit_{_oqs[0]}',
                                                                  right_name=f'qubit_{_oqs[1]}',
                                                                  edge_name=f'bond_{_oqs[0]}_{_oqs[1]}')
    else:
        gate_list = [tn.Node(gate.tensor, name=gate.name, axis_names=[f'inner_{_idx}', f'physics_{_idx}'])
                                                                                                for _idx in _oqs]
        for _i, _bit in enumerate(_oqs):
            tn.connect(_qubits[_bit][f'physics_{_bit}'], gate_list[_i][f'inner_{_bit}'])
            # contract the connected edge and inherit the name of the pre-qubit
            _qubits[_bit] = tn.contract_between(_qubits[_bit], gate_list[_i], name=_qubits[_bit].name)
            # ProcessFunction, for details, see the function definition.
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

def calculate_DM(_qubits, noisy: bool = False):
    r"""
    Calculate the density matrix of the state.

    Args:
        _qubits: qubits;
        noisy: whether the density matrix is noisy.

    Returns:
        _dm: the density matrix node;
        _dm_tensor: the density matrix tensor.
    """
    def _re_permute(_axis_names_: list[str]):
        _left_, _right_ = [], []
        for _idx_, _name_ in enumerate(_axis_names_):
            if 'con_' not in _name_:
                _left_.append(_idx_)
            else:
                _right_.append(_idx_)
        return _left_ + _right_

    _qubits_conj = copy.deepcopy(_qubits)
    _allowed_outer_product = True

    # Differential name the conjugate qubits' edges name to permute the order of the indices
    for _i, _qubit_conj in enumerate(_qubits_conj):
        _qubit_conj.name = 'con_' + _qubit_conj.name
        for _ii in range(len(_qubit_conj.edges)):
            if 'physics' in _qubit_conj[_ii].name:
                _qubit_conj[_ii].name = 'con_' + _qubit_conj[_ii].name
                _qubit_conj.axis_names[_ii] = 'con_' + _qubit_conj.axis_names[_ii]

    _contract_nodes = []
    for i in range(len(_qubits)):
        if noisy is True:
            tn.connect(_qubits[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])
            _allowed_outer_product = False      # Edges between ket-bra are now connected, outer product is not allowed.
        _contract_nodes.append(tn.contract_between(_qubits[i], _qubits_conj[i], name=f'contracted_qubit_{i}',
                                                   allow_outer_product=_allowed_outer_product))

    _dm = _contract_nodes[0]
    for _ii in range(1, len(_contract_nodes)):
        _dm = tn.contract_between(_dm, _contract_nodes[_ii], allow_outer_product=True)
    EdgeName2AxisName([_dm])

    _dm.tensor = tc.permute(_dm.tensor, _re_permute(_dm.axis_names))
    return _dm, _dm.tensor.reshape((2 ** len(_qubits), 2 ** len(_qubits)))