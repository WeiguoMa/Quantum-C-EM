"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
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

def add_gate(_qubits: list, _gate: TensorGate, _oqs: list):
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
    if single is False:     # Two-qubit gate
        if len(_oqs) > 2:
            raise NotImplementedError('Only two-qubit gates are supported currently.')
        if _oqs[0] == _oqs[1]:
            raise ValueError('Operating qubits must be different.')
        if is_nested(_oqs) is True:
            raise NotImplementedError('Series CNOT gates are not supported yet.')
        else:
            _edges = []
            _gate = tn.Node(_gate.tensor, name=_gate.name,
                            axis_names=[f'inner_{_oqs[0]}', f'inner_{_oqs[1]}', f'physics_{_oqs[0]}', f'physics_{_oqs[1]}'])
            # Created a new node in memory
            _contract_qubits = tn.contract_between(_qubits[_oqs[0]], _qubits[_oqs[1]],
                                                   name='q_{}_{}'.format(_oqs[0], _oqs[1]),
                                                   allow_outer_product=True)
            EdgeName2AxisName([_contract_qubits])
            for _i, _bit in enumerate(_oqs):
                _edges.append(tn.connect(_contract_qubits['physics_{}'.format(_bit)], _gate[f'inner_{_bit}']))

            # contract the connected edge and inherit the name of the pre-qubit
            _gate = tn.contract_between(_contract_qubits, _gate, name=_gate.name)

            # ProcessFunction, for details, see the function definition.
            EdgeName2AxisName([_gate])

            # Split back to two qubits
            _left_AxisName, _right_AxisName = _qubits[_oqs[0]].axis_names, _qubits[_oqs[1]].axis_names
            _left_edges, _right_edges = [_gate[name] for name in _left_AxisName],\
                                        [_gate[name] for name in _right_AxisName]

            _qubits[_oqs[0]], _qubits[_oqs[1]], _ = tn.split_node(_gate,
                                                                  left_edges=_left_edges,
                                                                  right_edges=_right_edges,
                                                                  left_name=f'qubit_{_oqs[0]}',
                                                                  right_name=f'qubit_{_oqs[1]}',
                                                                  edge_name=f'bond_{_oqs[0]}_{_oqs[1]}')
    else:
        _gate_list = [tn.Node(_gate.tensor, name=_gate.name, axis_names=[f'inner_{_idx}', f'physics_{_idx}'])
                                                                                                for _idx in _oqs]
        for _i, _bit in enumerate(_oqs):
            tn.connect(_qubits[_bit]['physics_{}'.format(_bit)], _gate_list[_i][f'inner_{_bit}'])
            # contract the connected edge and inherit the name of the pre-qubit
            _qubits[_bit] = tn.contract_between(_qubits[_bit], _gate_list[_i], name=_qubits[_bit].name)
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