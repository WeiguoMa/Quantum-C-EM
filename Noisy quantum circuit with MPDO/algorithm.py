"""
Author: weiguo_ma
Time: 04.13.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
from tools import EdgeName2AxisName
import copy

def qr_left2right(_qubits: list[tn.Node] or list[tn.AbstractNode]):
    r"""
    QR decomposition from left to right.

    Args:
        _qubits: List of qubits (as nodes).

    Returns:
        Nodes after QR decomposition, on progress.
    """
    if not isinstance(_qubits, list):
        raise TypeError('input should be a list of qubits nodes')

    for _i in range(len(_qubits) - 1):
        _left_edges_name, _right_edges_name = copy.deepcopy(_qubits[_i].axis_names), []
        for _name in _qubits[_i].axis_names:
            if 'bond_{}_'.format(_i) in _name:
                _left_edges_name.remove(_name)
                _right_edges_name.append(_name)
        _left_edges = [_qubits[_i][_name] for _name in _left_edges_name]
        _right_edges = [_qubits[_i][_name] for _name in _right_edges_name]
        _q, _r = tn.split_node_qr(_qubits[_i],
                                  left_edges=_left_edges,
                                  right_edges=_right_edges,
                                  left_name=f'qubit_{_i}',
                                  right_name='right_waiting4contract2form_right',
                                  edge_name=f'qrbond_{_i}_{_i+1}')
        _r = tn.contract_between(_r, _qubits[_i+1])
        _r.name = 'qubit_{}'.format(_i+1)
        _qubits[_i], _qubits[_i+1] = _q, _r
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([_qubits[_i], _qubits[_i + 1]])

def svd_right2left(_qubits: list[tn.Node] or list[tn.AbstractNode], _chi: int = None):
    r"""
    SVD from right to left

    Args:
        _qubits: list of nodes;
        _chi: bond dimension.

    Returns:
        _qubits: list of nodes.
    """
    if not isinstance(_qubits, list):
        raise TypeError('input should be a list of qubits nodes')

    if _chi is None:
        # A number who is big enough to keep all the information
        chi = 2 ** len(_qubits)
    for idx in range(len(_qubits) - 1, 0, -1):
        # SVD name cluster
        _left_edges = copy.deepcopy(_qubits[idx - 1].axis_names)
        _right_edges = copy.deepcopy(_qubits[idx].axis_names)
        for _left_name in _left_edges:
            for _right_name in _right_edges:
                if _left_name == _right_name:
                    # _connector is the connection edge name of two nodes where svd used to connect two nodes first
                    _connector = _left_name
                    _left_edges.remove(_connector), _right_edges.remove(_connector)
        _left_edges = [_qubits[idx - 1][_left_name] for _left_name in _left_edges]
        _right_edges = [_qubits[idx][_right_name] for _right_name in _right_edges]
        # Contract
        contracted_two_nodes = tn.contract_between(_qubits[idx - 1],
                                                   _qubits[idx],
                                                   name='contract_two_nodes',
                                                   allow_outer_product=True)    # May cause math error
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([contracted_two_nodes])
        # SVD
        _qubits[idx - 1], _qubits[idx], _ = tn.split_node(contracted_two_nodes,
                                         left_edges=_left_edges,
                                         right_edges=_right_edges,
                                         left_name=_qubits[idx - 1].name,
                                         right_name=_qubits[idx].name,
                                         edge_name=_connector,
                                         max_singular_values=_chi)