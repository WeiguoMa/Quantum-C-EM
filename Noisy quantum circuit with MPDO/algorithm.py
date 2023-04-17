"""
Author: weiguo_ma
Time: 04.13.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
from tools import sort_edges4dep, EdgeName2AxisName, rename_edgeAxis
import copy

def qr_left2right_old(_qubits: list):
    """
    QR decomposition from left to right, old version, may cause DecompositionError.

    Args:
        _qubits: List of qubits, which is a list of nodes.

    Returns:
        Nodes after QR decomposition, on progress.
    """
    # left-most
    _left_edges, _right_edges = sort_edges4dep(_qubits[0], _op_idx=[0], _purpose='qr')
    _q, _r = tn.split_node_qr(_qubits[0],
                              left_edges=_left_edges,
                              right_edges=_right_edges,
                              left_name=_qubits[0].name,
                              right_name='right_waiting4contract2form_right',
                              edge_name='qrbond_{}_{}'.format(0, 1))
    _r = _r @ _qubits[1]
    _r.name = 'qubit_' + str(0 + 1)
    _qubits[0], _qubits[1] = _q, _r
    # ProcessFunction, for details, see the function definition.
    EdgeName2AxisName([_qubits[0], _qubits[1]])

    # left to right
    for ii in range(1, len(_qubits) - 1):
        _left_edges, _right_edges = sort_edges4dep(_qubits[ii], _op_idx=[ii], _purpose='qr')
        _q, _r = tn.split_node_qr(_qubits[ii],
                                  left_edges=_left_edges,
                                  right_edges=_right_edges,
                                  left_name=_qubits[ii].name,
                                  edge_name='qrbond_{}_{}'.format(ii, ii + 1))
        _r = tn.contract_between(_r, _qubits[ii + 1], allow_outer_product=True)     # May cause math error
        # _r = _r @ _qubits[ii + 1]
        _r.name = 'qubit_' + str(ii + 1)
        _qubits[ii], _qubits[ii + 1] = _q, _r
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([_qubits[ii], _qubits[ii + 1]])

def svd_right2left_old(_qubits, chi: int = None):
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
    contracted_two_nodes = tn.contract_between(node1=_qubits[idx - 1],
                                               node2=_qubits[idx],
                                               name='contracted_two_nodes',
                                               allow_outer_product=True)    # May cause math error
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

def qr_left2right(_qubits: list[tn.Node] or list[tn.AbstractNode]):
    """
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
                                  left_name='q{}'.format(_i),
                                  right_name='right_waiting4contract2form_right',
                                  edge_name='qrbond_{}_{}'.format(_i, _i+1))
        _r = tn.contract_between(_r, _qubits[_i+1])
        _r.name = 'qubit_{}'.format(_i+1)
        _qubits[_i], _qubits[_i+1] = _q, _r
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([_qubits[_i], _qubits[_i + 1]])

def svd_right2left(_qubits: list[tn.Node] or list[tn.AbstractNode], _chi: int = None):
    """
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
        print('---------')
        print(contracted_two_nodes.axis_names)
        _left, _right, _ = tn.split_node(contracted_two_nodes,
                                         left_edges=_left_edges,
                                         right_edges=_right_edges,
                                         left_name=_qubits[idx - 1].name,
                                         right_name=_qubits[idx].name,
                                         edge_name=_connector,
                                         max_singular_values=_chi)
        _qubits[idx - 1], _qubits[idx] = _left, _right
        _qubits = rename_edgeAxis(_qubits, [idx - 1, idx])
        # /|\ May cost time for a whole check of qubits, but it's easy to generalize to q>=3.