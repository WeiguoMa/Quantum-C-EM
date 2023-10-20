"""
Author: weiguo_ma
Time: 04.13.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import tensornetwork as tn
import torch as tc
import numpy as np

from Library.tools import EdgeName2AxisName


def checkConnectivity(_qubits: list[tn.Node] or list[tn.AbstractNode]):

    assert len(_qubits) >= 1, 'input should be a list of qubits nodes'
    connectivity = None
    if len(_qubits) == 1:
        connectivity = False
        return connectivity
    elif len(_qubits) == 2:
        return _qubits[0].has_nondangling_edge()
    else:
        try:
            for _n in range(1, len(_qubits) - 1, 1):
                _, _ = _qubits[_n][f'bond_{_n - 1}_{_n}'], _qubits[_n][f'bond_{_n}_{_n + 1}']
                connectivity = True
        except ValueError:
            connectivity = False
        return connectivity

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

def svd_right2left(_qubits: list[tn.Node] or list[tn.AbstractNode], chi: int = None):
    r"""
    SVD from right to left

    Args:
        _qubits: list of nodes;
        chi: bond dimension.

    Returns:
        _qubits: list of nodes.
    """
    if not isinstance(_qubits, list):
        raise TypeError('input should be a list of qubits nodes')

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
                                                   name='contract_two_nodes')
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([contracted_two_nodes])
        # SVD
        _qubits[idx - 1], _qubits[idx], _ = tn.split_node(contracted_two_nodes,
                                         left_edges=_left_edges,
                                         right_edges=_right_edges,
                                         left_name=_qubits[idx - 1].name,
                                         right_name=_qubits[idx].name,
                                         edge_name=_connector,
                                         max_singular_values=chi)
        # chi=None means no truncation

def svdKappa_left2right(qubits: list[tn.Node] or list[tn.AbstractNode], kappa: int = None):
    r"""
    Perform SVD with optional dimension truncation on a list of quantum tensors.

    Args:
        qubits (list[tn.Node] or list[tn.AbstractNode]): List of quantum tensors.
        kappa (int, optional): The truncation dimension. If None, no truncation is performed.

    Returns:
        None: The function modifies the input tensors in-place.
    """
    for _num, _qubit in enumerate(qubits):
        # Shape-relating
        _qubitTensor, _qubitAxisNames = _qubit.tensor, _qubit.axis_names
        _qubitIdx = ''.join([
            'l' * (f'bond_{_num-1}_{_num}' in _qubitAxisNames),
            'i',
            'j' * (f'I_{_num}' in _qubitAxisNames),
            'r' * (f'bond_{_num}_{_num+1}' in _qubitAxisNames)
        ])

        _jIdx = _qubitIdx.find('j')
        if _jIdx != -1:
            _qubitIdxAOP = f'{_qubitIdx.replace("j", "")}j'
            _qubitTensor = tc.einsum(f'{_qubitIdx} -> {_qubitIdxAOP}', _qubitTensor)
            _shape = _qubitTensor.shape

            # SVD to truncate the inner dimension
            _u, _s, _ = tc.linalg.svd(tc.reshape(_qubitTensor, (-1, _shape[-1])), full_matrices=False)
            _s = _s.to(dtype=tc.complex128)

            # Truncate the inner dimension
            if kappa is None or kappa > _s.nelement():
                kappa = _s.nelement()

            _s = _s[: kappa]
            _u = _u[:, : kappa]

            if len(_s.shape) == 1:
                _s = tc.diag(_s)

            # Back to the former shape
            _qubitTensor = tc.einsum(f'{_qubitIdxAOP} -> {_qubitIdx}',
                                     tc.reshape(tc.matmul(_u, _s), _shape[:-1] + (kappa,)))
            _qubit.set_tensor(_qubitTensor)