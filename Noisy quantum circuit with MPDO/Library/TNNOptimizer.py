"""
Author: weiguo_ma
Time: 04.13.2023
Contact: weiguo.m@iphy.ac.cn
"""

import tensornetwork as tn
import torch as tc

from Library.tools import EdgeName2AxisName

__all__ = [
    'qr_left2right',
    'svd_left2right',
    'svd_right2left',
    'svdKappa_left2right'
]

def checkConnectivity(_qubits: list[tn.Node] or list[tn.AbstractNode]):
    """
    Check if the qubits have connectivity.

    Args:
        _qubits: List of qubit nodes.

    Returns:
        True if qubits are connected, False otherwise.
    """
    connectivity = None
    if len(_qubits) <= 1:
        return False
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
        _left_edges_name, _right_edges_name = [name for name in _qubits[_i].axis_names if f'bond_{_i}_' not in name],\
            [name for name in _qubits[_i].axis_names if f'bond_{_i}_' in name]
        _left_edges, _right_edges = [_qubits[_i][_name] for _name in _left_edges_name],\
            [_qubits[_i][_name] for _name in _right_edges_name]
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
    SVD from right to left.

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
        _left_edges = [name for name in _qubits[idx - 1].axis_names if name not in _qubits[idx].axis_names]
        _right_edges = [name for name in _qubits[idx].axis_names if name not in _qubits[idx - 1].axis_names]

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
                                         edge_name=f'bond_{idx-1}_{idx}',
                                         max_singular_values=chi)
        EdgeName2AxisName([_qubits[idx - 1], _qubits[idx]])

def svd_left2right(_qubits: list[tn.Node] or list[tn.AbstractNode], chi: int = None):
    """
        SVD from left to right.

        Args:
            _qubits: list of nodes;
            chi: bond dimension.

        Returns:
            _qubits: list of nodes.
        """
    if not isinstance(_qubits, list):
        raise TypeError('input should be a list of qubits nodes')

    for idx in range(len(_qubits) - 1):
        # SVD name cluster
        _left_edges = [name for name in _qubits[idx].axis_names if name not in _qubits[idx + 1].axis_names]
        _right_edges = [name for name in _qubits[idx + 1].axis_names if name not in _qubits[idx].axis_names]

        _left_edges = [_qubits[idx][_left_name] for _left_name in _left_edges]
        _right_edges = [_qubits[idx + 1][_right_name] for _right_name in _right_edges]

        # Contract
        contracted_two_nodes = tn.contract_between(_qubits[idx],
                                                   _qubits[idx + 1],
                                                   name='contract_two_nodes')
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([contracted_two_nodes])
        # SVD
        _qubits[idx], _qubits[idx + 1], _ = tn.split_node(contracted_two_nodes,
                                                          left_edges=_left_edges,
                                                          right_edges=_right_edges,
                                                          left_name=_qubits[idx].name,
                                                          right_name=_qubits[idx + 1].name,
                                                          edge_name=f'nbond_{idx}_{idx + 1}',
                                                          max_singular_values=chi)
        EdgeName2AxisName([_qubits[idx], _qubits[idx + 1]])

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