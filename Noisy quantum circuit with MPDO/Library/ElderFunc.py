"""
Author: weiguo_ma
Time: 04.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import tensornetwork as tn
import torch as tc

from Library.tools import EdgeName2AxisName


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


def calculate_DM(_qubits, noisy: bool = False, reduced_index: list = None):
    r"""
    Calculate the density matrix of the state.

    Args:
        _qubits: qubits;
        noisy: whether the density matrix is noisy;
        reduced_index: the _qubits[index] to be reduced, which means the physics_con-physics of sub-density matrix
                            will be connected and contracted.

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
            _allowed_outer_product = False  # Edges between ket-bra are now connected, outer product is not allowed.
        _contract_nodes.append(tn.contract_between(_qubits[i], _qubits_conj[i], name=f'contracted_qubit_{i}',
                                                   allow_outer_product=_allowed_outer_product))
    EdgeName2AxisName(_contract_nodes)

    # Reduced density matrix
    # Bra--Ket space of each qubit in memory are now contracted into a higher rank tensor with two physical indices.
    if reduced_index is not None:
        if isinstance(reduced_index, int):
            reduced_index = [reduced_index]
        if not isinstance(reduced_index, list):
            raise TypeError('reduced_index should be int or list[int]')
        for _idx in reduced_index:
            tn.connect(_contract_nodes[_idx][f'physics_{_idx}'], _contract_nodes[_idx][f'con_physics_{_idx}'])
            _contract_nodes[_idx] = tn.contract_trace_edges(_contract_nodes[_idx])
    else:
        reduced_index = []

    _dm = _contract_nodes[0]
    for _ii in range(1, len(_contract_nodes)):
        _dm = tn.contract_between(_dm, _contract_nodes[_ii], allow_outer_product=True, name='Contracted_DM_NODE')
    EdgeName2AxisName([_dm])

    _dm.tensor = tc.permute(_dm.tensor, _re_permute(_dm.axis_names))

    _reshape_size = len(_qubits) - len(reduced_index)
    return _dm, _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size))
