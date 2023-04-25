"""
Author: weiguo_ma
Time: 04.17.2023
Contact: weiguo.m@iphy.ac.cn
"""
import collections
import copy
import random
import string
import warnings
import numpy as np
import tensornetwork as tn
import torch as tc
from Library.tools import EdgeName2AxisName


tn.set_default_backend("pytorch")


def generate_random_string_without_duplicate(_n: int):
    r"""
    Generate a random string without duplicate characters.

    Args:
        _n: The length of the string.

    Returns:
        _str: The random string.
    """
    def _generate_random_string(_n_):
        return ''.join(random.choices(string.ascii_lowercase, k=_n_))

    _str = _generate_random_string(_n)
    while len(_str) != len(set(_str)):
        _str = _generate_random_string(_n)
    return _str


def move_index(_str: str, _idx1: int, _idx2: int):
    r"""
    Move the index from _idx1 to _idx2.

    Args:
        _str: The string;
        _idx1: The index to be moved;
        _idx2: The index to be moved to.

    Returns:
        _str: The string after moving the index.
    """
    _str = list(_str)
    _str.insert(_idx2, _str.pop(_idx1))
    return ''.join(_str)


def depolarization_noise_channel(p: float) -> tc.Tensor:
    r"""
    The depolarization noise channel.

    Args:
        p: The probability of the depolarization noise.

    Returns:
        _dpc_tensor: The depolarization noise channel.

    Additional information:
        .. math::

            \epsilon(\rho) = (1 - \frac{3p}{4})\rho + \frac{p}{4}\left(X\rho X + Y\rho Y + Z\rho Z\right)

            \epsilon(\rho) = (1 - p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right)
    """
    if p < 0 or p > 1:
        raise ValueError('The probability of the depolarization noise must be in [0, 1].')

    # Construct the error matrix
    _error_diag = tc.diag(tc.tensor([np.sqrt(1 - 3 * p / 4), np.sqrt(p / 4), np.sqrt(p / 4), np.sqrt(p / 4)],
                                    dtype=tc.complex128))
    _dpc_tensor = tc.tensor([[[1, 0],
                              [0, 1]], [[0, 1],
                                        [1, 0]], [[0, -1j],
                                                  [1j, 0]], [[1, 0],
                                                             [0, -1]]]
                            , dtype=tc.complex128)
    # Set the third edge as inner edge, which was introduced by the error_diag and the first edge as physics edge.
    _dpc_tensor = tc.einsum('ij, jfk -> fki', _error_diag, _dpc_tensor)
    return _dpc_tensor


def amp_phase_damping_error(time: float, T1: float, T2: float) -> tc.Tensor:
    r"""
    The amplitude-phase damping error.

    Args:
        time: The time of the amplitude-phase damping error;
        T1: The T1 time;
        T2: The T2 time.

    Returns:
        _apdc_tensor: The amplitude-phase damping error.

    Additional information:
        .. math::

            \epsilon(\rho) = \left(\begin{array}{cc}
            1 & 0 \\
            0 & \sqrt{1 - \frac{1}{2}e^{-\frac{t}{T_1}} - \frac{1}{2}e^{-\frac{t}{T_2}}} \\
            \end{array}\right)
            \left(\begin{array}{cc}
            1 & 0 \\
            0 & \sqrt{\frac{1}{2}e^{-\frac{t}{T_1}}} \\
            \end{array}\right)
            \left(\begin{array}{cc}
            1 & 0 \\
            0 & \sqrt{\frac{1}{2}e^{-\frac{t}{T_2}}} \\
            \end{array}\right)

    Attention:
        Time t should be as a working time in whole process, but not just a gate time.
        Time unit should be formed as nanoseconds: 'ns' uniformly.
    """
    if time < 0 or T1 <= 0 or T2 <= 0:
        raise ValueError('The time, T1, T2 must be greater than or equal to 0, '
                         'for some special cases time = 0 is allowed.')
    if time == 0:
        warnings.warn('The time is 0, which means the noise is not applied.')

    _T2p = 2 * T1 * T2 / (2 * T1 - T2)  # 1/T2 = 1/T2p + 1/(2T1)
    _param_T1 = 1 - np.exp(- time / T1)
    _param_T2 = 1 - np.exp(- time / _T2p)

    _apdc_tensor = tc.tensor([[[1, 0],
                               [0, np.sqrt(1 - (_param_T1 + _param_T2))]],
                                        [[0, 0],
                                         [0, np.sqrt(_param_T2)]],
                                                [[0, np.sqrt(_param_T1)],
                                                 [0, 0]]], dtype=tc.complex128)
    _apdc_tensor = tc.einsum('ijk -> kji', _apdc_tensor)
    return _apdc_tensor


def apply_noise_channel(_qubits: list[tn.Node] or list[tn.AbstractNode],
                        oqs: list[int] or int,
                        noise_type: str,
                        p: float = None,
                        time: float = None,
                        T1: float = None,
                        T2: float = None,
                        kappa: int = None):
    r"""
    Apply the noise channel to the qubits.

    Args:
        _qubits: The qubits to be applied the noise channel;
        oqs: The qubits to be applied the noise channel;
        noise_type: The type of the noise channel;
        p: The probability of the noise channel;
        time: The time of the noise channel;
        T1: The T1 time of the noise channel;
        T2: The T2 time of the noise channel;
        kappa: Truncation dimension upper bond of the noise channel.

    Returns:
        _qubits: The qubits after applying the noise channel.
            !!! Actually no return, but the qubits are changed in the memory. !!!

    Additional information:
        The noise channel is applied to the qubits by the following steps:
            1. Construct the noise channel;
            2. Construct the error tensor;
            3. Contract the error tensor with the qubits;
            4. Fix the axis format of the qubits.

    Attention:
        On account for the function _hard_fix_axis_format,
            The qubits should be in the following format:
                1. The first edge is the bond edge;
                2. The second edge is the physics edge;
                3. The third edge is the bond edge.
            The qubits should be in the following format:
                1. The first edge is the bond edge;
                2. The second edge is the physics edge;
                3. The third edge is the bond edge.
    """
    def _import_error_tensor(_noise_type_: str, _p_: float):
        if _noise_type_ == 'depolarization':
            return depolarization_noise_channel(_p_)
        elif _noise_type_ == 'amplitude_phase_damping_error':
            return amp_phase_damping_error(time, T1, T2)
        else:
            raise NotImplementedError(f'Noise type {_noise_type_} is not implemented yet.')

    def _find_duplicate(_lst_):
        """
        Find the duplicate items and their indices in a list.
        """
        _duplicate_item_ = [item for item, count in collections.Counter(_lst_).items() if count > 1]
        _duplicate_idx_ = [idx for idx, item in enumerate(_lst_) if item in _duplicate_item_]
        return _duplicate_item_, _duplicate_idx_

    if not isinstance(_qubits, list):
        if not isinstance(_qubits, tn.Node) or not isinstance(_qubits, tn.AbstractNode):
            raise TypeError(f'qubits must be a list of tn.Node or tn.AbstractNode, but got {type(_qubits)}')
        _qubits = [_qubits]
    if not isinstance(oqs, list):
        if not isinstance(oqs, int):
            raise TypeError(f'oqs must be a list of int, but got {type(oqs)}')
        oqs = [oqs]
    if len(oqs) > len(_qubits):
        raise ValueError(f'len(oqs) must be less than or equal to to len(qubits),'
                         f' but got {len(oqs)} and {len(_qubits)}')
    if p is None and (time is None or T1 is None or T2 is None):
        raise ValueError('The noise parameter must be specified.')

    # Create Node for noise channel
    _noise_tensor = _import_error_tensor(_noise_type_=noise_type, _p_=p)
    _noise_nodeList = []
    for _oq in oqs:
        _noise_node = tn.Node(_noise_tensor, name='noise_node',
                              axis_names=['inner', f'physics_{_oq}', f'I_{_oq}'])
        # copy.deepcopy is necessary to avoid the error of node reuse.
        _noise_nodeList.append(copy.deepcopy(_noise_node))

    # Operating the noise channel to qubits
    for _ii, _qnum in enumerate(oqs):
        _edge = tn.connect(_qubits[_qnum][f'physics_{_qnum}'], _noise_nodeList[_ii]['inner'])
        _qubits[_qnum] = tn.contract(_edge, name=f'qubit_{_qnum}')
        # ProcessFunction, for details, see the function definition.
        EdgeName2AxisName([_qubits[_qnum]])   # Tensor append a new rank call 'I_{}'.format(_qnum) here.
        # raise NotImplementedError('When double/multi errors are applied to a same qubit, problem occurs.'
        #                           'The reason is that the node connection broken while the node is working.')

        _dup_item, _dup_idx = _find_duplicate(_qubits[_qnum].axis_names)
        if _dup_item:
            # Number of axis name before the reshape operation(contain duplicates)
            _length = len(_qubits[_qnum].axis_names)
            # Find the shape of the tensor after the reshape operation
            _reshape_shape = copy.deepcopy(list(_qubits[_qnum].tensor.shape))
            _reshape_shape[_dup_idx[1]] = _reshape_shape[_dup_idx[0]] * _reshape_shape[_dup_idx[1]]
            _reshape_shape.pop(_dup_idx[0])
            # Generate a random string without duplicates, if len = 4, then the string is 'abcd' as Einstein notation.
            _random_string = generate_random_string_without_duplicate(_length)
            _random_string_reorder = move_index(_random_string, _dup_idx[0], _dup_idx[1]-1)  # Like 'ifcvbj' -> 'fcvbij'
            # Reshape the tensor
            _reshaped_tensor = tc.einsum(_random_string + ' -> ' + _random_string_reorder, _qubits[_qnum].tensor)\
                .reshape(_reshape_shape)
            _axis_names = copy.deepcopy(_qubits[_qnum].axis_names)
            _axis_names.pop(_dup_idx[0])

            """ Though we clarify a new node called _left_node, in memory, 
                        it is still the same node as connected to _qubits[_qnum]. """
            _left_node, _right_node = _qubits[_qnum][0].get_nodes()     # 0 is the hard code that bond_idx at first pos
            if _right_node is not None:     # Which means that the node's the most left edge is not bond_edge
                if 'bond' not in _qubits[_qnum][0].name:
                    raise ValueError(f'HardCodeERROR. The edge name must be bond, but got {_qubits[_qnum][0].name}')
                _left_edge_name = _left_node.axis_names[-1]     # -1 is the hard code that bond_idx at last pos
                _qubits[_qnum][0].disconnect(_left_edge_name, 'right_edge')
                # ProcessFunction, for details, see the function definition
                EdgeName2AxisName([_left_node])

            _qubits[_qnum] = tn.Node(_reshaped_tensor,
                                   name=f'qubit_{_qnum}',
                                   axis_names=_axis_names)  # Node's edge's, named 'I_{}', dimension has been promoted.
            if _right_node is not None:
                tn.connect(_left_node[_left_edge_name], _qubits[_qnum][0], name=_qubits[_qnum].axis_names[0])
                EdgeName2AxisName([_qubits[_qnum]])

        # Shape-relating
        _shape = _qubits[_qnum].tensor.shape
        _left_edge_shape = [_shape[_ii_] for _ii_ in range(len(_shape) - 1)]
        _left_dim = np.prod(_left_edge_shape)
        # SVD to truncate the inner dimension
        _u, _s, _ = tc.linalg.svd(tc.reshape(_qubits[_qnum].tensor, (_left_dim, _shape[-1])), full_matrices=False)
        _s = _s.to(dtype=tc.complex128)

        # Truncate the inner dimension
        if kappa is None:
            kappa = _s.nelement()
        _s = _s[: kappa]
        _u = _u[:, : kappa]

        if len(_s.shape) == 1:
            _s = tc.diag(_s)

        # Back to the former shape
        _left_edge_shape.append(_s.shape[-1])
        _qubits[_qnum].tensor = tc.reshape(tc.matmul(_u, _s), _left_edge_shape)
