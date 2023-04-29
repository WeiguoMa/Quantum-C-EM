"""
Author: weiguo_ma
Time: 04.17.2023
Contact: weiguo.m@iphy.ac.cn
"""
import warnings

import numpy as np
import tensornetwork as tn
import torch as tc

tn.set_default_backend("pytorch")

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