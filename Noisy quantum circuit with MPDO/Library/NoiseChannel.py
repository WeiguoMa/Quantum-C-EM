"""
Author: weiguo_ma
Time: 04.17.2023
Contact: weiguo.m@iphy.ac.cn
"""
import warnings

import numpy as np
import tensornetwork as tn
import torch as tc

from Library.TensorOperations import tensorDot
from Library.chipInfo import Chip_information
from Library.tools import select_device

tn.set_default_backend("pytorch")

class NoiseChannel(object):
    def __init__(self, chip: str = None, dtype=tc.complex128, device: str or int = 'cpu'):
        self.dtype = dtype
        self.device = select_device(device)

        # chip Info
        if chip is None:
            # chip = 'beta4Test'
            chip = 'worst4Test'
        self.chip = Chip_information().__getattribute__(chip)()
        self.T1 = self.chip.T1
        self.T2 = self.chip.T2
        self.GateTime = self.chip.gateTime
        self.dpc_errorRate = self.chip.dpc_errorRate

        self._basisPauli = [tc.tensor([[1, 0], [0, 1]], dtype=dtype, device=device),
                            tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
                            tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
                            tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)]

        # Noise channels' tensor
        self.dpCTensor = self.depolarization_noise_channel(p=self.dpc_errorRate)
        self.dpCTensor2 = self.depolarization_noise_channel(p=self.dpc_errorRate, qn=2)
        self.apdeCTensor = self.amp_phase_damping_error(time=self.GateTime, T1=self.T1, T2=self.T2)

    def depolarization_noise_channel(self, p: float, qn: int = 1) -> tc.Tensor:
        r"""
        The depolarization noise channel.

        Args:
            p: The probability of the depolarization noise.
            qn: The number of qubits.

        Returns:
            _dpc_tensor: The depolarization noise channel.

        Additional information:
            .. math::

                \epsilon(\rho) = (1 - \frac{3p}{4})\rho + \frac{p}{4}\left(X\rho X + Y\rho Y + Z\rho Z\right)

                \epsilon(\rho) = (1 - p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right)
        """

        def _iter_tensorDot(_inList: list, _qn: int):
            __outList = []
            for __element in _inList:
                __outList += [tensorDot(__element, __elementIn) for __elementIn in self._basisPauli]
            _qn -= 1
            if _qn == 1:
                return __outList
            elif _qn == 0:
                return self._basisPauli
            else:
                return _iter_tensorDot(__outList, _qn)

        if p < 0 or p > 1:
            raise ValueError('The probability of the depolarization noise must be in [0, 1].')

        # Construct the error matrix
        _error_probList = [np.sqrt(1 - (4 ** qn - 1) * p / (4 ** qn))] + [np.sqrt(p / (4 ** qn))] * (4 ** qn - 1)
        _error_diag = tc.diag(tc.tensor(_error_probList, dtype=tc.complex128, device=self.device))
        _dpc_tensor = tc.stack(_iter_tensorDot(self._basisPauli, qn)).to(device=self.device)
        # Set the third edge as inner edge, which was introduced by the error_diag and the first edge as physics edge.
        _dpc_tensor = tc.einsum('ij, jfk -> fki', _error_diag, _dpc_tensor)
        # Reshape as tensor
        _dpc_tensor = _dpc_tensor.reshape([2] * (2 * qn) + [_dpc_tensor.shape[-1]])
        return _dpc_tensor

    def amp_phase_damping_error(self, time: float, T1: float, T2: float) -> tc.Tensor:
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
                                                     [0, 0]]], dtype=self.dtype, device=self.device)
        _apdc_tensor = tc.einsum('ijk -> jki', _apdc_tensor)
        return _apdc_tensor


if __name__ == '__main__':
    noise = NoiseChannel()
    dpCTensor = noise.dpCTensor2
    print(dpCTensor.shape)
    cnotTensor = tc.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=tc.complex128).reshape((2, 2, 2, 2))
    exTensorB = tc.einsum('ijklp, klmn -> ijmnp', dpCTensor, cnotTensor)
    print(exTensorB[:, :, :, :, 4])
