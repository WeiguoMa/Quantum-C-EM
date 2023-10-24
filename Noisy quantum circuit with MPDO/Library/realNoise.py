"""
Author: weiguo_ma
Time: 05.01.2023
Contact: weiguo.m@iphy.ac.cn
"""
import os

import numpy as np
import torch as tc
from numpy.linalg import eig
from scipy.io import loadmat

from Library.tools import gates_list, name2matrix

__all__ = [
    'czExp_channel'
]


def readExpChi(filename: str = None):
    if filename is None:
        if os.path.split(os.getcwd())[1] == 'Noisy quantum circuit with MPDO':
            filename = os.path.join('data', 'chi', 'chi1.mat')
        else:
            filename = os.path.join('..', 'data', 'chi', 'chi1.mat')
    if '.mat' in filename:
        data = loadmat(filename)['exp']
        return data
    elif '.npz' in filename:
        data = np.load(filename)['chi']
        return data
    else:
        raise TypeError('Current file-type is not supported.')

def czNoisyTensor(chi, gate_factor: dict = None, dtype=tc.complex128, device: str or int = 'cpu') -> list[tc.Tensor]:
    r"""
    :param chi: Chi matrix from experiment;
    :param gate_factor: Manual API;
    :param dtype: Data type;
    :param device: Device;
    :return: qutip.Qobj

    Additional Information:
        Actual_space_size = int(np.log10(chi.shape[0]) / np.log10(4))
        N >= Actual_space_size:
            A bigger complete space can always find a subspace to describe smaller system.
        N < Actual_space_size:
            raise Error that space for describing such system is not complete.
    """
    def _czNoisySPara(_chi):
        r"""
        Get $ E_i = \sqrt{d_i}\sum_jU_{ji}\tilde{E}_j $
                where \tilde{E}_j represents gate basis
        :param _chi: Chi Matrix;
        :return: Unitary operations with linear combination coefficients.

        Sups:
            E --> {E_0: {-/}, E_1: {-/}, ...}
                    ==> E_0 --> {'I': Coefficient, ...},
                        E_0 --> {'I': Coefficient, ...},
                        ...
        """

        def __sortChiEig(_chi) -> tuple:
            r"""
            From a Chi matrix get its decomposition,
                                which is $ \chi = \sum_{xy}U_{mx}d_x\delta_xU^\dagger_{yn} $
                                         $ E_i = \sqrt{d_i}\sum_jU_{ji}\tilde{E}_j $.
            :return: chi's corresponding eigenvector and eigenvalue.
            """
            # Get eigenvalues and eigenvectors with descending order
            _s, _u = eig(_chi)
            _idx_ = _s.argsort()[::-1]
            _s = _s[_idx_]
            _u = _u[:, _idx_]
            return _u, _s

        def __opEIdx(_chi, _svd_error_tolerant: float = 1e-12) -> tuple:
            r"""

            :param _chi: Chi Matrix from experiment;
            :param _svd_error_tolerant: math in matrix operation may cause little fault or inaccuracy like 1e-33;
            :return: e as coefficient matrix(vector) for $\tilde{E}_j$;
                            non_zero_indices as idx of non-zero coefficient for zipping with its basis name in next step.
            """
            _u, _s = __sortChiEig(_chi)
            _effect_s = []
            for __value in _s:
                if np.abs(__value) > _svd_error_tolerant:
                    _effect_s.append(__value)
            # Get \vec{e}
            _e_ = [[]] * len(_effect_s)
            for _i_ in range(len(_effect_s)):
                _e_[_i_] = list(_u.T[_i_])
                for _ii_ in range(len(_e_[_i_])):
                    _e_[_i_][_ii_] = _e_[_i_][_ii_] * np.sqrt(_s[_i_])
                    if np.abs(_e_[_i_][_ii_]) < _svd_error_tolerant:
                        _e_[_i_][_ii_] = 0 + 0j
            #
            if len(_e_) == 1:  # Indicates only one eigenvalue exists
                non_zero_indices = []
                for _i_ in range(len(_e_[0])):
                    if _e_[0][_i_] != 0:
                        non_zero_indices.append(_i_)
            else:  # Not tested yet
                try:
                    non_zero_indices = [[]] * len(_e_)
                    for _i_ in range(len(_e_)):
                        for _ii_ in range(len(_e_[_i_])):
                            if _e_[_i_][_ii_] != 0:
                                non_zero_indices[_i_].append(_ii_)
                except:
                    raise IndexError('IndexError may take appear, codes here is not check yet.')
            return _e_, non_zero_indices

        _N = int(np.log10(_chi.shape[0]) / np.log10(4))

        _g_set = gates_list(_N)
        _e_matrix, index_list = __opEIdx(_chi)
        _effGate = {}
        if len(_e_matrix) == 1:
            for _idx in index_list:
                _effGate[_g_set[_idx]] = _e_matrix[0][_idx]
        else:
            for _i in range(len(_e_matrix)):
                _effGate['E_{}'.format(_i)] = {}
                for _idx in index_list[_i]:
                    _effGate['E_{}'.format(_i)][_g_set[_idx]] = _e_matrix[_i][_idx]
        return _effGate

    def is_nested_dict(d: dict):
        """
        check whether a dict is nested, True: { { } } -- False: { }.
        """
        return any(isinstance(j, dict) for j in d.values())

    if gate_factor is None:
        gate_factor = _czNoisySPara(chi)

    if is_nested_dict(gate_factor) is True:     # Multiple Error generator E_{} exist
        _tensor = []
        for _value in gate_factor.values():
            _step_tensor = tc.zeros((2, 2, 2, 2), dtype=dtype, device=device)
            for _name_, _value_ in _value.items():
                # linear equations
                _step_tensor += tc.reshape(tc.tensor(_value_, dtype=dtype, device=device) * name2matrix(_name_),
                                                                        shape=(2, 2, 2, 2))
            _tensor.append(_step_tensor)
    else:
        _tensor = tc.zeros((2, 2, 2, 2), dtype=dtype, device=device)
        for _name_, _value_ in gate_factor.items():
            # linear equations
            _tensor += tc.reshape(tc.tensor(_value_, dtype=dtype, device=device) * name2matrix(_name_),
                                                                        shape=(2, 2, 2, 2))
        _tensor = [_tensor]

    return _tensor

def czExp_channel(filename: str = None, device: int or str = 0):
    _chi = readExpChi(filename=filename)
    _czExp_tensor = tc.stack(czNoisyTensor(_chi))
    _czExp_tensor = tc.einsum('ijlmn -> jlmni', _czExp_tensor)
    return _czExp_tensor.to(device=device)