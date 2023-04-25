"""
Author: weiguo_ma
Time: 04.20.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import numpy as np
from Library.basic_operations import tensorDot

def pauli_z():
	_tensor = tc.tensor(
		[[1, 0],
		 [0, -1]], dtype=tc.complex128
	)
	return _tensor

def tensor_sigmaz(_N: int, _order: int):
	ls_qeye = [pauli_z()] * _N
	ls_qeye[_order] = pauli_z()
	return tensorDot(ls_qeye)

def ham_near_matrix(_N: int):
	""" A matrix computation for computing the expectation with DM or STATE is simple,
	        there is no need to use MPS or TN. """
	_Ham = 0
	for _i in range(_N-1):
		_j = _i + 1
		_Ham += - tc.matmul(tensor_sigmaz(_N, _i), tensor_sigmaz(_N, _j))
	return tc.tensor(np.asarray(_Ham), dtype=tc.complex128)