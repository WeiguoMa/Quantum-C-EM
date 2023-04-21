"""
Author: weiguo_ma
Time: 04.20.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import numpy as np
from basic_gates import TensorGate
from basic_operations import tensorDot

Gates = TensorGate()

def tensor_sigmaz(_N: int, _order: int):
	ls_qeye = [Gates.i().tensor] * _N
	ls_qeye[_order] = Gates.z().tensor
	return tensorDot(ls_qeye)

def ham_near_matrix(_N: int):
	_Ham = 0
	for _i in range(_N-1):
		_j = _i + 1
		_Ham += - tc.matmul(tensor_sigmaz(_N, _i), tensor_sigmaz(_N, _j))
	return tc.tensor(np.asarray(_Ham), dtype=tc.complex128)


if __name__ == '__main__':
	N = 3
	Ham = ham_near_matrix(N)
	print(Ham)