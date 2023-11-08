"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import List, Union

import tensornetwork as tn
import torch as tc

from Library.TNNOptimizer import svd_left2right
from Library.tools import EdgeName2AxisName
from Maps import Maps
from UpdateMAPs import UpdateNODES
from superOperator import SuperOperator

tn.set_default_backend("pytorch")

__all__ = [
	'calInverse'
]


def calInverse(
		tensor: Union[tc.Tensor, tn.AbstractNode],
		idealTensor: Union[tc.Tensor, tn.AbstractNode],
		epoch: int = 5,
		chi: int = None,
		_noisy: bool = True,
		_mpo: bool = True
) -> Union[List[tn.AbstractNode], tc.Tensor]:
	"""
	Compute the inverse of a tensor using tensor network methods.

    Args:
        tensor: The tensor to be inverted.
        idealTensor: The ideal reference tensor.
        epoch: Number of iterations for inversion.
        chi: Truncation dimension for SVD.
        _noisy: Flag indicating if noise is considered (build-in for test);
        _mpo : Return mpo or tensor (build-in for test).

    Returns:
        List of tn.Node representing the inverse of the tensor.
    """
	def _getStandardAxisNames(qubitNum: int, _phy: bool = True):
		if _phy:
			return [f'{prefix}{i}' for prefix in ('Dphysics_', 'physics_') for i in range(qubitNum)]
		else:
			return [f'{prefix}{i}' for prefix in ('Dinner_', 'inner_') for i in range(qubitNum)]

	_uMPO = SuperOperator(tensor, noisy=_noisy).superOperatorMPO
	_idealMPO = SuperOperator(idealTensor, noisy=False).superOperatorMPO

	_maps = Maps(superOperatorMPO=_uMPO)
	_uPMPO = UpdateNODES(maps=_maps, epoch=epoch).uPMPO

	_inverseMPO = ErrorInverse(idealMPO=_idealMPO, uPMPO=_uPMPO, chi=chi).EpsilonMinus1

	if _mpo:
		return _inverseMPO
	else:
		_outOrder = [_inverseMPO[_i][_name] for _i, _name in enumerate(_getStandardAxisNames(len(_uMPO) // 2))] +\
		            [_inverseMPO[_i][_name] for _i, _name in enumerate(_getStandardAxisNames(len(_uMPO) // 2, _phy=False))]
		_inverseNode = tn.contractors.auto(_inverseMPO, output_edge_order=_outOrder)
		EdgeName2AxisName([_inverseNode])
		return _inverseNode.tensor

class ErrorInverse:
	def __init__(self, idealMPO: Union[List[tn.AbstractNode], SuperOperator],
	             uPMPO: Union[List[tn.AbstractNode], SuperOperator], chi: int = None):
		self.chi = chi
		self.uMPO, self.uPMPO = idealMPO, uPMPO
		self.qubitNum = int(len(self.uPMPO) / 2)

		self.EpsilonMinus1 = self.calErrorInverse()

	def calErrorInverse(self):
		def _getName(index: int):
			prefix = 'DEpsilonMinus1_' if index < self.qubitNum else 'EpsilonMinus1_'
			return f'{prefix}{index % self.qubitNum}'

		# Connection
		for _num in range(self.qubitNum):  # Dual Space
			tn.connect(self.uMPO[_num][f'Dinner_{_num}'], self.uPMPO[_num][f'Dphysics_{_num}'])
			tn.connect(self.uMPO[_num + self.qubitNum][f'inner_{_num}'],
			           self.uPMPO[_num + self.qubitNum][f'physics_{_num}'])

		EpsilonMinus1 = [
			tn.contract_between(self.uMPO[_num], self.uPMPO[_num], name=_getName(_num)) for _num in
			range(2 * self.qubitNum)
		]
		EdgeName2AxisName(EpsilonMinus1)

		if self.chi is not None:
			svd_left2right(EpsilonMinus1, chi=self.chi)

		return EpsilonMinus1
