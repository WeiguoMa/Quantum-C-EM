"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
from typing import List, Union

import tensornetwork as tn

from Library.TNNOptimizer import svd_left2right
from Library.tools import EdgeName2AxisName
from superOperator import SuperOperator

tn.set_default_backend("pytorch")


class ErrorInverse:
	def __init__(self, idealMPO: Union[List[tn.AbstractNode], SuperOperator],
	             uPMPO: Union[List[tn.AbstractNode], SuperOperator], chi: int = None):
		self.chi = chi
		self.uMPO, self.uPMPO = idealMPO, uPMPO
		self.qubitNum = int(len(self.uPMPO) / 2)

		self.EpsilonMinus1 = self.calErrorInverse()

	def calErrorInverse(self):
		def _getName(num: int):
			if num < self.qubitNum:
				return f'DEpsilonMinus1_{num}'
			else:
				return f'EpsilonMinus1_{num - self.qubitNum}'
		# Connection
		for _num in range(self.qubitNum):  # Dual Space
			tn.connect(self.uMPO[_num][f'Dinner_{_num}'], self.uPMPO[_num][f'Dphysics_{_num}'])
			tn.connect(self.uMPO[_num + self.qubitNum][f'inner_{_num}'], self.uPMPO[_num + self.qubitNum][f'physics_{_num}'])

		EpsilonMinus1 = [
			tn.contract_between(self.uMPO[_num], self.uPMPO[_num], name=_getName(_num)) for _num in range(2 * self.qubitNum)
		]
		EdgeName2AxisName(EpsilonMinus1)

		if self.chi is not None:
			svd_left2right(EpsilonMinus1, chi=self.chi)
		return EpsilonMinus1