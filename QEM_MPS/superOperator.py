"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import torch as tc
from numpy import prod
import tensornetwork as tn
from Library.realNoise import czExp_channel
from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate


tn.set_default_backend("pytorch")


class SuperOperator(object):
	def __init__(self, operator: tc.Tensor or tn.AbstractNode = None, noisy: bool = True):
		self.originalINPUT = operator
		self.shape = operator.shape
		self.qubitNum = int((len(self.shape) - 1) / 2)
		self.noisy = noisy
		self.axisNames = self._getAxisNames()

		if operator is None:
			operator = czExp_channel('/Users/weiguo_ma/Python_Program/Quantum_error_mitigation'
			                         '/Noisy quantum circuit with MPDO/data/chi/chi1.mat')

		if not isinstance(operator, tc.Tensor) and not isinstance(operator, tn.AbstractNode):
			raise TypeError("operator must be a tensor or a node")
		if isinstance(operator, tc.Tensor):
			if len(self.shape) != len(self.axisNames):
				raise ValueError(f'Shape {self.shape} not match with axisNames {self.axisNames},'
				                 f' check the noisy status and its corresponds tensor shape.')
			self.operator = tn.Node(operator, name='realNoise', axis_names=self.axisNames)
		if isinstance(operator, tn.Node):
			if operator.axis_names != self.axisNames:
				raise ValueError("operator axis names must be uniformed as {}".format(self.axisNames))
			self.operator = copy.deepcopy(operator)

		self.superOperator = None
		self.superOperatorMPO = None

		self.uMPO()

	def _getAxisNames(self):
		_axisNames = [f'physics_{_i}' for _i in range(self.qubitNum)] + \
		             [f'inner_{_i}' for _i in range(self.qubitNum)]
		if self.noisy is True:
			_axisNames.append('I')
		return _axisNames

	def _getTTAxisNames(self):
		_TTAxisNames = []
		for _i in reversed(range(self.qubitNum)):
			_TTAxisNames += [f'Dinner_{_i}', f'Dphysics_{_i}']
		for _i in range(self.qubitNum):
			_TTAxisNames += [f'physics_{_i}', f'inner_{_i}']
		return _TTAxisNames

	def getSuperOperator(self) -> tn.AbstractNode:
		_daggerAxisNames = ['D' + _name for _name in self.axisNames if _name != 'I'] + ['I']
		_operatorDagger = tn.Node(self.operator.tensor.conj(), name='realNoiseDagger', axis_names=_daggerAxisNames)
		tn.connect(self.operator['I'], _operatorDagger['I'])
		_superOperatorNode = tn.contract_between(self.operator, _operatorDagger,
		                                         name='superOperatorU', allow_outer_product=True)
		# Process Function
		EdgeName2AxisName(_superOperatorNode)

		self.superOperator = _superOperatorNode
		return _superOperatorNode

	def uMPO(self) -> list[tn.AbstractNode]:
		# TT-decomposition to superOperatorNode
		u = self.getSuperOperator()
		_TTAxisNames = self._getTTAxisNames()

		TTSeries = []
		_leftNode, _rightNode = None, None

		_countA, _countB = self.qubitNum - 1, 0
		_countA_, _countB_ = self.qubitNum - 1, 0

		for _i in range(self.qubitNum * 2):
			_leftNames = _TTAxisNames[0:2]
			for _item in _leftNames:
				_TTAxisNames.remove(_item)
			_rightNames = _TTAxisNames
			if _rightNames:
				if 0 < _i <= self.qubitNum - 1:
					_leftNames.append(f'Dbond_{_countA - 1}_{_countA}')
					_countA -= 1
				elif _i == self.qubitNum:
					_leftNames.append('Ebond')
				elif _i > self.qubitNum:
					_leftNames.append(f'bond_{_countB}_{_countB + 1}')
					_countB += 1
				else:
					pass

				if _i > 0:
					u = _rightNode

				_leftEdges, _rightEdges = [u[_edgeName] for _edgeName in _leftNames], [u[_edgeName] for _edgeName in _rightNames]

				if _i < self.qubitNum - 1:
					_leftNodeName, _rightNodeName = f'DU_{_countA_}', f'DU_{_countA_ - 1}'
					_edgeName = f'Dbond_{_countA_ - 1}_{_countA_}'
					_countA_ -= 1
				elif _i == self.qubitNum - 1:
					_leftNodeName, _rightNodeName = f'DU_{_countA}', f'U_{_countB_}'
					_edgeName = 'Ebond'
				else:
					_leftNodeName, _rightNodeName = f'DU_{_countB_}', f'U_{_countB_ + 1}'
					_edgeName = f'bond_{_countB_}_{_countB_ + 1}'
					_countB_ += 1

				_leftNode, _rightNode, _ = tn.split_node(node=u, left_edges=_leftEdges, right_edges=_rightEdges,
				                                         left_name=_leftNodeName, right_name=_rightNodeName, edge_name=_edgeName)

				TTSeries.append(_leftNode)
			else:
				TTSeries.append(_rightNode)
		self.superOperatorMPO = TTSeries
		return TTSeries


if __name__ == '__main__':
	from Library.realNoise import czExp_channel

	realNoiseTensor = czExp_channel(
		'/Users/weiguo_ma/Python_Program/Quantum_error_mitigation/Noisy quantum circuit with MPDO/data/chi/chi1.mat')

	superOperator = SuperOperator(realNoiseTensor).superOperatorMPO