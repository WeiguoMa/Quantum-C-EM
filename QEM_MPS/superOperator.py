"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import torch as tc
import tensornetwork as tn
from Library.realNoise import czExp_channel
from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate


tn.set_default_backend("pytorch")


class SuperOperator(object):
	def __init__(self, operator: tc.Tensor or tn.AbstractNode = None, noisy: bool = True):
		self.originalINPUT = operator
		self.shape = operator.shape
		self.noisy = noisy
		if noisy is True:
			self.qubitNum = int((len(self.shape) - 1) / 2)
		else:
			self.qubitNum = int(len(self.shape) / 2)
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
		for _i in range(self.qubitNum):
			_TTAxisNames += [f'Dphysics_{_i}', f'Dinner_{_i}']
		for _i in range(self.qubitNum):
			_TTAxisNames += [f'physics_{_i}', f'inner_{_i}']
		return _TTAxisNames

	def getSuperOperator(self) -> tn.AbstractNode:
		_daggerAxisNames = ['D' + _name for _name in self.axisNames if _name != 'I']
		if self.noisy is True:
			_daggerAxisNames += ['I']

		_operatorDagger = tn.Node(self.operator.tensor.conj(), name='realNoiseDagger', axis_names=_daggerAxisNames)
		if self.noisy is True:
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

		_count = 0
		_count_ = 0

		for _i in range(self.qubitNum * 2):
			_leftNames = _TTAxisNames[0:2]
			for _item in _leftNames:
				_TTAxisNames.remove(_item)
			_rightNames = _TTAxisNames
			if _rightNames:
				if 0 < _i <= self.qubitNum - 1:
					_leftNames.append(f'Dbond_{_i - 1}_{_i}')
				elif _i == self.qubitNum:
					_leftNames.append('Ebond')
				elif _i > self.qubitNum:
					_leftNames.append(f'bond_{_count}_{_count + 1}')
					_count += 1
				else:
					pass

				if _i > 0:
					u = _rightNode

				_leftEdges, _rightEdges = [u[_edgeName] for _edgeName in _leftNames], [u[_edgeName] for _edgeName in _rightNames]

				if _i < self.qubitNum - 1:
					_leftNodeName, _rightNodeName = f'DU_{_i}', f'DU_{_i + 1}'
					_edgeName = f'Dbond_{_i}_{_i + 1}'
				elif _i == self.qubitNum - 1:
					_leftNodeName, _rightNodeName = f'DU_{_i}', f'U_{_count_}'
					_edgeName = 'Ebond'
				else:
					_leftNodeName, _rightNodeName = f'U_{_count_}', f'U_{_count_ + 1}'
					_edgeName = f'bond_{_count_}_{_count_ + 1}'
					_count_ += 1

				_leftNode, _rightNode, _ = tn.split_node(node=u, left_edges=_leftEdges, right_edges=_rightEdges,
				                                         left_name=_leftNodeName, right_name=_rightNodeName, edge_name=_edgeName)

				TTSeries.append(_leftNode)
			else:
				TTSeries.append(_rightNode)
		self.superOperatorMPO = TTSeries
		return TTSeries