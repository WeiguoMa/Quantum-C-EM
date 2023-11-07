"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
from copy import deepcopy
from typing import Union

import tensornetwork as tn
import torch as tc

from Library.realNoise import czExp_channel
from Library.tools import EdgeName2AxisName

tn.set_default_backend("pytorch")


class SuperOperator:
	def __init__(self, operator: Union[tc.Tensor, tn.AbstractNode] = None, noisy: bool = True):
		if operator is None:
			operator = czExp_channel('/Users/weiguo_ma/Python_Program/Quantum_error_mitigation'
			                         '/Noisy quantum circuit with MPDO/data/chi/chi1.mat')
		if not isinstance(operator, (tc.Tensor, tn.AbstractNode)):
			raise TypeError("Operator must be a tensor or a node")

		self.inverseTensor = None
		self.shape = operator.shape

		self.noisy = noisy
		self.qubitNum = (len(operator.shape) - 1) // 2 if noisy else len(operator.shape) // 2
		self.axisNames = self._getAxisNames()

		self.operator = self._initialize_operator(operator)
		self.superOperator = self.getSuperOperator()
		self.superOperatorMPO = self.operatorMPO()
		self.inverseSuperOperatorMPO = self.inverseMPO()

	def _getAxisNames(self):
		return [f'{phase}_{i}' for phase in ('physics', 'inner') for i in range(self.qubitNum)] + ['I'] * self.noisy

	def _getTTAxisNames(self):
		return [f'{prefix}_{i}' for i in range(self.qubitNum) for prefix in ('Dphysics', 'Dinner')] \
			+ [f'{prefix}_{i}' for i in range(self.qubitNum) for prefix in ('physics', 'inner')]

	def _getStandardAxisNames(self):
		return [f'{prefix}{i}' for prefix in ('Dphysics_', 'physics_', 'Dinner_', 'inner_') for i in
		        range(self.qubitNum)]

	def _initialize_operator(self, operator):
		if isinstance(operator, tc.Tensor):
			operator_node = tn.Node(operator, name='realNoise', axis_names=self.axisNames)
		else:  # operator is already a tn.AbstractNode
			operator_node = deepcopy(operator)
		if operator_node.axis_names != self.axisNames:
			raise ValueError(f"Operator's axis names must be uniformed as {self.axisNames}")
		return operator_node

	def getSuperOperator(self) -> tn.AbstractNode:
		_daggerAxisNames = ['D' + _name for _name in self.axisNames if _name != 'I'] + ['I'] * self.noisy

		_operatorDagger = tn.Node(self.operator.tensor.conj(), name='realNoiseDagger', axis_names=_daggerAxisNames)
		if self.noisy:
			tn.connect(self.operator['I'], _operatorDagger['I'])

		_superOperatorNode = tn.contract_between(_operatorDagger, self.operator,
		                                         name='superOperatorU', allow_outer_product=True)
		# Process Function
		EdgeName2AxisName(_superOperatorNode)
		_superOperatorNode.reorder_edges([_superOperatorNode[_edge] for _edge in self._getStandardAxisNames()])
		return _superOperatorNode

	def operatorMPO(self):
		_tensor = self.superOperator.tensor
		return self.uMPO(_tensor)

	def inverseMPO(self):
		_tensor = self.superOperator.tensor
		_shape = (2,) * (self.qubitNum * 2 * 2)
		_mShape = (2 ** (2 * self.qubitNum), 2 ** (2 * self.qubitNum))
		_tensor = tc.reshape(tc.inverse(_tensor.reshape(_mShape)), shape=_shape)
		self.inverseTensor = _tensor
		return self.uMPO(_tensor)

	def uMPO(self, _tensor: tc.Tensor) -> list[tn.AbstractNode]:
		# TT-decomposition to superOperatorNode
		u = tn.Node(tensor=_tensor, name='u', axis_names=self._getStandardAxisNames())
		_TTAxisNames = self._getTTAxisNames()

		TTSeries = []
		_leftNode, _rightNode = None, None
		_count, _count_ = 0, 0

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

				_leftEdges, _rightEdges = [u[_edgeName] for _edgeName in _leftNames],\
					[u[_edgeName] for _edgeName in _rightNames]

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
				                                         left_name=_leftNodeName, right_name=_rightNodeName,
				                                         edge_name=_edgeName)
				EdgeName2AxisName([_leftNode, _rightNode])
				TTSeries.append(_leftNode)
			else:
				TTSeries.append(_rightNode)

		return TTSeries
