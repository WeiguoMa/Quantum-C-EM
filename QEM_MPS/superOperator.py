"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import torch as tc
from numpy import prod
import tensornetwork as tn
from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate


class SuperOperator(object):
	def __init__(self, operator: tc.Tensor or tn.AbstractNode = None, noisy: bool = True):
		self.originalINPUT = operator
		self.shape = operator.shape
		self.noisy = noisy
		self.axisNames = self._getAxisNames()

		if operator is None:
			raise NotImplementedError

		if not isinstance(operator, tc.Tensor) and not isinstance(operator, tn.AbstractNode):
			raise TypeError("operator must be a tensor or a node")
		if isinstance(operator, tc.Tensor):
			if self.shape != len(self.axisNames):
				raise ValueError('Shape not match with axisNames, check the noisy status and its corresponds tensor shape.')
			self.operator = tn.Node(operator, name='realNoise', axis_names=self.axisNames)
		if isinstance(operator, tn.Node):
			if operator.axis_names != self.axisNames:
				raise ValueError("operator axis names must be uniformed as {}".format(self.axisNames))
			self.operator = copy.deepcopy(operator)

		self.superOperatorMPO = None
		self.superOperator = None

	def _getAxisNames(self):
		_axisNames = []
		for _i in range(len(self.shape) - 1):
			if _i < len(self.shape) // 2:
				_axisNames.append('physics_{}'.format(_i))
			elif _i < len(self.shape) - 1:
				_axisNames.append('inner_{}'.format(_i))
			else:
				if self.noisy is True:
					_axisNames.append('I')
		return _axisNames

	@staticmethod
	def _reOrderString(_string, _shape):
		_len = len(_string)
		_divider4 = int(_len / 4)
		print(_divider4)
		_pos1 = list(reversed([_element for _element in _string[-_divider4:]]))
		_pos1Shape = prod(_shape[-_divider4:])
		_pos2 = [_element for _element in _string[: _divider4]]
		_pos2Shape = prod(_shape[: _divider4])
		_pos3 = list(reversed([_element for _element in _string[- 2 * _divider4: -_divider4]]))
		_pos3Shape = prod(_shape[- 2 * _divider4: -_divider4])
		_pos4 = [_element for _element in _string[_divider4: 2 * _divider4]]
		_pos4Shape = prod(_shape[_divider4: 2 * _divider4])

		_newString = ''.join(_pos1 + _pos2 + _pos3 + _pos4)
		return _newString, [_pos1Shape, _pos2Shape, _pos3Shape, _pos4Shape]

	def getSuperOperator(self) -> tn.AbstractNode:
		_operatorDagger = tn.Node(self.operator.tensor.conj(), name='realNoiseDagger', axis_names=self.axisNames)
		tn.connect(self.operator['I'], _operatorDagger['I'])
		_superOperatorNode = tn.contract_between(self.operator, _operatorDagger, allow_outer_product=True)
		# Process Function
		EdgeName2AxisName(_superOperatorNode)

		_len = len(_superOperatorNode.axis_names)
		_randomString = generate_random_string_without_duplicate(_len)
		_reorderString, _reorderShape = self._reOrderString(_randomString, _superOperatorNode.tensor.shape)
		_superOperatorTensor \
			= tc.einsum(f'{_randomString} -> {_reorderString}', _superOperatorNode.tensor)\
			.reshape(_reorderShape)
		# Reconstruct superOperatorNode U
		_superOperatorNode = tn.Node(_superOperatorTensor, name='superOperatorU', axis_names=self.axisNames[:-1])
		self.superOperator = _superOperatorNode
		return _superOperatorNode

	def uMPO(self) -> list[tn.AbstractNode]:
		# TT-decomposition to superOperatorNode
		u = self.getSuperOperator()
		TTSeries = []
		_left, _right = None, None
		for _i in range(len(u.shape) // 2):
			_leftEdges = [u[f'physics_{_i}'], u[f'inner_{_i}']]
			if _i > 0:
				_leftEdges.append(u[f'bond{_i}'])
			_rightEdges = [u[f'physics_{_ii}'] for _ii in range(_i + 1, len(u.shape) // 2)] +\
			              [u[f'inner_{_ii}'] for _ii in range(_i + 1, len(u.shape) // 2)]
			if _rightEdges:
				_left, _right, _ = tn.split_node(u, left_edges=_leftEdges, right_edges=_rightEdges,
				                            left_name=f'TTL_{_i}', right_name=f'TTR_{_i}', edge_name=f'bond_{_i}{_i+1}')
				TTSeries.append(_left)
			else:
				TTSeries.append(_right)
		self.superOperatorMPO = TTSeries
		return TTSeries