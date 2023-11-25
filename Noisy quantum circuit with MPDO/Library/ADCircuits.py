"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
from copy import deepcopy
from typing import Optional, Dict

import numpy as np
import tensornetwork as tn
import torch as tc
from torch import nn

from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.AbstractGate import AbstractGate
from Library.NoiseChannel import NoiseChannel
from Library.TNNOptimizer import svd_right2left, qr_left2right, checkConnectivity, svdKappa_left2right, cal_entropy
from Library.realNoise import czExp_channel
from Library.tools import select_device, EdgeName2AxisName, is_nested


class TensorCircuit(nn.Module):
	def __init__(self, qn: int, ideal: bool = True, noiseType: str = 'no', chiFilename: str = None,
	             crossTalk: bool = False,
	             chi: Optional[int] = None, kappa: Optional[int] = None, tnn_optimize: bool = True,
	             chip: Optional[str] = None, device: str or int = None, _entropy: bool = False):
		"""
		Args:
			ideal: Whether the circuit is ideal.  --> Whether the one-qubit gate is ideal or not.
			noiseType: The type of the noise channel.   --> Which kind of noise channel is added to the two-qubit gate.
			chiFilename: The filename of the chi matrix.
			chi: The chi parameter of the TNN.
			kappa: The kappa parameter of the TNN.
			tnn_optimize: Whether the TNN is optimized.
			chip: The name of the chip.
			device: The device of the torch;
			_entropy: Whether to record the system entropy.
		"""
		super(TensorCircuit, self).__init__()
		# Paras. About Torch
		self.device = select_device(device)
		self.dtype = tc.complex64
		self.layers = nn.Sequential()
		self._oqs_list = []

		# About program
		self.fVR = None
		self.i = 0
		self.qnumber = qn
		self.state = None
		self.initState = None

		# Entropy
		self._entropy = _entropy
		self._entropyList = {f'qEntropy_{_i}': [] for _i in range(self.qnumber)}

		# Noisy Circuit Setting
		self.ideal = ideal
		self.realNoise = False
		self.idealNoise = False
		self.unified = None
		self.crossTalk = crossTalk

		if not self.ideal:
			self.Noise = NoiseChannel(chip=chip, device=self.device)
			if noiseType == 'unified':
				self.unified = True
			elif noiseType == 'realNoise':
				self.realNoise, self.crossTalk = True, False
				self.realNoiseChannelTensor = czExp_channel(filename=chiFilename, device=self.device)
			elif noiseType == 'idealNoise':
				self.idealNoise = True
			else:
				raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

		# Density Matrix
		self.DM = None
		self.DMNode = None

		# TNN Truncation
		self.chi = chi
		self.kappa = kappa
		self.tnn_optimize = tnn_optimize

	@staticmethod
	def _calOrder(minIdx: int, maxIdx: int = None, lBond: bool = False,
	              smallI: bool = False, bigI: bool = False, rBond: bool = False, single: bool = False):
		if single:
			_return = [f'bond_{minIdx - 1}_{minIdx}'] * lBond \
			          + [f'physics_{minIdx}'] \
			          + [f'I_{minIdx}'] * smallI + [f'bond_{minIdx}_{minIdx + 1}'] * rBond
		else:
			_return = [f'bond_{minIdx - 1}_{minIdx}'] * lBond \
			          + [f'physics_{minIdx}', f'physics_{maxIdx}'] \
			          + [f'I_{minIdx}'] * smallI + [f'I_{maxIdx}'] * bigI \
			          + [f'bond_{maxIdx}_{maxIdx + 1}'] * rBond
		return _return

	def _transpile_gate(self, _gate_: AbstractGate, _oqs_: list[int]):
		"""
	    Transpile a quantum gate into a sequence of gates and operating qubits.

	    Args:
	        _gate_: AbstractGate to be transpiled.
	        _oqs_: Operating qubits.

	    Returns:
	        A tuple of transpiled gate list and operating qubits list.
	    """
		_gateName_ = _gate_.name.lower()
		_gateTrunc = _gate_._lastTruncation

		if _gateName_ == 'cnot' or _gateName_ == 'cx':
			_gateList_ = [AbstractGate(ideal=True).ry(-np.pi / 2),
			              AbstractGate(_lastTrunc=_gateTrunc).czEXP(EXPTensor=self.realNoiseChannelTensor),
			              AbstractGate(ideal=True).ry(np.pi / 2)]
			_oqsList_ = [[_oqs_[-1]], _oqs_, [_oqs_[-1]]]
		elif _gateName_ == 'rzz':
			_gateList_ = [AbstractGate(_lastTrunc=_gateTrunc).cnot(),
			              AbstractGate(ideal=True).rz(_gate_.para),
			              AbstractGate(_lastTrunc=_gateTrunc).cnot()]
			_oqsList_ = [_oqs_, [_oqs_[-1]], _oqs_]
		elif _gateName_ == 'rxx':
			_gateList_ = [AbstractGate(ideal=True).h(), AbstractGate(_lastTrunc=_gateTrunc).cnot(),
			              AbstractGate(ideal=True).rz(_gate_.para),
			              AbstractGate(_lastTrunc=_gateTrunc).cnot(), AbstractGate(ideal=True).h()]
			_oqsList_ = [_oqs_, _oqs_, [_oqs_[-1]], _oqs_, _oqs_]
		elif _gateName_ == 'ryy':
			_gateList_ = [AbstractGate(ideal=True).rx(np.pi / 2), AbstractGate(_lastTrunc=_gateTrunc).cnot(),
			              AbstractGate(ideal=True).rz(_gate_.para), AbstractGate(_lastTrunc=_gateTrunc).cnot(),
			              AbstractGate(ideal=True).rx(-np.pi / 2)]
			_oqsList_ = [_oqs_, _oqs_, [_oqs_[-1]], _oqs_, _oqs_]
		else:
			_gateList_, _oqsList_ = [_gate_], [_oqs_]
		return _gateList_, _oqsList_

	def _crossTalkZ_transpile(self, _gate_: AbstractGate, _oqs_: list[int]):
		_minOqs, _maxOqs = min(_oqs_), max(_oqs_)
		if _minOqs == _maxOqs:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 2))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), _gate_, AbstractGate(ideal=True).rz(_Angle[0][1])]
			_oqsList_ = [[_oqs_[0] - 1], _oqs_, [_oqs_[0] + 1]]
		elif _minOqs + 1 == _maxOqs:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 2))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), _gate_, AbstractGate(ideal=True).rz(_Angle[0][1])]
			_oqsList_ = [[_minOqs - 1], _oqs_, [_maxOqs + 1]]
		else:
			_Angle = np.random.normal(loc=np.pi / 16, scale=np.pi / 128,
			                          size=(1, 4))  # Should be related to the chip-Exp information
			_gateList_ = [AbstractGate(ideal=True).rz(_Angle[0][0]), AbstractGate(ideal=True).rz(_Angle[0][1]),
			              _gate_, AbstractGate(ideal=True).rz(_Angle[0][2]), AbstractGate(ideal=True).rz(_Angle[0][3])]
			_oqsList_ = [[_minOqs - 1], [_minOqs + 1], _oqs_, [_maxOqs - 1], [_maxOqs + 1]]
		if _oqsList_[0][0] < 0:
			_gateList_.pop(0), _oqsList_.pop(0)
		if _oqsList_[-1][0] > self.qnumber - 1:
			_gateList_.pop(-1), _oqsList_.pop(-1)
		return _gateList_, _oqsList_

	def _add_gate(self, _qubits: list[tn.Node] or list[tn.AbstractNode],
	              _layer_num: int, _oqs: list[int]):
		r"""
		Add quantum Gate to tensornetwork

		Args:
			_oqs: operating qubits.

		Returns:
			Tensornetwork after adding gate.
		"""

		gate = self.layers[_layer_num].gate
		_maxIdx, _minIdx = max(_oqs), min(_oqs)

		if not isinstance(_qubits, list):
			raise TypeError('Qubit must be a list of nodes.')
		if not isinstance(gate, (TensorGate, NoisyTensorGate)):
			raise TypeError(f'Gate must be a TensorGate, current type is {type(gate)}.')
		if not isinstance(_oqs, list):
			raise TypeError('Operating qubits must be a list.')
		if _maxIdx >= self.qnumber:
			raise ValueError('Qubit index out of range.')

		single = gate.single
		gate.tensor = gate.tensor.to(device=self.device)

		if single is False:  # Two-qubit gate
			"""
						   | i  | j                         | w              | p
						___|____|___                    ____|____        ____|____
						|          |                    |       |        |       |
						|    DG    |---- q        l ----| Qubit |--------| Qubit |---- r
						|__________|                    |_______|    k   |_______|
						   |    |                           |                |
						   | w  | p                         | m              | n
			"""
			if len(_oqs) != 2 or _minIdx == _maxIdx:
				raise ValueError('Invalid operating qubits for a two-qubit gate.')
			if is_nested(_oqs):
				raise NotImplementedError('Series CNOT gates are not supported yet.')

			_gateAxisNames = [f'physics_{_minIdx}', f'physics_{_maxIdx}', f'inner_{_minIdx}', f'inner_{_maxIdx}']
			if self.realNoise or self.idealNoise:
				if not self.realNoise:
					# Adding depolarization noise channel to Two-qubit gate
					gate.tensor = tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, gate.tensor)
				if (gate.tensor.shape[-1] != 1 or len(gate.tensor.shape) != 4) and not gate.ideal:
					_gateAxisNames.append(f'I_{_minIdx}')

			gate = tn.Node(gate.tensor.squeeze(), name=gate.name, axis_names=_gateAxisNames)

			# Created a new node in memory
			_contract_qubits = tn.contract_between(_qubits[_minIdx], _qubits[_maxIdx],
			                                       name=f'q_{_oqs[0]}_{_oqs[1]}',
			                                       allow_outer_product=True)
			# ProcessFunction, for details, see the function definition.
			EdgeName2AxisName([_contract_qubits])

			_contract_qubitsAxisName = _contract_qubits.axis_names

			_gString = 'ijwpq' if len(gate.tensor.shape) == 5 else 'ijwp'
			if _oqs[0] > _oqs[1]:
				_gString = _gString.replace('wp', 'pw')

			_smallI, _bigI, _lBond, _rBond = f'I_{min(_oqs)}' in _contract_qubitsAxisName, \
			                                 f'I_{max(_oqs)}' in _contract_qubitsAxisName, \
			                                 f'bond_{_minIdx - 1}_{_minIdx}' in _contract_qubitsAxisName, \
			                                 f'bond_{_maxIdx}_{_maxIdx + 1}' in _contract_qubitsAxisName

			_qString = ''.join(['l' * _lBond, 'wp', 'm' * _smallI, 'n' * _bigI, 'r' * _rBond])

			_qAFString = _qString.replace('wp', _gString[:2] + _gString[-1] * (len(gate.tensor.shape) == 5))
			if _oqs[0] > _oqs[1]:
				_qAFString = _qAFString.replace('ij', 'ji')

			_reorderAxisName = self._calOrder(_minIdx, _maxIdx, _lBond, _smallI, _bigI, _rBond)

			_contract_qubits.reorder_edges([_contract_qubits[_element] for _element in _reorderAxisName])
			_contract_qubitsTensor_AoP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}',
			                                       gate.tensor, _contract_qubits.tensor)
			_qShape = _contract_qubitsTensor_AoP.shape

			_mIdx, _qIdx = _qAFString.find('m'), _qAFString.find('n')
			if _mIdx != -1:  # qubit[min] is Noisy before this gate-add operation
				_qShape = _qShape[:_mIdx - 1] + (_qShape[_mIdx - 1] * _qShape[_mIdx],) + _qShape[_mIdx + 1:]
				_contract_qubits.set_tensor(tc.reshape(_contract_qubitsTensor_AoP, _qShape))
			else:
				if 'q' in _gString:
					_AFStringT = f'{_qAFString.replace("q", "")}{"q"}'
					_contract_qubits.set_tensor(
						tc.einsum(f"{_qAFString} -> {_AFStringT}", _contract_qubitsTensor_AoP))

					_contract_qubitsAxisName.append(f'I_{_minIdx}')
					_contract_qubits.add_axis_names(_contract_qubitsAxisName)
					_new_edge = tn.Edge(_contract_qubits,
					                    axis1=len(_contract_qubits.edges), name=f'I_{_minIdx}')
					_contract_qubits.edges.append(_new_edge)
					_contract_qubits.add_edge(_new_edge, f'I_{_minIdx}')

					_smallI = True
					_reorderAxisName = self._calOrder(_minIdx, _maxIdx, _lBond, _smallI, _bigI, _rBond)
					_contract_qubits.reorder_edges([_contract_qubits[_element] for _element in _reorderAxisName])
				else:
					_contract_qubits.set_tensor(_contract_qubitsTensor_AoP)

			# Split back to two qubits
			_left_AxisName = ['bond_{}_{}'.format(_minIdx - 1, _minIdx)] * _lBond + [f'physics_{_minIdx}'] +\
			                 [f'I_{_minIdx}'] * (self.idealNoise or self.realNoise)
			_right_AxisName = [f'physics_{_maxIdx}']\
			                  + [f'I_{_maxIdx}'] * ((self.idealNoise or self.realNoise) and _bigI)\
			                  + ['bond_{}_{}'.format(_maxIdx, _maxIdx + 1)] * _rBond

			_left_edges, _right_edges = [_contract_qubits[name] for name in _left_AxisName], \
				[_contract_qubits[name] for name in _right_AxisName]

			_qubits[_minIdx], _qubits[_maxIdx], _ = tn.split_node(_contract_qubits,
			                                                      left_edges=_left_edges,
			                                                      right_edges=_right_edges,
			                                                      left_name=f'qubit_{_minIdx}',
			                                                      right_name=f'qubit_{_maxIdx}',
			                                                      edge_name=f'bond_{_minIdx}_{_maxIdx}')
			EdgeName2AxisName([_qubits[_minIdx], _qubits[_maxIdx]])
		else:
			"""
							| m                             | i
						____|___                        ____|____
						|      |                        |       |   
						|  SG  |---- n            l ----| Qubit |---- r
						|______|                        |_______|
							|                               |
							| i                             | j
			"""
			gate_list = [
				tn.Node(
					tc.reshape(
						tc.einsum('nlm, ljk, ji -> nimk', self.Noise.dpCTensor, self.Noise.apdeCTensor, gate.tensor),
						(2, 2, -1))
					if (
							   self.idealNoise or self.unified) and not gate.ideal
					else gate.tensor,
					name=gate.name,
					axis_names=[f'physics_{_idx}', f'inner_{_idx}',
					            f'I_{_idx}'] if self.idealNoise or self.unified else [
						f'physics_{_idx}', f'inner_{_idx}']
				)
				for _idx in _oqs
			]

			for _i, _bit in enumerate(_oqs):
				_qubit = _qubits[_bit]
				_qubitTensor, _qubitAxisName, _qubitShape = \
					_qubit.tensor, _qubit.axis_names, _qubit.tensor.shape
				_gShape = gate_list[_i].shape

				_I, _lBond, _rBond = \
					f'I_{_bit}' in _qubitAxisName, f'bond_{_bit-1}_{_bit}' in _qubitAxisName,\
						f'bond_{_bit}_{_bit+1}' in _qubitAxisName
				_qString = ''.join(['l' * _lBond, 'i', 'j' * _I, 'r' * _rBond])

				_gString = 'min' if len(gate_list[_i].shape) == 3 else 'mi'

				_qAFString = _qString.replace('i', 'mn') if _gString == 'min' else _qString.replace('i', 'm')

				_reorderAxisName = self._calOrder(minIdx=_bit, lBond=_lBond, smallI=_I, rBond=_rBond, single=True)
				if _qubitAxisName != _reorderAxisName:
					_qubit.reorder_edges([_qubit[_element] for _element in _reorderAxisName])

				_qubitTensor_AOP = tc.einsum(f'{_gString}, {_qString} -> {_qAFString}', gate_list[_i].tensor,
				                             _qubitTensor)
				_qShape = _qubitTensor_AOP.shape

				_jIdx, _nIdx = _qAFString.find('j'), _qAFString.find('n')
				if _jIdx != -1:     # Exist j -> Noisy Qubit
					if _nIdx != -1:
						_qShape = _qShape[:_jIdx - 1] + (_qShape[_jIdx - 1] * _qShape[_jIdx],) + _qShape[_jIdx + 1:]
						_qubit.set_tensor(tc.reshape(_qubitTensor_AOP, _qShape))
				else:
					if 'n' in _gString:
						_AFStringT = f'{_qAFString.replace("n", "")}{"n"}'
						_qubit.set_tensor(
							tc.einsum(f"{_qAFString} -> {_AFStringT}", _qubitTensor_AOP))

						_qubitAxisName.insert(_qubitAxisName.index(f'physics_{_bit}') + 1, f'I_{_bit}')
						_qubit.add_axis_names(_qubitAxisName)
						_new_edge = tn.Edge(_qubit, axis1=len(_qubit.edges), name=f'I_{_bit}')
						_qubit.edges.insert(_qubitAxisName.index(f'I_{_bit}'), _new_edge)
						_qubit.add_edge(_new_edge, f'I_{_bit}')

						_I = True
						_reorderAxisName = self._calOrder(minIdx=_bit, lBond=_lBond, smallI=_I, rBond=_rBond,
						                                  single=True)
						_qubit.reorder_edges([_qubit[_element] for _element in _reorderAxisName])
					else:   # Ideal Gate and Qubit
						_qubit.set_tensor(_qubitTensor_AOP)

	def add_gate(self, gate: AbstractGate, oqs: list):
		r"""
		Add quantum gate to circuit layer by layer.

		Args:
			gate: gate to be added;
			oqs: operating qubits.

		Returns:
			None
		"""
		if not isinstance(gate, AbstractGate):
			raise TypeError('Gate must be a AbstractGate.')

		oqs = [oqs] if isinstance(oqs, int) else oqs
		if not isinstance(oqs, list):
			raise TypeError('Operating qubits must be a list.')

		if self.realNoise:
			_transpile_gateList, _transpile_oqsList = self._transpile_gate(gate, oqs)
		elif self.crossTalk:
			_transpile_gateList, _transpile_oqsList = self._crossTalkZ_transpile(gate, oqs)
		else:
			_transpile_gateList, _transpile_oqsList = [gate], [oqs]

		for _num, _gate in enumerate(_transpile_gateList):
			if _gate.para is None:
				_para = None
			elif type(_gate.para) is tc.Tensor and _gate.para.shape != ():
				_para = None
			else:
				_para = _gate.para
				if type(_para) is float:
					_para = tc.tensor(_para)
				_paraI, _paraD = '{:.3f}'.format(_para.squeeze()).split('.')
				_para = f'{_paraI}|{_paraD} pi'

			self.layers.add_module(f'{_gate.name}{_transpile_oqsList[_num]}({_para})-G{self.i}', _gate)
			self._oqs_list.append(_transpile_oqsList[_num])
		self.i += 1

	def _calculate_DM(self, state_vector: bool = False,
	                  reduced_index: Optional[list[list[int] or int]] = None) -> tc.Tensor:
		r"""
		Calculate the density matrix of the state.

		Args:
			state_vector: if True, the state is a state vector, otherwise, the state is a density matrix.
			reduced_index: the state[index] to be reduced, which means the physics_con-physics of sub-density matrix
								will be connected and contracted.

		Returns:
			_dm: the density matrix node;
			_dm_tensor: the density matrix tensor.

		Additional information:
			A mixed state(noisy situation) is generally cannot be efficiently represented by a state vector but DM.

		Raises:
			ValueError: if the state is chosen to be a vector but is noisy;
			ValueError: Reduced index should be empty as [] or None.
		"""
		if not reduced_index:
			reduced_index = None

		if reduced_index and max(reduced_index) >= self.qnumber:
			raise ValueError('Reduced index should not be larger than the qubit number.')

		if not state_vector:
			_qubits_conj = deepcopy(self.state)  # Node.copy() func. cannot work correctly
			for _qubit in _qubits_conj:
				_qubit.set_tensor(_qubit.tensor.conj())

			# Differential name the conjugate qubits' edges name to permute the order of the indices
			for _i, _qubit_conj in enumerate(_qubits_conj):
				_qubit_conj.set_name(f'con_{_qubit_conj.name}')
				for _ii, _edge in enumerate(_qubit_conj.edges):
					if 'physics' in _edge.name:
						_edge.set_name(f'con_{_qubit_conj[_ii].name}')
						_qubit_conj.axis_names[_ii] = f'con_{_qubit_conj.axis_names[_ii]}'

			for i in range(len(self.state)):
				if not self.ideal and f'I_{i}' in self.state[i].axis_names:
					tn.connect(self.state[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])

			# Reduced density matrix
			if reduced_index is not None:
				reduced_index = [reduced_index] if isinstance(reduced_index, int) else reduced_index
				if not isinstance(reduced_index, list):
					raise TypeError('reduced_index should be int or list[int]')
				for _idx in reduced_index:
					tn.connect(self.state[_idx][f'physics_{_idx}'], _qubits_conj[_idx][f'con_physics_{_idx}'])
			else:
				reduced_index = []

			_numList = [_i for _i in range(self.qnumber) if _i not in reduced_index]
			_qOutOrder, _conQOutOrder = [self.state[i][f'physics_{i}'] for i in _numList], \
				[_qubits_conj[i][f'con_physics_{i}'] for i in _numList]

			_dm = tn.contractors.auto(self.state + _qubits_conj,
			                            output_edge_order=_qOutOrder + _conQOutOrder)

			_reshape_size = self.qnumber - len(reduced_index)

			self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
			return self.DM
		else:
			if not self.fVR and not self.ideal:
				raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
			if reduced_index is not None:
				raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

			_outOrder = [self.state[i][f'physics_{i}'] for i in list(range(self.qnumber))]
			_vector = tn.contractors.auto(self.state, output_edge_order=_outOrder)

			self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1)) if not self.fVR else _vector.tensor
			return self.DM

	def forward(self, _state: list[tn.Node] = None,
	            state_vector: bool = False, reduced_index: list = None, forceVectorRequire: bool = False) -> (tc.Tensor, Dict):
		r"""
		Forward propagation of tensornetwork.

		Returns:
			self.state: tensornetwork after forward propagation.
		"""
		self.initState, self.qnumber, self.fVR = _state, len(_state), forceVectorRequire

		for _i, layer in enumerate(self.layers):
			self._add_gate(_state, _i, _oqs=self._oqs_list[_i])
			if layer._lastTruncation and self.tnn_optimize:
				if not self.ideal:
					svdKappa_left2right(_state, kappa=self.kappa)
				if checkConnectivity(_state):
					qr_left2right(_state)
					svd_right2left(_state, chi=self.chi)
			if self._entropy:
				_entropy = cal_entropy(_state, kappa=self.kappa)
				for _oqs in self._oqs_list[_i]:
					self._entropyList[f'qEntropy_{_oqs}'].append(_entropy[f'qEntropy_{_oqs}'])
		# LastLayer noise-truncation
		if self.tnn_optimize and not self.ideal:
			svdKappa_left2right(_state, kappa=self.kappa)

		self.state = _state
		_dm = self._calculate_DM(state_vector=state_vector, reduced_index=reduced_index)

		return _dm
