"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import numpy as np
import tensornetwork as tn
import torch as tc
from torch import nn
from typing import Optional

import Library.tools as tools
from Library.ADGate import TensorGate
from Library.ADNGate import NoisyTensorGate
from Library.AbstractGate import AbstractGate
from Library.NoiseChannel import NoiseChannel
from Library.realNoise import czExp_channel
from Library.TNNOptimizer import svd_right2left, qr_left2right, checkConnectivity

class TensorCircuit(nn.Module):
	def __init__(self, qn: int, ideal: bool = True, noiseType: str = 'no', chiFilename: str = None, crossTalk: bool = False,
				chi: Optional[int] = None, kappa: Optional[int] = None, tnn_optimize: bool = True,
				chip: Optional[str] = None, device: str or int = 'cpu'):
		"""
		Args:
			ideal: Whether the circuit is ideal.  --> Whether the one-qubit gate is ideal or not.
			noiseType: The type of the noise channel.   --> Which kind of noise channel is added to the two-qubit gate.
			chiFilename: The filename of the chi matrix.
			chi: The chi parameter of the TNN.
			kappa: The kappa parameter of the TNN.
			tnn_optimize: Whether the TNN is optimized.
			chip: The name of the chip.
			device: The device of the torch.
		"""
		super(TensorCircuit, self).__init__()
		self.fVR = None
		self.i = 0
		self.qnumber = qn
		self.state = None
		self.initState = None
		self.chip = chip

		# Noisy Circuit Setting
		self.ideal = ideal
		self.realNoise = False
		self.idealNoise = False
		self.unified = None
		self.crossTalk = crossTalk
		if self.ideal is False:
			self.Noise = NoiseChannel(chip=self.chip)
			if noiseType == 'unified':
				self.unified = True
			elif noiseType == 'realNoise':
				self.realNoise = True
				self.crossTalk = False
				# self.realNoise = realNoise
				self.realNoiseChannelTensor = None
				self.realNoiseChannelTensor = czExp_channel(filename=chiFilename)
				#
			elif noiseType == 'idealNoise':
				self.idealNoise = True
			else:
				raise TypeError(f'Noise type "{noiseType}" is not supported yet.')

		# Paras. About Torch
		self.device = tools.select_device(device)
		self.dtype = tc.complex128
		self.layers = nn.Sequential()
		self._oqs_list = []

		# Density Matrix
		self.DM = None
		self.DMNode = None

		# TNN Truncation
		self.chi = chi
		self.kappa = kappa
		self.tnn_optimize = tnn_optimize

	def _transpile_gate(self, _gate_: AbstractGate, _oqs_: list[int]):
		_gateName_ = _gate_.name.lower()
		if _gateName_ == 'cnot' or _gateName_ == 'cx':
			_gateList_ = [AbstractGate(ideal=True).ry(-np.pi / 2),
						  AbstractGate().czEXP(EXPTensor=self.realNoiseChannelTensor),
														 AbstractGate(ideal=True).ry(np.pi / 2)]
			_oqsList_ = [[_oqs_[-1]], _oqs_, [_oqs_[-1]]]
		elif _gateName_ == 'rzz':
			_gateList_ = [AbstractGate().cnot(), AbstractGate(ideal=True).rz(_gate_.para), AbstractGate().cnot()]
			_oqsList_ = [_oqs_, [_oqs_[-1]], _oqs_]
		elif _gateName_ == 'rxx':
			_gateList_ = [AbstractGate(ideal=True).h(), AbstractGate().cnot(), AbstractGate(ideal=True).rz(_gate_.para),
						  AbstractGate().cnot(), AbstractGate(ideal=True).h()]
			_oqsList_ = [_oqs_, _oqs_, [_oqs_[-1]], _oqs_, _oqs_]
		elif _gateName_ == 'ryy':
			_gateList_ = [AbstractGate(ideal=True).rx(np.pi / 2), AbstractGate().cnot(),
						  AbstractGate(ideal=True).rz(_gate_.para), AbstractGate().cnot(),
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

		if isinstance(_qubits, list) is False:
			raise TypeError('Qubit must be a list of nodes.')
		if isinstance(gate, TensorGate) is False and isinstance(gate, NoisyTensorGate) is False:
			raise TypeError(f'Gate must be a TensorGate, current type is {type(gate)}.')
		if isinstance(_oqs, list) is False:
			raise TypeError('Operating qubits must be a list.')
		if _oqs[np.argmax(_oqs)] > self.qnumber - 1:
			raise ValueError('Qubit index out of range.')

		single = gate.single
		if single is False:  # Two-qubit gate
			if len(_oqs) > 2:
				raise NotImplementedError('Only two-qubit gates are supported currently.')
			if _oqs[0] == _oqs[1]:
				raise ValueError(f'Operating qubits must be different, current qubits {_oqs[0]} -- {_oqs[1]}.')
			if tools.is_nested(_oqs) is True:
				raise NotImplementedError('Series CNOT gates are not supported yet.')
			_edges = []
			if self.realNoise is False and self.idealNoise is False:
				gate = tn.Node(gate.tensor, name=gate.name,
							   axis_names=[f'physics_{_oqs[0]}', f'physics_{_oqs[1]}',
										   f'inner_{_oqs[0]}', f'inner_{_oqs[1]}'])
			else:
				if self.idealNoise is True and self.realNoise is False:
					gate.tensor = tc.einsum('ijklp, klmn -> ijmnp', self.Noise.dpCTensor2, gate.tensor)

				if gate.tensor.shape[-1] == 1:
					gate = tn.Node(gate.tensor.squeeze(), name=gate.name,
								   axis_names=[f'physics_{_oqs[0]}', f'physics_{_oqs[1]}',
											   f'inner_{_oqs[0]}', f'inner_{_oqs[1]}'])
				else:
					gate = tn.Node(gate.tensor, name=gate.name,
							   axis_names=[f'physics_{_oqs[0]}', f'physics_{_oqs[1]}',
										   f'inner_{_oqs[0]}', f'inner_{_oqs[1]}', f'I_{_oqs[1]}'])

			# Created a new node in memory
			_contract_qubits = tn.contract_between(_qubits[_oqs[0]], _qubits[_oqs[1]],
												   name=f'q_{_oqs[0]}_{_oqs[1]}',
												   allow_outer_product=True)
			# ProcessFunction, for details, see the function definition.
			tools.EdgeName2AxisName([_contract_qubits])
			for _i, _bit in enumerate(_oqs):
				_edges.append(tn.connect(_contract_qubits['physics_{}'.format(_bit)], gate[f'inner_{_bit}']))

			# contract the connected edge and inherit the name of the pre-qubit
			gate = tn.contract_between(_contract_qubits, gate, name=gate.name)

			# ProcessFunction, for details, see the function definition.
			tools.EdgeName2AxisName([gate])

			if self.realNoise is True:
				_dup_item, _dup_idx = tools.find_duplicate(gate.axis_names)
				if _dup_item:
					# Number of axis name before the reshape operation(contain duplicates)
					_length = len(gate.axis_names)
					# Find the shape of the tensor after the reshape operation
					_reshape_shape = list(gate.tensor.shape)
					_reshape_shape[_dup_idx[1]] = _reshape_shape[_dup_idx[0]] * _reshape_shape[_dup_idx[1]]
					_reshape_shape.pop(_dup_idx[0])
					# Generate a random string without duplicates, if len = 4, then the string is 'abcd' as Einstein notation.
					_random_string = tools.generate_random_string_without_duplicate(_length)
					_random_string_reorder = tools.move_index(_random_string, _dup_idx[0],
															  _dup_idx[1] - 1)  # Like 'ifcvbj' -> 'fcvbij'
					# Reshape the tensor
					_reshaped_tensor = tc.einsum(_random_string + ' -> ' + _random_string_reorder,
												 gate.tensor).reshape(_reshape_shape)
					_axis_names = copy.deepcopy(gate.axis_names)
					_axis_names.pop(_dup_idx[0])

					""" Reform the qubit from adding noise channel, which causes edges' error in tensornetwork,
							easily speaking, reform the tensor. TensorNetwork package is not perfect. """
					_bond_list, _l_name, _r_name = [], None, None
					for _name in gate.axis_names:
						if 'bond' in _name:
							_bond_list.append(_name)
							for _string_lst in _bond_list:
								_string_lst = _name.split('_')
								if int(_string_lst[-1]) == int(_oqs[0]):
									_l_name = _name
								elif int(_string_lst[-2]) == int(_oqs[1]):
									_r_name = _name

					if _l_name is not None:
						_left_qubit, _reforming_qubit = gate[_l_name].get_nodes()
						gate[_l_name].disconnect(_l_name, 'middle2left_edge')
					else:
						_left_qubit = None
					if _r_name is not None:
						_reforming_qubit, _right_qubit = gate[_r_name].get_nodes()
						gate[_r_name].disconnect('middle2right_edge', _r_name)
					else:
						_right_qubit = None

					# Previous information of _qubits[_qnum] is extracted, now we remade a new _qubits[_qnum]
					gate = tn.Node(_reshaped_tensor,
											 name=f'q_{_oqs[0]}_{_oqs[1]}',
											 axis_names=_axis_names)  # Node's dimension of Edge f'I_{_qnum}' has been promoted.

					if _l_name is not None:
						tn.connect(_left_qubit[_l_name], gate[_l_name], name=_l_name)
					else:
						pass
					if _r_name is not None:
						tn.connect(gate[_r_name], _right_qubit[_r_name], name=_r_name)
					else:
						pass
					# ProcessFunction, for details, see the function definition
					tools.EdgeName2AxisName([gate])

			# Split back to two qubits
			_left_AxisName, _right_AxisName = _qubits[_oqs[0]].axis_names, _qubits[_oqs[1]].axis_names
			if 'bond_{}_{}'.format(_oqs[0], _oqs[1]) in _left_AxisName:
				# remove the bond_name between two qubits, which take appearance in both qubits simultaneously
				_left_AxisName.remove('bond_{}_{}'.format(_oqs[0], _oqs[1]))
				_right_AxisName.remove('bond_{}_{}'.format(_oqs[0], _oqs[1]))
			elif 'bond_{}_{}'.format(_oqs[1], _oqs[0]) in _left_AxisName:
				# remove the bond_name between two qubits, which take appearance in both qubits simultaneously
				_left_AxisName.remove('bond_{}_{}'.format(_oqs[1], _oqs[0]))
				_right_AxisName.remove('bond_{}_{}'.format(_oqs[1], _oqs[0]))

			# Cases that the EXPNoise is added as the first layer
			if self.realNoise is True or self.idealNoise is True:
				if f'I_{_oqs[0]}' in gate.axis_names and f'I_{_oqs[0]}' not in _left_AxisName \
						and f'I_{_oqs[0]}' not in _right_AxisName:
					_right_AxisName.append(f'I_{_oqs[1]}')
				elif f'I_{_oqs[1]}' in gate.axis_names and f'I_{_oqs[1]}' not in _left_AxisName \
						and f'I_{_oqs[1]}' not in _right_AxisName:
					_right_AxisName.append(f'I_{_oqs[1]}')

			_left_edges, _right_edges = [gate[name] for name in _left_AxisName], \
										[gate[name] for name in _right_AxisName]

			_qubits[_oqs[0]], _qubits[_oqs[1]], _ = tn.split_node(gate,
																  left_edges=_left_edges,
																  right_edges=_right_edges,
																  left_name=f'qubit_{_oqs[0]}',
																  right_name=f'qubit_{_oqs[1]}',
																  edge_name=f'bond_{_oqs[0]}_{_oqs[1]}')
			tools.EdgeName2AxisName([_qubits[_oqs[0]], _qubits[_oqs[1]]])

			if self.realNoise is True and f'I_{_oqs[1]}' in _qubits[_oqs[1]].axis_names:
				# Shape-relating
				_shape = _qubits[_oqs[1]].tensor.shape
				_left_edge_shape = [_shape[_ii_] for _ii_ in range(len(_shape) - 1)]
				_left_dim = int(np.prod(_left_edge_shape))
				# SVD to truncate the inner dimension
				_u, _s, _ = tc.linalg.svd(tc.reshape(_qubits[_oqs[1]].tensor, (_left_dim, _shape[-1])),
										  full_matrices=False)
				_s = _s.to(dtype=tc.complex128)

				# Truncate the inner dimension
				if self.kappa is None:
					_kappa = _s.nelement()
				else:
					_kappa = self.kappa

				_s = _s[: _kappa]
				_u = _u[:, : _kappa]

				if len(_s.shape) == 1:
					_s = tc.diag(_s)

				# Back to the former shape
				_left_edge_shape.append(_s.shape[-1])
				_qubits[_oqs[1]].tensor = tc.reshape(tc.matmul(_u, _s), _left_edge_shape)
		else:
			gate_list = [tn.Node(gate.tensor, name=gate.name, axis_names=[f'physics_{_idx}', f'inner_{_idx}'])
						 for _idx in _oqs]
			for _i, _bit in enumerate(_oqs):
				tn.connect(_qubits[_bit][f'physics_{_bit}'], gate_list[_i][f'inner_{_bit}'])
				# contract the connected edge and inherit the name of the pre-qubit
				_qubits[_bit] = tn.contract_between(_qubits[_bit], gate_list[_i], name=_qubits[_bit].name)
				# ProcessFunction, for details, see the function definition.
				tools.EdgeName2AxisName([_qubits[_bit]])

	def _add_noise(self, _qubits: list[tn.Node] or list[tn.AbstractNode],
							oqs: list[int] or int, reverse: Optional[bool],
							noiseInfo: NoiseChannel,
							noise_type: str):
		r"""
		Apply the noise channel to the qubits.

		Args:
			_qubits: The qubits to be applied the noise channel;
			oqs: The qubits to be applied the noise channel;
			reverse: Whether the control-target positions are reversed;
			noiseInfo: The information of the noise channel;
			noise_type: The type of the noise channel.

		Returns:
			_qubits: The qubits after applying the noise channel.
				!!! Actually no return, but the qubits are changed in the memory. !!!

		Additional information:
			The noise channel is applied to the qubits by the following steps:
				1. Construct the noise channel;
				2. Construct the error tensor;
				3. Contract the error tensor with the qubits;
				4. Fix the axis format of the qubits.
		"""

		def _import_error_tensor(_noise_type_: str):
			if _noise_type_ == 'depolarization':
				return noiseInfo.dpCTensor
			elif _noise_type_ == 'amplitude_phase_damping_error':
				return noiseInfo.apdeCTensor
			else:
				raise NotImplementedError(f'Noise type {_noise_type_} is not implemented yet.')

		if not isinstance(_qubits, list):
			if not isinstance(_qubits, tn.Node) or not isinstance(_qubits, tn.AbstractNode):
				raise TypeError(f'qubits must be a list of tn.Node or tn.AbstractNode, but got {type(_qubits)}')
			_qubits = [_qubits]
		if not isinstance(oqs, list):
			if not isinstance(oqs, int):
				raise TypeError(f'oqs must be a list of int, but got {type(oqs)}')
			oqs = [oqs]
		if len(oqs) > len(_qubits):
			raise ValueError(f'len(oqs) must be less than or equal to to len(qubits),'
							 f' but got {len(oqs)} and {len(_qubits)}')

		# Create Node for noise channel
		_noise_tensor = _import_error_tensor(_noise_type_=noise_type)

		_noise_nodeList = []
		for _oq in oqs:
			_noise_node = tn.Node(_noise_tensor, name='noise_node',
								  axis_names=[f'physics_{_oq}', 'inner', f'I_{_oq}'])
			# copy.deepcopy is necessary to avoid the error of node reuse.
			_noise_nodeList.append(copy.deepcopy(_noise_node))

		# Operating the noise channel to qubits
		for _ii, _qnum in enumerate(oqs):
			# Contract the noise channel with the qubits
			tn.connect(_qubits[_qnum][f'physics_{_qnum}'], _noise_nodeList[_ii]['inner'])
			_qubits[_qnum] = tn.contract_between(_qubits[_qnum], _noise_nodeList[_ii], name=f'qubit_{_qnum}')
			tools.EdgeName2AxisName([_qubits[_qnum]])  # Tensor append a new rank call 'I_{}'.format(_qnum) here.
			_dup_item, _dup_idx = tools.find_duplicate(_qubits[_qnum].axis_names)
			if _dup_item:
				# Number of axis name before the reshape operation(contain duplicates)
				_length = len(_qubits[_qnum].axis_names)
				# Find the shape of the tensor after the reshape operation
				_reshape_shape = list(_qubits[_qnum].tensor.shape)
				_reshape_shape[_dup_idx[1]] = _reshape_shape[_dup_idx[0]] * _reshape_shape[_dup_idx[1]]
				_reshape_shape.pop(_dup_idx[0])
				# Generate a random string without duplicates, if len = 4, then the string is 'abcd' as Einstein notation.
				_random_string = tools.generate_random_string_without_duplicate(_length)
				_random_string_reorder = tools.move_index(_random_string, _dup_idx[0],
													_dup_idx[1] - 1)  # Like 'ifcvbj' -> 'fcvbij'
				# Reshape the tensor
				_reshaped_tensor = tc.einsum(_random_string + ' -> ' + _random_string_reorder, _qubits[_qnum].tensor) \
					.reshape(_reshape_shape)
				_axis_names = copy.deepcopy(_qubits[_qnum].axis_names)
				_axis_names.pop(_dup_idx[0])

				""" Reform the qubit from adding noise channel, which causes edges' error in tensornetwork,
						easily speaking, reform the tensor. TensorNetwork package is not perfect. """
				_bond_list, _l_name, _r_name = [], None, None
				for _name in _qubits[_qnum].axis_names:
					if 'bond' in _name:
						_bond_list.append(_name)
						for _string_lst in _bond_list:
							_string_lst = _name.split('_')
							if reverse is False or reverse is None:
								if int(_string_lst[-1]) == int(_qnum):
									_l_name = _name
								elif int(_string_lst[-2]) == int(_qnum):
									_r_name = _name
							elif reverse is True:
								if int(_string_lst[-1]) == int(_qnum):
									_r_name = _name
								elif int(_string_lst[-2]) == int(_qnum):
									_l_name = _name

				if _l_name is not None:
					_left_qubit, _reforming_qubit = _qubits[_qnum][_l_name].get_nodes()
					_qubits[_qnum][_l_name].disconnect(_l_name, 'middle2left_edge')
				else:
					_left_qubit = None
				if _r_name is not None:
					_reforming_qubit, _right_qubit = _qubits[_qnum][_r_name].get_nodes()
					_qubits[_qnum][_r_name].disconnect('middle2right_edge', _r_name)
				else:
					_right_qubit = None

				# Previous information of _qubits[_qnum] is extracted, now we remade a new _qubits[_qnum]
				_qubits[_qnum] = tn.Node(_reshaped_tensor,
										 name=f'qubit_{_qnum}',
										 axis_names=_axis_names)  # Node's dimension of Edge f'I_{_qnum}' has been promoted.

				if _l_name is not None:
					tn.connect(_left_qubit[_l_name], _qubits[_qnum][_l_name], name=_l_name)
				else:
					pass
				if _r_name is not None:
					tn.connect(_qubits[_qnum][_r_name], _right_qubit[_r_name], name=_r_name)
				else:
					pass
				# ProcessFunction, for details, see the function definition
				tools.EdgeName2AxisName([_qubits[_qnum]])

			# Shape-relating
			_shape = _qubits[_qnum].tensor.shape
			_left_edge_shape = [_shape[_ii_] for _ii_ in range(len(_shape) - 1)]
			_left_dim = int(np.prod(_left_edge_shape))
			# SVD to truncate the inner dimension
			_u, _s, _ = tc.linalg.svd(tc.reshape(_qubits[_qnum].tensor, (_left_dim, _shape[-1])), full_matrices=False)
			_s = _s.to(dtype=tc.complex128)

			# Truncate the inner dimension
			if self.kappa is None:
				_kappa = _s.nelement()
			else:
				_kappa = self.kappa
			_s = _s[: _kappa]
			_u = _u[:, : _kappa]

			if len(_s.shape) == 1:
				_s = tc.diag(_s)

			# Back to the former shape
			_left_edge_shape.append(_s.shape[-1])
			_qubits[_qnum].tensor = tc.reshape(tc.matmul(_u, _s), _left_edge_shape)

	def add_gate(self, gate: AbstractGate, oqs: list):
		r"""
		Add quantum gate to circuit layer by layer.

		Args:
			gate: gate to be added;
			oqs: operating qubits.

		Returns:
			None
		"""
		if isinstance(gate, AbstractGate) is False:
			raise TypeError('Gate must be a AbstractGate.')
		if isinstance(oqs, int):
			oqs = [oqs]
		if isinstance(oqs, list) is False:
			raise TypeError('Operating qubits must be a list.')

		if self.realNoise is True:
			_transpile_gateList, _transpile_oqsList = self._transpile_gate(gate, oqs)
		elif self.crossTalk is True:
			_transpile_gateList, _transpile_oqsList = self._crossTalkZ_transpile(gate, oqs)
		else:
			_transpile_gateList, _transpile_oqsList = [gate], [oqs]

		for _num, _gate in enumerate(_transpile_gateList):
			if _gate.para is None:
				_para = None
			else:
				_para = (_gate.para / np.pi).as_integer_ratio()
				_para = repr(f'{_para[0]}/{_para[1]} pi')

			self.layers.add_module(_gate.name + str(_transpile_oqsList[_num]) + f'({_para})' + f'-G{self.i}', _gate)
			self._oqs_list.append(_transpile_oqsList[_num])
		self.i += 1

	def _calculate_DM(self, state_vector: bool = False, reduced_index: Optional[list[list[int] or int]] = None) -> tc.Tensor:
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

		def _re_permute(_axis_names_: list[str]):
			_left_, _right_ = [], []
			for _idx_, _name_ in enumerate(_axis_names_):
				if 'con_' not in _name_:
					_left_.append(_idx_)
				else:
					_right_.append(_idx_)
			return _left_ + _right_

		if not reduced_index:
			reduced_index = None

		if reduced_index is not None:
			if reduced_index[np.argmax(reduced_index)] >= self.qnumber:
				raise ValueError('Reduced index should not be larger than the qubit number.')

		if state_vector is False:
			_qubits_conj = copy.deepcopy(self.state)
			for _n in range(self.qnumber):
				_qubits_conj[_n].tensor = _qubits_conj[_n].tensor.conj()
			_allowed_outer_product = True

			# Differential name the conjugate qubits' edges name to permute the order of the indices
			for _i, _qubit_conj in enumerate(_qubits_conj):
				_qubit_conj.name = 'con_' + _qubit_conj.name
				for _ii in range(len(_qubit_conj.edges)):
					if 'physics' in _qubit_conj[_ii].name:
						_qubit_conj[_ii].name = 'con_' + _qubit_conj[_ii].name
						_qubit_conj.axis_names[_ii] = 'con_' + _qubit_conj.axis_names[_ii]

			_contract_nodes = []
			for i in range(len(self.state)):
				if self.ideal is False:
					try:
						tn.connect(self.state[i][f'I_{i}'], _qubits_conj[i][f'I_{i}'])
						_allowed_outer_product = False  # Edges between ket-bra are now connected, outer product is not allowed.
					except ValueError:
						_allowed_outer_product = True   # Cases that not every qubit has been added noise.
				_contract_nodes.append(tn.contract_between(self.state[i], _qubits_conj[i], name=f'contracted_qubit_{i}',
														   allow_outer_product=_allowed_outer_product))
			tools.EdgeName2AxisName(_contract_nodes)

			# Reduced density matrix
			# Bra--Ket space of each qubit in memory are now contracted into a higher rank tensor with two physical indices.
			if reduced_index is not None:
				if isinstance(reduced_index, int):
					reduced_index = [reduced_index]
				if not isinstance(reduced_index, list):
					raise TypeError('reduced_index should be int or list[int]')
				for _idx in reduced_index:
					tn.connect(_contract_nodes[_idx][f'physics_{_idx}'],
							   _contract_nodes[_idx][f'con_physics_{_idx}'])
					_contract_nodes[_idx] = tn.contract_trace_edges(_contract_nodes[_idx])
			else:
				reduced_index = []

			_dm = _contract_nodes[0]
			for _ii in range(1, len(_contract_nodes)):
				_dm = tn.contract_between(_dm, _contract_nodes[_ii], allow_outer_product=True,
										  name='Contracted_DM_NODE')
			tools.EdgeName2AxisName([_dm])

			_dm.tensor = tc.permute(_dm.tensor, _re_permute(_dm.axis_names))

			_reshape_size = self.qnumber - len(reduced_index)

			self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
			return self.DM
		else:
			if self.fVR is False:
				if self.ideal is False:
					raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
			if reduced_index is not None:
				raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

			_vector = self.state[0]
			for _i in range(1, len(self.state)):
				_vector = tn.contract_between(_vector, self.state[_i], allow_outer_product=True)

			if self.fVR is False:
				self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1))
			else:
				self.DM = _vector.tensor
			return self.DM

	def forward(self, _state: list[tn.Node] = None,
				state_vector: bool = False, reduced_index: list = None, forceVectorRequire: bool = False) -> tc.Tensor:
		r"""
		Forward propagation of tensornetwork.

		Returns:
			self.state: tensornetwork after forward propagation.
		"""
		self.initState = _state
		self.qnumber = len(_state)
		self.fVR = forceVectorRequire

		for _i in range(len(self.layers)):
			self._add_gate(_state, _i, _oqs=self._oqs_list[_i])
			if self.ideal is False:
				_oqs = self._oqs_list[_i]
				_reverse = False
				if self.layers[_i].single is False:
					if self.unified is True:
						if self._oqs_list[_i][0] > self._oqs_list[_i][-1]:
							_reverse = True
						self._add_noise(_state, oqs=_oqs, reverse=_reverse, noiseInfo=self.Noise,
										noise_type='depolarization')
						self._add_noise(_state, oqs=_oqs, reverse=_reverse, noiseInfo=self.Noise,
										noise_type='amplitude_phase_damping_error')
					else:
						pass
				else:       # single Qubit gate
					if self.layers[_i].ideal is True:
						pass
					else:
						_reverse = None

						self._add_noise(_state, oqs=_oqs, reverse=_reverse, noiseInfo=self.Noise, noise_type='depolarization')
						self._add_noise(_state, oqs=_oqs, reverse=_reverse, noiseInfo=self.Noise, noise_type='amplitude_phase_damping_error')
			if self.tnn_optimize is True:
				check = checkConnectivity(_state)
				if check is True:
					qr_left2right(_state)
					svd_right2left(_state, chi=self.chi)
		self.state = _state
		_dm = self._calculate_DM(state_vector=state_vector, reduced_index=reduced_index)

		return _dm