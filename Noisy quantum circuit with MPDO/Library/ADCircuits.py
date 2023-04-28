"""
Author: weiguo_ma
Time: 04.26.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import copy
import collections
import tensornetwork as tn
from noise_channel import depolarization_noise_channel, amp_phase_damping_error
import numpy as np
from ADGate import TensorGate
from torch import nn
from chipInfo import Chip_information
import Library.tools as tools

class TensorCircuit(nn.Module):
	def __init__(self, qnumber: int = None, initState: str or tc.Tensor = None, ideal: bool = True,
	            chi: int = None, kappa: int = None,
	            chip: str = None, device: str or int = 'cpu'):
		super(TensorCircuit, self).__init__()
		self.qnumber = qnumber
		self.ideal = ideal
		self.device = tools.select_device(device)
		self.dtype = tc.complex128

		if chip is None:
			chip = 'beta4Test'
		self.chip = Chip_information().__getattribute__(chip)()
		self.T1 = self.chip.T1
		self.T2 = self.chip.T2
		self.GateTime = self.chip.gateTime
		self.dpc_errorRate = self.chip.dpc_errorRate

		if initState == '0' or 'ket0':
			self.state = tools.create_ket0Series(qnumber, dtype=self.dtype)
		elif initState == '1' or 'ket1':
			self.state = tools.create_ket1Series(qnumber, dtype=self.dtype)
		elif initState == '+' or 'ket+':
			self.state = tools.create_ketPlusSeries(qnumber, dtype=self.dtype)
		elif initState == '-' or 'ket-':
			self.state = tools.create_ketMinusSeries(qnumber, dtype=self.dtype)
		else:
			self.state = tools.create_ketRandomSeries(qnumber, initState, dtype=self.dtype)

		self.layers = nn.Sequential()

		self.DM = None
		self.DMNode = None

	def _add_gate(self, gate: TensorGate, _oqs: list):
		r"""
		Add quantum Gate to tensornetwork

		Args:
			gate: gate to be added;
			_oqs: operating qubits.

		Returns:
			Tensornetwork after adding gate.
		"""
		if isinstance(self.state, list) is False:
			raise TypeError('Qubit must be a list.')
		if isinstance(gate, TensorGate) is False:
			raise TypeError('Gate must be a TensorGate.')
		if isinstance(_oqs, list) is False:
			raise TypeError('Operating qubits must be a list.')

		single = gate.single
		if single is False:  # Two-qubit gate
			if len(_oqs) > 2:
				raise NotImplementedError('Only two-qubit gates are supported currently.')
			if _oqs[0] == _oqs[1]:
				raise ValueError('Operating qubits must be different.')
			if tools.is_nested(_oqs) is True:
				raise NotImplementedError('Series CNOT gates are not supported yet.')
			else:
				_edges = []
				gate = tn.Node(gate.tensor, name=gate.name,
				               axis_names=[f'inner_{_oqs[0]}', f'inner_{_oqs[1]}', f'physics_{_oqs[0]}',
				                           f'physics_{_oqs[1]}'])
				# Created a new node in memory
				_contract_qubits = tn.contract_between(self.state[_oqs[0]], self.state[_oqs[1]],
				                                       name='q_{}_{}'.format(_oqs[0], _oqs[1]),
				                                       allow_outer_product=True)
				tools.EdgeName2AxisName([_contract_qubits])
				for _i, _bit in enumerate(_oqs):
					_edges.append(tn.connect(_contract_qubits['physics_{}'.format(_bit)], gate[f'inner_{_bit}']))

				# contract the connected edge and inherit the name of the pre-qubit
				gate = tn.contract_between(_contract_qubits, gate, name=gate.name)

				# ProcessFunction, for details, see the function definition.
				tools.EdgeName2AxisName([gate])

				# Split back to two qubits
				_left_AxisName, _right_AxisName = self.state[_oqs[0]].axis_names, self.state[_oqs[1]].axis_names
				if 'bond_{}_{}'.format(_oqs[0], _oqs[1]) in _left_AxisName:
					# remove the bond_name between two qubits, which take appearance in both qubits simultaneously
					_left_AxisName.remove('bond_{}_{}'.format(_oqs[0], _oqs[1]))
					_right_AxisName.remove('bond_{}_{}'.format(_oqs[0], _oqs[1]))

				_left_edges, _right_edges = [gate[name] for name in _left_AxisName], \
				                            [gate[name] for name in _right_AxisName]

				self.state[_oqs[0]], self.state[_oqs[1]], _ = tn.split_node(gate,
				                                                      left_edges=_left_edges,
				                                                      right_edges=_right_edges,
				                                                      left_name=f'qubit_{_oqs[0]}',
				                                                      right_name=f'qubit_{_oqs[1]}',
				                                                      edge_name=f'bond_{_oqs[0]}_{_oqs[1]}')
		else:
			gate_list = [tn.Node(gate.tensor, name=gate.name, axis_names=[f'inner_{_idx}', f'physics_{_idx}'])
			             for _idx in _oqs]
			for _i, _bit in enumerate(_oqs):
				tn.connect(self.state[_bit][f'physics_{_bit}'], gate_list[_i][f'inner_{_bit}'])
				# contract the connected edge and inherit the name of the pre-qubit
				self.state[_bit] = tn.contract_between(self.state[_bit], gate_list[_i], name=self.state[_bit].name)
				# ProcessFunction, for details, see the function definition.
				tools.EdgeName2AxisName([self.state[_bit]])

	def add_noise(self, oqs: list[int] or int,
	                        noise_type: str,
	                        p: float = None,
	                        time: float = None,
	                        T1: float = None,
	                        T2: float = None,
	                        kappa: int = None):
		r"""
		Apply the noise channel to the qubits.
		Args:
			oqs: The qubits to be applied the noise channel;
			noise_type: The type of the noise channel;
			p: The probability of the noise channel;
			time: The time of the noise channel;
			T1: The T1 time of the noise channel;
			T2: The T2 time of the noise channel;
			kappa: Truncation dimension upper bond of the noise channel.
		Returns:
			The qubits after applying the noise channel.
				!!! Actually no return, but the qubits are changed in the memory. !!!
		Additional information:
			The noise channel is applied to the qubits by the following steps:
				1. Construct the noise channel;
				2. Construct the error tensor;
				3. Contract the error tensor with the qubits;
				4. Fix the axis format of the qubits.
		Attention:
			On account for the function _hard_fix_axis_format,
				The qubits should be in the following format:
					1. The first edge is the bond edge;
					2. The second edge is the physics edge;
					3. The third edge is the bond edge.
				The qubits should be in the following format:
					1. The first edge is the bond edge;
					2. The second edge is the physics edge;
					3. The third edge is the bond edge.
		"""

		def _import_error_tensor(_noise_type_: str, _p_: float):
			if _noise_type_ == 'depolarization':
				return depolarization_noise_channel(_p_)
			elif _noise_type_ == 'amplitude_phase_damping_error':
				return amp_phase_damping_error(time, T1, T2)
			else:
				raise NotImplementedError(f'Noise type {_noise_type_} is not implemented yet.')

		def _find_duplicate(_lst_):
			"""
			Find the duplicate items and their indices in a list.
			"""
			_duplicate_item_ = [item for item, count in collections.Counter(_lst_).items() if count > 1]
			_duplicate_idx_ = [idx for idx, item in enumerate(_lst_) if item in _duplicate_item_]
			return _duplicate_item_, _duplicate_idx_

		if not isinstance(self.state, list):
			if not isinstance(self.state, tn.Node) or not isinstance(self.state, tn.AbstractNode):
				raise TypeError(f'qubits must be a list of tn.Node or tn.AbstractNode, but got {type(self.state)}')
			self.state = [self.state]
		if not isinstance(oqs, list):
			if not isinstance(oqs, int):
				raise TypeError(f'oqs must be a list of int, but got {type(oqs)}')
			oqs = [oqs]
		if len(oqs) > len(self.state):
			raise ValueError(f'len(oqs) must be less than or equal to to len(qubits),'
			                 f' but got {len(oqs)} and {len(self.state)}')
		if p is None and (time is None or T1 is None or T2 is None):
			raise ValueError('The noise parameter must be specified.')

		# Create Node for noise channel
		_noise_tensor = _import_error_tensor(_noise_type_=noise_type, _p_=p)
		_noise_nodeList = []
		for _oq in oqs:
			_noise_node = tn.Node(_noise_tensor, name='noise_node',
			                      axis_names=['inner', f'physics_{_oq}', f'I_{_oq}'])
			# copy.deepcopy is necessary to avoid the error of node reuse.
			_noise_nodeList.append(copy.deepcopy(_noise_node))

		# Operating the noise channel to qubits
		for _ii, _qnum in enumerate(oqs):
			_edge = tn.connect(self.state[_qnum][f'physics_{_qnum}'], _noise_nodeList[_ii]['inner'])
			self.state[_qnum] = tn.contract(_edge, name=f'qubit_{_qnum}')
			# ProcessFunction, for details, see the function definition.
			tools.EdgeName2AxisName([self.state[_qnum]])  # Tensor append a new rank call 'I_{}'.format(_qnum) here.
			# raise NotImplementedError('When double/multi errors are applied to a same qubit, problem occurs.'
			#                           'The reason is that the node connection broken while the node is working.')

			_dup_item, _dup_idx = _find_duplicate(self.state[_qnum].axis_names)
			if _dup_item:
				# Number of axis name before the reshape operation(contain duplicates)
				_length = len(self.state[_qnum].axis_names)
				# Find the shape of the tensor after the reshape operation
				_reshape_shape = copy.deepcopy(list(self.state[_qnum].tensor.shape))
				_reshape_shape[_dup_idx[1]] = _reshape_shape[_dup_idx[0]] * _reshape_shape[_dup_idx[1]]
				_reshape_shape.pop(_dup_idx[0])
				# Generate a random string without duplicates, if len = 4, then the string is 'abcd' as Einstein notation.
				_random_string = tools.generate_random_string_without_duplicate(_length)
				_random_string_reorder = tools.move_index(_random_string, _dup_idx[0],
				                                    _dup_idx[1] - 1)  # Like 'ifcvbj' -> 'fcvbij'
				# Reshape the tensor
				_reshaped_tensor = tc.einsum(_random_string + ' -> ' + _random_string_reorder, self.state[_qnum].tensor) \
					.reshape(_reshape_shape)
				_axis_names = copy.deepcopy(self.state[_qnum].axis_names)
				_axis_names.pop(_dup_idx[0])

				""" Though we clarify a new node called _left_node, in memory, 
							it is still the same node as connected to self.state[_qnum]. """
				_left_node, _right_node = self.state[_qnum][0].get_nodes()  # 0 is the hard code that bond_idx at first pos
				if _right_node is not None:  # Which means that the node's the most left edge is not bond_edge
					if 'bond' not in self.state[_qnum][0].name:
						raise ValueError(f'HardCodeERROR. The edge name must be bond, but got {self.state[_qnum][0].name}')
					_left_edge_name = _left_node.axis_names[-1]  # -1 is the hard code that bond_idx at last pos
					self.state[_qnum][0].disconnect(_left_edge_name, 'right_edge')
					# ProcessFunction, for details, see the function definition
					tools.EdgeName2AxisName([_left_node])

				self.state[_qnum] = tn.Node(_reshaped_tensor,
				                         name=f'qubit_{_qnum}',
				                         axis_names=_axis_names)  # Node's edge's, named 'I_{}', dimension has been promoted.
				if _right_node is not None:
					tn.connect(_left_node[_left_edge_name], self.state[_qnum][0], name=self.state[_qnum].axis_names[0])
					tools.EdgeName2AxisName([self.state[_qnum]])

			# Shape-relating
			_shape = self.state[_qnum].tensor.shape
			_left_edge_shape = [_shape[_ii_] for _ii_ in range(len(_shape) - 1)]
			_left_dim = int(np.prod(_left_edge_shape))
			# SVD to truncate the inner dimension
			_u, _s, _ = tc.linalg.svd(tc.reshape(self.state[_qnum].tensor, (_left_dim, _shape[-1])), full_matrices=False)
			_s = _s.to(dtype=tc.complex128)

			# Truncate the inner dimension
			if kappa is None:
				kappa = _s.nelement()
			_s = _s[: kappa]
			_u = _u[:, : kappa]

			if len(_s.shape) == 1:
				_s = tc.diag(_s)

			# Back to the former shape
			_left_edge_shape.append(_s.shape[-1])
			self.state[_qnum].tensor = tc.reshape(tc.matmul(_u, _s), _left_edge_shape)

	def add_gate(self, gate: TensorGate, oqs: list):
		r"""
		        Add quantum gate to circuit layer by layer.

		        Args:
		            gate: gate to be added;
		            oqs: operating qubits.

		        Returns:
		            None
		        """
		if isinstance(gate, TensorGate) is False:
			raise TypeError('Gate must be a TensorGate.')
		if isinstance(oqs, int):
			oqs = [oqs]
		if isinstance(oqs, list) is False:
			raise TypeError('Operating qubits must be a list.')

		self.layers.add_module(gate.name + str(oqs), gate)
		self._add_gate(gate, oqs)
		if self.ideal is False:
			self.add_noise(oqs=oqs, noise_type='depolarization', p=self.dpc_errorRate)
			self.add_noise(oqs=oqs, noise_type='amplitude_phase_damping_error'
			                                  , time=self.GateTime, T1=self.T1, T2=self.T2)

	def forward(self):
		r"""
        Forward propagation of tensornetwork.

        Returns:
            self.state: tensornetwork after forward propagation.
        """
		self.state = self.layers(self.state)
		return self.state

	def calculate_DM(self, state_vector: bool = False, reduced_index: list = None):
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

		if state_vector is False:
			_qubits_conj = copy.deepcopy(self.state)
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
					tn.connect(_contract_nodes[_idx][f'physics_{_idx}'], _contract_nodes[_idx][f'con_physics_{_idx}'])
					_contract_nodes[_idx] = tn.contract_trace_edges(_contract_nodes[_idx])
			else:
				reduced_index = []

			_dm = _contract_nodes[0]
			for _ii in range(1, len(_contract_nodes)):
				_dm = tn.contract_between(_dm, _contract_nodes[_ii], allow_outer_product=True, name='Contracted_DM_NODE')
			tools.EdgeName2AxisName([_dm])

			_dm.tensor = tc.permute(_dm.tensor, _re_permute(_dm.axis_names))

			_reshape_size = self.qnumber - len(reduced_index)

			self.DM, self.DMNode = _dm.tensor.reshape((2 ** _reshape_size, 2 ** _reshape_size)), self.DMNode
			return self.DM
		else:
			if self.ideal is False:
				raise ValueError('Noisy circuit cannot be represented by state vector efficiently.')
			if reduced_index is not None:
				raise ValueError('State vector cannot efficiently represents the reduced density matrix.')

			_vector = self.state[0]
			for _i in range(1, len(self.state)):
				_vector = tn.contract_between(_vector, self.state[_i], allow_outer_product=True)

			self.DM = _vector.tensor.reshape((2 ** self.qnumber, 1))
			return self.DM


if __name__ == '__main__':
	Gates = TensorGate()
	circuit = TensorCircuit(2, initState='0', ideal=False)
	circuit.add_gate(Gates.h(), oqs=[0])
	circuit.add_gate(Gates.cnot(), oqs=[0, 1])
	print(circuit.state)
	dm = circuit.calculate_DM()
	print(dm)