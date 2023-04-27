"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
from Library.basic_gates import TensorGate
import numpy as np
import Library.tools as tools
import tensornetwork as tn
import Library.tnn_optimizer as opt
import Library.noise_channel as noise_channel
import torch as tc

tn.set_default_backend("pytorch")

def ghzLike_nodes(_qnumber, chi: int = None, noise: bool = False):
	r"""
	ghzLike state preparation with nodes.

	Args:
		_qnumber: Node number of the state;
		chi: Maximum bond dimension to be saved in SVD.
		noise: Whether to add noise channel.

	Returns:
		Node list of the state after preparation.
	"""
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(_qnumber)
	# Apply hardmard gate
	tools.add_gate(_qubits, Gates.h(), [0])
	if noise is True:
		noise_channel.apply_noise_channel(_qubits, [0, 1], noise_type='depolarization', p=1e-2)
		noise_channel.apply_noise_channel(_qubits, [0, 1], noise_type='amplitude_phase_damping_error'
		                                  , time=30, T1=2e3, T2=2e2)
	if _qnumber > 1:
		# Apply CNOT gate
		for i in range(_qnumber - 1):
			tools.add_gate(_qubits, Gates.cnot(), [i, i + 1])
			if noise is True:
				noise_channel.apply_noise_channel(_qubits, [i, i + 1], noise_type='depolarization', p=1e-2)
				# noise_channel.apply_noise_channel(_qubits, [i, i + 1], noise_type='amplitude_phase_damping_error'
				#                                   , time=30, T1=2e3, T2=2e2)
		print(_qubits)
	# Optimization
	opt.qr_left2right(_qubits)
	opt.svd_right2left(_qubits, chi=chi)
	return _qubits

def scalable_simulation_scheme2(theta: float, chi: int = None):
	r"""
	Scalable simulation scheme 2.
	Args:
		theta: The angle of rotation;
		chi: Maximum bond dimension to be saved in SVD.

	Returns:
		Node list of the state after preparation.

	Additional information:
		Currently did not use the Adam to optimize the parameters.
	"""
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(7)
	# Initialize the state
	print('Initializing the state...')
	tools.add_gate(_qubits, Gates.ry(theta), [3])
	print('adding h')
	tools.add_gate(_qubits, Gates.h(), [0, 1, 2, 4, 5, 6])

	# Apply rzz gate
	print('Applying rzz gate...')
	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [0, 1])
	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [2, 3])
	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [4, 5])

	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [1, 2])
	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [3, 4])
	tools.add_gate(_qubits, Gates.rzz(np.pi / 2), [5, 6])

	# Apply rx gate
	print('Applying rx gate...')
	tools.add_gate(_qubits, Gates.rx(np.pi / 2), [0, 1, 2, 4, 5, 6])
	# Optimization
	opt.qr_left2right(_qubits)
	opt.svd_right2left(_qubits, chi=chi)

	return _qubits

def used4test(chi: int = None):
	r"""
	Generate a random circuit to test whether the program is correct and the function of add_gate, compare with qutip
		simulation, the result is the same, which implies that the program is correct.

		'''
		from qutip import basis, tensor, qeye
		from qutip_qip.operations import cnot, rx, ry, rz, x_gate, y_gate, z_gate, hadamard_transform

		I = qeye(2)
		init = tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))
		layer1= tensor(hadamard_transform(), x_gate(), hadamard_transform(), I)
		layer2 = cnot(4, 0, 1) * cnot(4, 2, 3)
		layer3 = cnot(4, 1, 2)
		layer4 = tensor(x_gate(), hadamard_transform(), x_gate(), x_gate())

		output = layer4 * layer3 * layer2 * layer1 * init
		print(output)
		'''

	I noticed that the optimization layer can be ONLY apply while the whole system are CONNECTED together, allow
		outer product in qr and svd may solve the problem, which might cause a mathematical problem,
			but there still raises other problems.

	Args:
		chi: Maximum bond dimension of the state;

	Returns:
		Node list of the state after preparation;
	"""
	_qnumber = 4
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(_qnumber)
	# layer1
	tools.add_gate(_qubits, Gates.h(), [0, 2])
	tools.add_gate(_qubits, Gates.x(), [1])
	# layer2
	tools.add_gate(_qubits, Gates.cnot(), [0, 1])
	tools.add_gate(_qubits, Gates.cnot(), [2, 3])
	# layer3
	tools.add_gate(_qubits, Gates.cnot(), [1, 2])
	opt.qr_left2right(_qubits)
	opt.svd_right2left(_qubits, chi=chi)
	# layer4
	tools.add_gate(_qubits, Gates.x(), [0, 2, 3])
	tools.add_gate(_qubits, Gates.h(), [1])

	return _qubits


def random_circuit_DM4Train(qnumber: int, depth: int, chi: int = None, kappa: int = None):
	r"""
	Generate a UNIFORM random-parameter circuit

	Args:
		qnumber: Number of qubits;
					While qnumber = 1, the circuit is only added with single gates;
		depth: Depth of the circuit;
					While depth = 1, the circuit is only added with single gates;
								> 1, the circuit returns into a UNIFORM circuit, whose depth is not exactly equal to
										the depth you input.
		chi: Maximum bond dimension of the state;
		kappa: Maximum inner dimension of the state.

	Returns:
		_dm: Density matrix of the state after preparation;
		_random_paraList[1:, :] : Random parameters of the circuit(first row is all zeros -- remove);
		_double_gate_choice: Choice of double gate.

	Additional information:
		A UNIFORM random circuit is defined as below:

			[q0] ---[RY: float_para]---[cx or      -----------------[RX: float_para]---
			[q1] ---[RY: float_para]---      pass]---[cx or      ---[RX: float_para]---
			[q2] ---[RY: float_para]---[cx or               pass]---[RX: float_para]---
			[q3] ---[RY: float_para]---      pass]---[cx or      ---[RX: float_para]---
				......
			[qn] ---[RY: float_para]---      pass]------------------[RX: float_para]---

		Where:
			[RX: float_para] is a RX gate with a random parameter;
			[cx or pass] is a CNOT gate or a pass gate -- Double-qubit gate;
			[RZ: float_para] is a RZ gate with a random parameter.

		ReturnValue _random_paraList:
			A row of parameters form a UNIFORM gate parameters in ONE DEPTH, that is,
					[qnumber | qnumber] :: [rx * qnumber | rz * qnumber]

		ReturnValue _double_gate_choice:
			A list with splitter '|' to split the double gate choice in each DEPTH,
				'ci' or 'ic' represent no actions were taken;
				'cx' represents a CNOT gate was taken, 'xc' represents a CNOT gate with reorder of qubits.
				'_{}_{}' --> '_{_control}_{_target}'
					while 'xc' is taken, int(_control) > int(_target)
	"""
	def _transpose_str(_string: str):
		r""" str transformation: 'ab' --> 'ba' """
		_string = list(_string)
		_string[0], _string[1] = _string[1], _string[0]
		return ''.join(_string)

	def _random_double_gate(_qubits, _start: int, _double_gate_choice_: list):
		r""" Generate a random double gate """
		assert _start == 0 or _start == 1   # 0 --> Even layers, 1 --> Odd layers
		_Gates = TensorGate()
		for _i in range(_start, qnumber - 1, 2):
			_random_gate_choice = np.random.choice(['ci', 'cx'], p=[0.5, 0.5])
			_random_reversal = bool(np.random.choice([True, False], p=[0.5, 0.5]))

			if _random_reversal is True:
				_random_gate_choice = _transpose_str(_random_gate_choice)
				_control, _target = _i + 1, _i
			else:
				_control, _target = _i, _i + 1

			_double_gate_choice_.append(_random_gate_choice + f'_{_control}_{_target}')
			if _random_gate_choice == 'ci' or 'ic':
				pass        # No actions were taken
			else:
				tools.add_gate(_qubits, _Gates.cnot(), [_control, _target])

	if qnumber <= 0 or isinstance(qnumber, int) is False:
		raise ValueError('Qnumber must be a positive integer.')
	if depth <= 0 or isinstance(depth, int) is False:
		raise ValueError('Depth must be a positive integer.')

	_random_paraList, _random_paraList1 = tc.zeros(qnumber * 2), tc.zeros(1)
	_double_gate_choice = []

	Gates = TensorGate()
	_qubits = tools.create_ket0Series(qnumber)

	for _depth in range(depth):
		# A layer of Single-qubit gates
		_random_paraList1 = tc.rand(qnumber) * tc.pi
		for _qnum in range(qnumber):
			tools.add_gate(_qubits, Gates.rz(_random_paraList1[_qnum]), [_qnum])

		if qnumber >= 2 and depth >= 2:
			_random_double_gate(_qubits, _start=0, _double_gate_choice_=_double_gate_choice)   # Independent random selection
			_random_double_gate(_qubits, _start=1, _double_gate_choice_=_double_gate_choice)   # Independent random selection
			_double_gate_choice.append('|')

			# A layer of Single-qubit gates
			_random_paraList2 = tc.rand(qnumber) * tc.pi
			for _qnum in range(qnumber):
				tools.add_gate(_qubits, Gates.rx(_random_paraList2[_qnum]), [_qnum])

			_random_paraList1 = tc.cat((_random_paraList1, _random_paraList2), dim=0)

			# Tensor Optimization
			opt.qr_left2right(_qubits)
			opt.svd_right2left(_qubits, chi=chi)

			# Record the parameters
			_random_paraList = tc.vstack((_random_paraList, _random_paraList1))

	# Calculate Density Matrix
	_, _dm = tools.calculate_DM(_qubits)

	if qnumber == 1 or depth == 1:
		return _dm, _random_paraList1, _double_gate_choice
	else:
		return _dm, _random_paraList[1:, :], _double_gate_choice


if __name__ == '__main__':
	dm, paralist, gate_choice = random_circuit_DM4Train(qnumber=5, depth=10)
	print('density matrix:\n', dm)
	print('parameter list:\n', paralist)
	print('gate choice:\n', gate_choice)