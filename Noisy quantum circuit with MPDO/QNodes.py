"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
from Library.ADGate import TensorGate
from Library.AbstractGate import AbstractGate
from Library.ADCircuits import TensorCircuit
import numpy as np
import Library.tools as tools
import tensornetwork as tn
import torch as tc

tn.set_default_backend("pytorch")

def ghzLike_nodes(qnumber, chi: int = None, kappa: int = None, ideal: bool = True):
	r"""
	ghzLike state preparation with nodes.

	Args:
		qnumber: Node number of the state;
		chi: Maximum bond dimension to be saved in SVD;
		kappa: Maximum bond dimension to be saved in SVD for noisy bond;
		ideal: Whether to add noise channel.

	Returns:
		Node list of the state after preparation.
	"""
	_circuit = TensorCircuit(ideal=ideal)
	# Apply hardmard gate
	_circuit.add_gate(AbstractGate().h(), [0])

	if qnumber > 1:
		# Apply CNOT gate
		for i in range(qnumber - 1):
			_circuit.add_gate(AbstractGate().cnot(), [i, i + 1])

	return _circuit

def used4test(qnumber=4, chi: int = None, ideal: bool = True):
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
		qnumber: Number of qubits;
		chi: Maximum bond dimension of the state;
		ideal: Whether to add noise channel.

	Returns:
		Node list of the state after preparation;
	"""
	assert qnumber == 4
	_circuit = TensorCircuit(ideal=ideal, chi=chi)
	# layer1
	_circuit.add_gate(AbstractGate().h(), [0, 2])
	_circuit.add_gate(AbstractGate().x(), [1])
	# layer2
	_circuit.add_gate(AbstractGate().cnot(), [0, 1])
	_circuit.add_gate(AbstractGate().cnot(), [2, 3])
	# layer3
	_circuit.add_gate(AbstractGate().cnot(), [1, 2])
	# layer4
	_circuit.add_gate(AbstractGate().x(), [0, 2, 3])
	_circuit.add_gate(AbstractGate().h(), [1])

	return _circuit


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

	def _random_double_gate(_circuit_, _start: int, _double_gate_choice_: list):
		r""" Generate a random double gate """
		assert _start == 0 or _start == 1   # 0 --> Even layers, 1 --> Odd layers
		for _i in range(_start, qnumber - 1, 2):
			_random_gate_choice = np.random.choice(['ci', 'cx'], p=[0.5, 0.5])
			_random_reversal = bool(np.random.choice([True, False], p=[0.5, 0.5]))

			if _random_reversal is True:
				_random_gate_choice = _transpose_str(_random_gate_choice)
				_control, _target = _i + 1, _i
			else:
				_control, _target = _i, _i + 1

			_double_gate_choice_.append(_random_gate_choice + f'_{_control}_{_target}')
			if _random_gate_choice == 'cx' or 'xc':
				_circuit_.add_gate(AbstractGate().cnot(), [_control, _target])
			else:
				pass        # No actions were taken

	if qnumber <= 0 or isinstance(qnumber, int) is False:
		raise ValueError('Qnumber must be a positive integer.')
	if depth <= 0 or isinstance(depth, int) is False:
		raise ValueError('Depth must be a positive integer.')

	_random_paraList, _random_paraList1 = tc.zeros(qnumber * 2), tc.zeros(1)
	_double_gate_choice = []

	_circuit = TensorCircuit(ideal=False, chi=chi, kappa=kappa)

	for _depth in range(depth):
		# A layer of Single-qubit gates
		_random_paraList1 = tc.rand(qnumber) * tc.pi
		for _qnum in range(qnumber):
			_circuit.add_gate(AbstractGate().rz(_random_paraList1[_qnum]), [_qnum])

		if qnumber >= 2 and depth >= 2:
			_random_double_gate(_circuit, _start=0, _double_gate_choice_=_double_gate_choice)   # Independent random selection
			_random_double_gate(_circuit, _start=1, _double_gate_choice_=_double_gate_choice)   # Independent random selection
			_double_gate_choice.append('|')

			# A layer of Single-qubit gates
			_random_paraList2 = tc.rand(qnumber) * tc.pi
			for _qnum in range(qnumber):
				_circuit.add_gate(AbstractGate().rx(_random_paraList2[_qnum]), [_qnum])

			_random_paraList1 = tc.cat((_random_paraList1, _random_paraList2), dim=0)

			# Record the parameters
			_random_paraList = tc.vstack((_random_paraList, _random_paraList1))

	if qnumber == 1 or depth == 1:
		return _circuit, _random_paraList1, _double_gate_choice
	else:
		return _circuit, _random_paraList[1:, :], _double_gate_choice


if __name__ == '__main__':
	qn = 4
	# circuit, paraList, gate_choice = random_circuit_DM4Train(qnumber=qn, depth=10)
	circuit = used4test(ideal=True)
	initState = tools.create_ket0Series(qnumber=qn)
	state = circuit(initState, state_vector=True)

	print('density matrix:\n', state)
	# print('parameter list:\n', paraList)
	# print('gate choice:\n', gate_choice)