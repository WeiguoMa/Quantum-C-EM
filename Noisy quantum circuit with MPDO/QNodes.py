"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
from basic_gates import *
import tools
import tensornetwork as tn
import algorithm
import noise_channel

tn.set_default_backend("pytorch")

def ghzLike_edges(_qnumber):
	r"""
	ghzLike state preparation with edges.

	Args:
		_qnumber: Node number of the state.

	Returns:
		Edge list of the state after preparation.
	"""
	all_nodes = []
	Gates = TensorGate()
	# NodeCollection allows us to keep track of all the nodes in the network
	with tn.NodeCollection(all_nodes):
		state_nodes = tools.create_ket0Series(_qnumber)
		# which generated a list of two rank-1 tensors representing the initial state |00>
		_qubits = [node[0] for node in state_nodes]  # every is an edge of tensor
		# Apply the Hadamard gate to the first qubit
		tools.add_gate(_qubits, Gates.h(), [0])
		# Apply the CNOT gate to the first and second qubits
		for i in range(_qnumber - 1):
			tools.add_gate(_qubits, Gates.cnot(), [i, i + 1])
	return all_nodes, _qubits

def ghzLike_nodes(_qnumber, _chi: int = None):
	r"""
	ghzLike state preparation with nodes.

	Args:
		_qnumber: Node number of the state;
		_chi: Maximum bond dimension to be saved in SVD.

	Returns:
		Node list of the state after preparation.
	"""
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(_qnumber)
	# Apply hardmard gate
	tools.add_gate_truncate(_qubits, Gates.h(), [0])
	# Apply CNOT gate
	for i in range(_qnumber - 1):
		tools.add_gate_truncate(_qubits, Gates.cnot(), [i, i + 1])
	# Optimization
	algorithm.qr_left2right(_qubits)
	algorithm.svd_right2left(_qubits, _chi=_chi)
	return _qubits

def scalable_simulation_scheme2(_theta: float, _chi: float = None):
	r"""
	Scalable simulation scheme 2.
	Args:
		_theta: The angle of rotation;
		_chi: Maximum bond dimension to be saved in SVD.

	Returns:
		Node list of the state after preparation.

	Additional information:
		Currently did not use the Adam to optimize the parameters.
	"""
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(7)
	# Initialize the state
	print('Initializing the state...')
	tools.add_gate_truncate(_qubits, Gates.ry(_theta), [3])
	print('adding h')
	tools.add_gate_truncate(_qubits, Gates.h(), [0, 1, 2, 4, 5, 6])

	# Apply rzz gate
	print('Applying rzz gate...')
	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi/2), [0, 1])
	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi / 2), [2, 3])
	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi / 2), [4, 5])

	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi / 2), [1, 2])
	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi / 2), [3, 4])
	tools.add_gate_truncate(_qubits, Gates.rzz(np.pi / 2), [5, 6])

	# Apply rx gate
	print('Applying rx gate...')
	tools.add_gate_truncate(_qubits, Gates.rx(np.pi/2), [0, 1, 2, 4, 5, 6])
	# Optimization
	algorithm.qr_left2right(_qubits)
	algorithm.svd_right2left(_qubits, _chi=_chi)

	return _qubits

def used4test(_chi=None):
	r"""
	Generate a random circuit to test whether the program is correct and the function of add_gate, compare with qutip
		simulation, the result is the same, which implies that the program is correct.

		'''
		from qutip import basis, tensor
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
		_chi: Maximum bond dimension of the state;

	Returns:
		Node list of the state after preparation;
	"""
	_qnumber = 4
	Gates = TensorGate()
	_qubits = tools.create_ket0Series(_qnumber)
	# layer1
	tools.add_gate_truncate(_qubits, Gates.h(), [0, 2])
	tools.add_gate_truncate(_qubits, Gates.x(), [1])
	noise_channel.apply_noise_channel(_qubits, [0, 1, 2], _noise_type='depolarization', _p=11e-4)

	# layer2
	tools.add_gate_truncate(_qubits, Gates.cnot(), [0, 1])
	tools.add_gate_truncate(_qubits, Gates.cnot(), [2, 3])
	# layer3
	tools.add_gate_truncate(_qubits, Gates.cnot(), [1, 2])
	algorithm.qr_left2right(_qubits)
	algorithm.svd_right2left(_qubits, _chi=_chi)
	# layer4
	tools.add_gate_truncate(_qubits, Gates.x(), [0, 2, 3])
	tools.add_gate_truncate(_qubits, Gates.h(), [1])
	return _qubits


if __name__ == '__main__':
	import numpy as np
	qnumber = 2
	qubits = tools.create_ket0Series(qnumber)
	# Apply hardmard gate
	tools.add_gate_truncate(qubits, TensorGate().x(), [0])
	tools.add_gate_truncate(qubits, TensorGate().x(), [0])

	# print(qubits)
	result = tools.contract_mps(qubits)
	print(tc.reshape(result.tensor, (2 ** qnumber, 1)))
