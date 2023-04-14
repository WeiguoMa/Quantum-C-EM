"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
from basic_gates import *
import tools
import tensornetwork as tn
import algorithm

tn.set_default_backend("pytorch")

def ghzLike_edges(_qnumber):
	"""
	ghzLike state preparation with edges.
	:param _qnumber: Edge number of the state;
	:return: Edge list of the state after preparation;
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
	"""
	ghzLike state preparation with nodes.
	:param _qnumber: Node number of the state;
	:param _chi: Maximum bond dimension of the state;
	:return: Node list of the state after preparation;
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

def scalable_simulation_scheme2(_theta: float):
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
