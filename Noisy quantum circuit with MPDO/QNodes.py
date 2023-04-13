"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
from basic_gates import *
import tools
import tensornetwork as tn

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
		state_nodes = tools.create_init_mpo(_qnumber)
		# which generated a list of two rank-1 tensors representing the initial state |00>
		_qubits = [node[0] for node in state_nodes]  # every is an edge of tensor
		# Apply the Hadamard gate to the first qubit
		tools.add_gate(_qubits, Gates.h(), [0])
		# Apply the CNOT gate to the first and second qubits
		for i in range(_qnumber - 1):
			tools.add_gate(_qubits, Gates.cnot(), [i, i + 1])
	return all_nodes, _qubits

def ghzLike_nodes(_qnumber, chi: int = None):
	"""
	ghzLike state preparation with nodes.
	:param _qnumber: Node number of the state;
	:param chi: Maximum bond dimension of the state;
	:return: Node list of the state after preparation;
	"""
	Gates = TensorGate()
	_qubits = tools.create_init_mpo(_qnumber)
	# Apply hardmard gate
	tools.add_gate_truncate(_qubits, Gates.h(), [0])
	# Apply CNOT gate
	for i in range(_qnumber - 1):
		tools.add_gate_truncate(_qubits, Gates.cnot(), [i, i + 1])
	# Optimization
	tools.qr_left2right(_qubits)
	tools.svd_right2left(_qubits, chi=chi)
	return _qubits


if __name__ == '__main__':
	qnumber = 10
	nodes, qubits = ghzLike_edges(qnumber)
	result = tn.contractors.optimal(
		nodes, output_edge_order=qubits
	)
	print(tc.reshape(result.tensor, (2 ** qnumber, 1)))
	print(2 ** qnumber)