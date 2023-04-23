"""
Author: weiguo_ma
Time: 04.20.2023
Contact: weiguo.m@iphy.ac.cn
"""
from basic_gates import TensorGate
from Ising_model import ham_near_matrix
from basic_operations import tensorDot, tc_expect
import tools
import algorithm
import torch as tc
import tensornetwork as tn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

tn.set_default_backend("pytorch")
Gates = TensorGate()

# Parameters
qnumber = 3
lr, it_time = 1e-3, 10

# Basic coefficients
theta = tc.tensor(0. + 0.j)
gamma = tc.tensor(0. + 0.j, requires_grad=True)
beta = tc.tensor(0. + 0.j, requires_grad=True)

# Hamiltonian
hamiltonian = ham_near_matrix(qnumber)

# Optimizer
optimizer = Adam([gamma, beta], lr=lr)
loss_rec = tc.zeros(it_time, )


# --------------------------------------------  Quantum Circuit
def QAOA_circ(_qnumber, _theta, _gamma, _beta):
	_qubits = tools.create_ket0Series(_qnumber)

	for _i in range(len(_qubits)):
		_qubits[_i].tensor.requires_grad = True

	# Apply hardmard gate
	tools.add_gate(_qubits, Gates.h(), [0, 2])
	tools.add_gate(_qubits, Gates.ry(_theta), [1])

	# Apply ZZ gate
	tools.add_gate(_qubits, Gates.rzz(_gamma), [0, 1])
	tools.add_gate(_qubits, Gates.rzz(_gamma), [1, 2])

	# Apply RX gate
	tools.add_gate(_qubits, Gates.rx(_beta), [0, 2])

	# Gate state vector
	_state = tc.reshape(tools.contract_mps(_qubits).tensor, (2**qnumber, 1))
	return _state


# print('Start Optimizing...')
for t in range(it_time):
	state = QAOA_circ(qnumber, theta, gamma, beta)
	result_expect = tc_expect(hamiltonian, state)
	print(result_expect.requires_grad)
	result_expect.backward()
	optimizer.step()
	optimizer.zero_grad()
	loss_rec[t] = result_expect.item()
	if t % 20 == 19:
		print('Epoch={:d}, Loss={:4f}'.format(t, result_expect.item()))


plt.plot(loss_rec)
plt.xlabel('iteration time')
plt.ylabel('loss/expectation')
plt.show()
tc.linalg.svd()