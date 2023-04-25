"""
Author: weiguo_ma
Time: 04.20.2023
Contact: weiguo.m@iphy.ac.cn
"""
from Library.ADGate import TensorGate
from Library.Ising_model import ham_near_matrix
from Library.basic_operations import tc_expect
import Library.tools as tools
import torch as tc
import tensornetwork as tn
from torch.optim import Adam

tn.set_default_backend("pytorch")
Gates = TensorGate()

# Parameters
qnumber = 3
lr, it_time = 1e-3, 50

# Basic coefficients
theta = tc.tensor(3*tc.pi/4, dtype=tc.complex128)
gamma = tc.tensor(tc.pi, requires_grad=True, dtype=tc.complex128)
beta = tc.tensor(tc.pi, requires_grad=True, dtype=tc.complex128)

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
	print(gamma.grad)
	result_expect = tc_expect(hamiltonian, state)
	result_expect.backward()
	optimizer.step()
	optimizer.zero_grad()
	loss_rec[t] = result_expect.item()
	if t % 20 == 19:
		print('Epoch={:d}, Loss={:4f}'.format(t, result_expect.item()))


# plt.plot(loss_rec)
# plt.xlabel('iteration time')
# plt.ylabel('loss/expectation')
# plt.show()
