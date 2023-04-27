"""
Author: weiguo_ma
Time: 04.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
from Library.ADGate import TensorGate
from Library.ADCircuits import TensorCircuit
from Library.Ising_model import ham_near_matrix
from Library.basic_operations import tc_expect
import Library.tools as tools
import torch as tc
import tensornetwork as tn
from torch.optim import Adam
import matplotlib.pyplot as plt

tn.set_default_backend("pytorch")
Gates = TensorGate()

# Parameters
qnumber = 3
lr, it_time = 1e-3, 50

# Basic coefficients
theta = tc.tensor(3*tc.pi/4, dtype=tc.complex128)
paras = tc.randn(2, dtype=tc.complex128, requires_grad=True)

# Hamiltonian
hamiltonian = ham_near_matrix(qnumber)

# Optimizer
optimizer = Adam([paras], lr=lr)
loss_rec = tc.zeros(it_time, )

# print('Start Optimizing...')
for t in range(it_time):

	circuit = TensorCircuit(qnumber, initState='ket0', ideal=True)

	# Apply hardmard gate
	circuit.add_gate(Gates.h(), [0, 2])
	circuit.add_gate(Gates.ry(theta), [1])

	# Apply ZZ gate
	circuit.add_gate(Gates.rzz(paras[0]), [0, 1])
	circuit.add_gate(Gates.rzz(paras[0]), [1, 2])

	# Apply RX gate
	circuit.add_gate(Gates.rx(paras[1]), [0, 2])

	dm = circuit.calculate_DM()

	expectation = tc_expect(hamiltonian, dm)
	expectation.backward()
	optimizer.step()
	optimizer.zero_grad()
	loss_rec[t] = expectation.item()
	if t % 20 == 19:
		print('Epoch={:d}, Loss={:4f}'.format(t, expectation.item()))


# plt.plot(loss_rec)
# plt.xlabel('iteration time')
# plt.ylabel('loss/expectation')
# plt.show()
