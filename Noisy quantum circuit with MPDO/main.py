"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
import QNodes
from Library.basic_gates import TensorGate
import Library.tools as tools
import Library.noise_channel as noise_channel
import torch as tc

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # I fixed this bug in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

# qnumber = 2
# qubits = QNodes.ghzLike_nodes(qnumber, noise=True)
# qubits = QNodes.used4test()

# qubits = tools.create_ket0Series(qnumber)
# node, dm = tools.calculate_DM(qubits, noisy=True)
# print(dm)
# prob = tools.density2prob(dm)
# tools.plot_histogram(prob)

# print(result.tensor.reshape(2 ** qnumber, 2 ** qnumber))
#
# print(qubits)
# result = tools.contract_mps(qubits).tensor.reshape(2 ** qnumber, 1)
# print(result)
# dm = tc.einsum('ij, kj -> ik', result, result.conj())
# prob = tools.density2prob(dm)
# tools.plot_histogram(prob)

# idx_list = []
# for i, value in enumerate(result):
# 	if tc.abs(value) > 1e-5:
# 		idx_list.append(i)
# print('---------- split line in main.py ----------')
# print(idx_list)



# Test NOISY CHANNEL
qnumber = 2
Gates = TensorGate()
qubits = tools.create_ket_hadamardSeries(qnumber)
tools.add_gate(qubits, Gates.cnot(), [0, 1])
noise_channel.apply_noise_channel(qubits, [0], noise_type='depolarization', p=1e-2)
noise_channel.apply_noise_channel(qubits, [0], noise_type='amplitude_phase_damping_error',
                                           time=30, T1=2e2, T2=2e1)

_, result = tools.calculate_DM(qubits, noisy=True, reduced_index=[1])
print(result)