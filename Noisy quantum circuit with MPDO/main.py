"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
import QNodes
import Library.tools as tools
import torch as tc

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # I fixed this bug in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

qnumber = 2
qubits = QNodes.ghzLike_nodes(qnumber, noise=True)
# qubits = QNodes.used4test()

# qubits = tools.create_ket0Series(qnumber)
node, dm = tools.calculate_DM(qubits, noisy=True)
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



# # Test NOISY CHANNEL
# qnumber = 5
# qubits = tools.create_ket_hadamardSeries(qnumber)
# noise_channel.apply_noise_channel(qubits, [1, 3], _noise_type='depolarization', _p=1e-2)
# noise_channel.apply_noise_channel(qubits, [1, 3], _noise_type='amplitude_phase_damping_error',
#                                            _time=30, _T1=2e2, _T2=2e1)
# # print('----------------------------')
#
# result0 = tc.einsum('ij, kj -> ik', qubits[1].tensor, qubits[1].tensor.conj())
# print(qubits[1].tensor)
# # print('result0:', result0)
# # result1 = tc.einsum('ij, kj -> ik', qubits[3].tensor, qubits[3].tensor.conj())
# # print('result1', result1)