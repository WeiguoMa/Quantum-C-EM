"""
Author: weiguo_ma
Time: 04.07.2023
Contact: weiguo.m@iphy.ac.cn
"""

import time
import torch as tc
import numpy as np
import tensornetwork as tn
import QNodes
import tools
import warnings
import noise_channel

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # I fixed this bug in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

qnumber = 4
# # qubits = QNodes.ghzLike_nodes(qnumber)
# et = time.time()
#
qubits = QNodes.used4test()

result = tools.contract_mps(qubits)

print(tc.reshape(result.tensor, (2 ** qnumber, 1)))
print(2 ** qnumber)
# Contract in different chi, like chi = [1, 2, None], which spends different time as [13.757s, 21.24s, 27.86s]


# # Test NOISY CHANNEL
# qnumber = 5
# qubits = tools.create_ket_hadamardSeries(qnumber)
# noise_channel.apply_noise_channel(qubits, [1, 3], _noise_type='depolarization', _p=1e-2)
# noise_channel.apply_noise_channel(qubits, [1, 3], _noise_type='amplitude_phase_damping_error',
#                                            _time=30, _T1=2e2, _T2=2e1)
# # print('----------------------------')
#
#
# print(qubits)
#
# result0 = tc.einsum('ij, kj -> ik', qubits[1].tensor, qubits[1].tensor.conj())
# print('result0:', result0)
# result1 = tc.einsum('ij, kj -> ik', qubits[3].tensor, qubits[3].tensor.conj())
# print('result1', result1)