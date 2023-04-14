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

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # I fixed this bug in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

qnumber = 7
# qubits = QNodes.ghzLike_nodes(qnumber)


qubits = QNodes.scalable_simulation_scheme2(np.pi/2)

result = tools.contract_mps(qubits)
print(tc.reshape(result.tensor, (2 ** qnumber, 1)))
print(2 ** qnumber)

# Contract in different chi, like chi = [1, 2, None], which spends different time as [13.757s, 21.24s, 27.86s]