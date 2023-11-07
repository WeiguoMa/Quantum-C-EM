"""
Author: weiguo_ma
Time: 05.02.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import tensornetwork as tn
import torch as tc

from ErrorInverse import ErrorInverse
from Library.AbstractGate import AbstractGate
from Library.NoiseChannel import NoiseChannel
from Library.realNoise import czExp_channel
from Library.tools import EdgeName2AxisName
from Maps import Maps
from UpdateMAPs import UpdateNODES
from superOperator import SuperOperator

realNoiseTensor = czExp_channel(
	'/Users/weiguo_ma/Python_Program/Quantum_error_mitigation/Noisy quantum circuit with MPDO/data/chi/chi1.mat')

noise = NoiseChannel()
dpcNoiseTensor = noise.dpCTensor
apdeNoiseTensor = noise.apdeCTensor

abX = AbstractGate().x().gate.tensor
XTensor = tc.einsum('ijk, jl -> ilk', dpcNoiseTensor, abX)

abCZ = AbstractGate().cz().gate.tensor


def check_matrixSame(matrix1, matrix2):
	if tc.allclose(matrix1, matrix2):
		print('Same')
	else:
		print('Not Same')


# --------------------- IMPORT BASIC Info ABOVE ---------------------------------------

uMPO = SuperOperator(realNoiseTensor).superOperatorMPO
idealUMPO = SuperOperator(abCZ, noisy=False).superOperatorMPO
idealUMPO2 = copy.deepcopy(idealUMPO)

maps = Maps(superOperatorMPO=uMPO)
uPMPO = UpdateNODES(maps=maps, epoch=10).uPMPO

ErrorInverse = ErrorInverse(idealMPO=idealUMPO, uPMPO=uPMPO, chi=None).EpsilonMinus1

# --------------------- CAL ERROR INVERSE ABOVE ---------------------------------------

for num in range(2):
	tn.connect(ErrorInverse[num][f'Dinner_{num}'], uMPO[num][f'Dphysics_{num}'])
for num in range(2):
	num_ = num + 2
	tn.connect(ErrorInverse[num_][f'inner_{num}'], uMPO[num_][f'physics_{num}'])

contractor = []
for item in ErrorInverse:
	contractor.append(item)
for item in uMPO:
	contractor.append(item)

idealOperator = tn.contractors.auto(contractor,
                                    output_edge_order=[contractor[0]['Dphysics_0'], contractor[1]['Dphysics_1'],
                                                       contractor[2]['physics_0'], contractor[3]['physics_1'],
                                                       contractor[4]['Dinner_0'], contractor[5]['Dinner_1'],
                                                       contractor[6]['inner_0'], contractor[7]['inner_1']])
idealUMPO2 = tn.contractors.auto(idealUMPO2,
                                 output_edge_order=[idealUMPO2[0]['Dphysics_0'], idealUMPO2[1]['Dphysics_1'],
                                                    idealUMPO2[2]['physics_0'], idealUMPO2[3]['physics_1'],
                                                    idealUMPO2[0]['Dinner_0'], idealUMPO2[1]['Dinner_1'],
                                                    idealUMPO2[2]['inner_0'], idealUMPO2[3]['inner_1']])
EdgeName2AxisName(idealUMPO2)

check_matrixSame(idealOperator.tensor, idealUMPO2.tensor)
print(tc.diag(idealOperator.tensor.reshape(16, 16)))
print(tc.diag(idealUMPO2.tensor.reshape(16, 16)))
