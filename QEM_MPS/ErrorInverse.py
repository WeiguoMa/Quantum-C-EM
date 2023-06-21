"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

import torch as tc
import tensornetwork as tn
from superOperator import SuperOperator
from Library.tools import EdgeName2AxisName
from Maps import Maps
from UpdateMAPs import UpdateNODES


tn.set_default_backend("pytorch")


class ErrorInverse(object):
	def __init__(self, idealMPO: list[tn.AbstractNode] or SuperOperator, uPMPO: list[tn.AbstractNode] or SuperOperator):
		self.uMPO = idealMPO
		self.uPMPO = uPMPO

		self.qubitNum = int(len(self.uPMPO) / 2)

		self.EpsilonMinus1 = None

		self.calErrorInverse()

	def calErrorInverse(self):
		# Connection
		for _num in range(self.qubitNum):       # Dual Space
			tn.connect(self.uMPO[_num][f'Dinner_{_num}'], self.uPMPO[_num][f'Dphysics_{_num}'])
		for _num in range(self.qubitNum):
			_num_ = _num + self.qubitNum
			tn.connect(self.uMPO[_num_][f'inner_{_num}'], self.uPMPO[_num_][f'physics_{_num}'])

		# Contraction
		EpsilonMinus1 = []
		for _num in range(self.qubitNum):
			EpsilonMinus1.append(tn.contract_between(self.uMPO[_num], self.uPMPO[_num], name=f'DEpsilonMinus1_{_num}'))
		for _num in range(self.qubitNum):
			_num_ = _num + self.qubitNum
			EpsilonMinus1.append(tn.contract_between(self.uMPO[_num_], self.uPMPO[_num_], name=f'EpsilonMinus1_{_num}'))
		EdgeName2AxisName(EpsilonMinus1)

		self.EpsilonMinus1 = EpsilonMinus1
		return EpsilonMinus1


if __name__ == '__main__':
	from Library.realNoise import czExp_channel

	realNoiseTensor = czExp_channel(
		'/Users/weiguo_ma/Python_Program/Quantum_error_mitigation/Noisy quantum circuit with MPDO/data/chi/chi1.mat')

	from Library.NoiseChannel import NoiseChannel
	from Library.AbstractGate import AbstractGate

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
	uPMPO = UpdateNODES(maps=maps, epoch=2).uPMPO

	ErrorInverse = ErrorInverse(idealMPO=idealUMPO, uPMPO=uPMPO).EpsilonMinus1

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

	idealOperator = tn.contractors.auto(contractor, output_edge_order=[contractor[0]['Dphysics_0'], contractor[1]['Dphysics_1'],
	                                                                   contractor[2]['physics_0'], contractor[3]['physics_1'],
	                                                                   contractor[4]['Dinner_0'], contractor[5]['Dinner_1'],
	                                                                   contractor[6]['inner_0'], contractor[7]['inner_1']])
	idealUMPO2 = tn.contractors.auto(idealUMPO2, output_edge_order=[idealUMPO2[0]['Dphysics_0'], idealUMPO2[1]['Dphysics_1'],
	                                                                   idealUMPO2[2]['physics_0'], idealUMPO2[3]['physics_1'],
	                                                                   idealUMPO2[0]['Dinner_0'], idealUMPO2[1]['Dinner_1'],
	                                                                   idealUMPO2[2]['inner_0'], idealUMPO2[3]['inner_1']])
	EdgeName2AxisName(idealUMPO2)

	check_matrixSame(idealOperator.tensor, idealUMPO2.tensor)
	print(tc.diag(idealOperator.tensor.reshape(16, 16)))
	print(tc.diag(idealUMPO2.tensor.reshape(16, 16)))