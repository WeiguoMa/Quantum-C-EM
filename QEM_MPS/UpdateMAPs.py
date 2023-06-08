"""
Author: weiguo_ma
Time: 05.31.2023
Contact: weiguo.m@iphy.ac.cn
"""
import copy

from Maps import Maps, UPMPO
from superOperator import SuperOperator
from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate

import numpy as np
import torch as tc
import tensornetwork as tn


tn.set_default_backend("pytorch")


class UpdateNODES(object):
	def __init__(self, uMPO: SuperOperator = None, uPMPO: UPMPO = None, epoch: int = 1):
		if uMPO is None:
			uMPO = SuperOperator().superOperatorMPO
		if uPMPO is None:
			uPMPO = UPMPO(uMPO=uMPO).uPMPO

		self.uMPO = uMPO
		self.uPMPO = uPMPO
		self.qubitNum = int(len(self.uMPO) / 2)
		self.maps = Maps(superOperatorMPO=uMPO, uPMPO=uPMPO)

		self.MInverseMatrix = None
		self.NMatrix = None
		self.orderMoveN = None
		self.NNodeShape = None
		self.nameOrderN = None

		self.bTensor = None

		self.updateProcess(epoch=epoch)

	@staticmethod
	def findElementsInListWithOrder(inList, orders: list[str]):
		idxList = []
		for element in orders:
			idxList.append(inList.index(element))
		return idxList

	def _reshapeTensor2Matrix(self, _Node: tn.AbstractNode, _nameOrder: list[str], _method: str):
		_len = len(_nameOrder)
		_order = self.findElementsInListWithOrder(_Node.axis_names, _nameOrder)
		_randomString = generate_random_string_without_duplicate(_len)
		_reorderString = ''.join([_randomString[_i] for _i in _order])

		_Tensor = tc.einsum(f'{_randomString} -> {_reorderString}', _Node.tensor)
		if _method == 'M':
			_matrix = _Tensor.reshape((np.prod(_Tensor.shape[:int(_len/2)]), np.prod(_Tensor.shape[int(_len/2):])))
		elif _method == 'N':
			self.NNodeShape = _Tensor.shape
			_matrix = _Tensor.reshape((np.prod(_Tensor.shape[:_len-1]), np.prod(_Tensor.shape[-1])))
		else:
			raise TypeError('Invalid method, interior method: M, N')
		return _matrix

	def contractMMap(self, BNum: int):
		""" contract the MMap """
		contractorMDict, contractorM = copy.deepcopy(self.maps.MMap), []
		contractorMDict['uPMPO'] = [_node for _i, _node in enumerate(contractorMDict['uPMPO']) if _i != BNum]
		contractorMDict['uPDMPO'] = [_node for _i, _node in enumerate(contractorMDict['uPDMPO']) if _i != BNum]

		for _value in contractorMDict.values():
			contractorM += _value
		#
		MNode = tn.contractors.auto(contractorM, ignore_edge_order=True)
		EdgeName2AxisName(MNode)
		print(MNode.axis_names)
		#
		if BNum == 0:     # HARD-CODE Designed
			nameOrder = [f'DuD_uPD_{BNum}', f'DaggerDbond_{BNum}_{BNum+1}', f'DuP_u_{BNum}', f'Dbond_{BNum}_{BNum+1}']
		elif BNum == self.uMPO.__len__ - 1:
			nameOrder = [f'uD_uPD_{BNum}', f'Daggerbond_{BNum-1}_{BNum}', f'uP_u_{BNum}', f'bond_{BNum-1}_{BNum}']
		else:
			if BNum < self.qubitNum:
				nameOrder = [f'DaggerDbond_{BNum-1}_{BNum}', f'DuD_uPD_{BNum}', f'DaggerDbond_{BNum}_{BNum+1}',
				             f'Dbond_{BNum-1}_{BNum}', f'DuP_u_{BNum}', f'Dbond_{BNum}_{BNum+1}']
			else:
				nameOrder = [f'Daggerbond_{BNum - 1}_{BNum}', f'uD_uPD_{BNum}', f'Daggerbond_{BNum}_{BNum + 1}',
				             f'bond_{BNum - 1}_{BNum}', f'uP_u_{BNum}', f'bond_{BNum}_{BNum + 1}']
		_MMatrix = self._reshapeTensor2Matrix(_Node=MNode, _nameOrder=nameOrder, _method='M')
		_MInverseMatrix = tc.linalg.inv(_MMatrix)

		self.MInverseMatrix = _MInverseMatrix
		return _MInverseMatrix

	def contractNMap(self, BNum: int):
		""" Contract the NMap """
		contractorNDict, contractorN = copy.deepcopy(self.maps.NMap), []
		contractorNDict['uPDMPO'] = [_node for _i, _node in enumerate(contractorNDict['uPDMPO']) if _i != BNum]

		for _value in contractorNDict.values():
			contractorN += _value
		#
		NNode = tn.contractors.auto(contractorN, ignore_edge_order=True)
		EdgeName2AxisName(NNode)
		#
		if BNum == 0:     # HARD-CODE Designed
			self.nameOrderN = [f'uD_uPD_{BNum}', f'Daggerbond_{BNum}_{BNum+1}', f'uD_uPD_tr_{BNum}']
		elif BNum == len(self.uMPO) - 1:
			self.nameOrderN = [f'uD_uPD_{BNum}', f'Daggerbond_{BNum-1}_{BNum}', f'uD_uPD_tr_{BNum}']
		else:
			self.nameOrderN = [f'Daggerbond_{BNum-1}_{BNum}', f'uD_uPD_{BNum}', f'Daggerbond_{BNum}_{BNum+1}', f'uD_uPD_tr_{BNum}']

		_NMatrix = self._reshapeTensor2Matrix(_Node=NNode, _nameOrder=self.nameOrderN, _method='N')

		self.NMatrix = _NMatrix
		return _NMatrix

	def calculateTensorB(self, BNum: int):
		""" Calculate BMap """
		self.contractMMap(BNum=BNum), self.contractNMap(BNum=BNum)
		_bTensor = tc.matmul(self.MInverseMatrix, self.NMatrix).reshape(self.NNodeShape)
		self.bTensor = _bTensor
		return _bTensor

	def updateProcess(self, epoch: int):
		for _epoch in range(epoch):
			for BNum in range(len(self.uMPO)):
				_bTensor = self.calculateTensorB(BNum=BNum)
				self.maps.updateMMap(bTensor=_bTensor, BNum=BNum, fixedNameOrder=self.nameOrderN)
				self.maps.updateNMap(bTensor=_bTensor, BNum=BNum, fixedNameOrder=self.nameOrderN)


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

	uMPO = SuperOperator(realNoiseTensor).superOperatorMPO
	uPMPO = UPMPO(uMPO=uMPO).uPMPO

	update = UpdateNODES(uMPO=uMPO, uPMPO=uPMPO, epoch=1)

	mMap = update.maps.MMap

	# # Check uP.mm(u) = I
	# IMap = copy.deepcopy(mMap)
	# del IMap['uPDMPO'], IMap['uDMPO']
	#
	# # Free left/right bond
	# for num in range(len(IMap['uMPO'])):
	# 	for name in IMap['uMPO'][num].axis_names:
	# 		if 'D' in name and 'tr' not in name:
	# 			IMap['uMPO'][num][name].disconnect(name, name)
	# 	for name in IMap['uPMPO'][num].axis_names:
	# 		if 'tr' in name:
	# 			IMap['uPMPO'][num][name].disconnect(name, name)
	#
	# contractorI = []
	# for item in IMap['uMPO']:
	# 	contractorI.append(item)
	# for item in IMap['uPMPO']:
	# 	contractorI.append(item)
	#
	# # Contract
	# INode = tn.contractors.auto(contractorI, ignore_edge_order=True)
	# EdgeName2AxisName(INode)
	# # print(INode.tensor.reshape(16, 16))
	# # print(tc.diag(INode.tensor.reshape(16, 16)))



