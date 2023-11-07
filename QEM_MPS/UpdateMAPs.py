"""
Author: weiguo_ma
Time: 05.31.2023
Contact: weiguo.m@iphy.ac.cn
"""
from copy import deepcopy
from typing import List

import numpy as np
import tensornetwork as tn
import torch as tc
from tqdm import tqdm

from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate
from Maps import Maps

tn.set_default_backend("pytorch")


class UpdateNODES:
	def __init__(self, maps: Maps = None, epoch: int = 1):
		"""
		    Manages the update process of tensor nodes in a tensor network,
		    performing contractions, reshaping tensors to matrices, and updating the network.
	    """
		self.maps = maps
		self.qubitNum = self.maps.qubitNum

		self.MMatrix = None
		self.NMatrix = None
		self.orderMoveN = None
		self.NNodeShape = None
		self.nameOrderN = None

		self.bTensor = None

		self.updateProcess(epoch=epoch)

		self.uPMPO = self.getUPMPO()

	@staticmethod
	def findElementsInListWithOrder(inList, orders: List[str]):
		return [inList.index(element) for element in orders]

	def _reshapeTensor2Matrix(self, _Node: tn.AbstractNode, _nameOrder: List[str], _method: str):
		_len = len(_nameOrder)
		_order = self.findElementsInListWithOrder(_Node.axis_names, _nameOrder)
		_randomString = generate_random_string_without_duplicate(_len)
		_reorderString = ''.join([_randomString[_i] for _i in _order])

		_Tensor = tc.einsum(f'{_randomString} -> {_reorderString}', _Node.tensor)
		if _method == 'M':
			_matrix = _Tensor.reshape((np.prod(_Tensor.shape[:int(_len / 2)]), np.prod(_Tensor.shape[int(_len / 2):])))
		elif _method == 'N':
			self.NNodeShape = _Tensor.shape
			_matrix = _Tensor.reshape((np.prod(_Tensor.shape[:_len - 1]), np.prod(_Tensor.shape[-1])))
		else:
			raise TypeError('Invalid method, interior method: M, N')
		return _matrix

	def contractMMap(self, BNum: int):
		""" contract the MMap """
		contractorMDict, contractorM = deepcopy(self.maps.MMap), []
		contractorMDict['uPMPO'] = [_node for _i, _node in enumerate(contractorMDict['uPMPO']) if _i != BNum]
		contractorMDict['uPDMPO'] = [_node for _i, _node in enumerate(contractorMDict['uPDMPO']) if _i != BNum]

		for _value in contractorMDict.values():
			contractorM += _value
		#
		MNode = tn.contractors.auto(contractorM, ignore_edge_order=True)
		EdgeName2AxisName(MNode)
		#
		if self.qubitNum == 1:
			assert BNum < 2, 'BNum should be 0 or 1'
			if BNum == 0:
				nameOrder = ['DuD_uPD_0', 'DaggerEbond'] + ['DuP_u_0', 'Ebond']
			else:
				nameOrder = ['DaggerEbond', 'uD_uPD_0'] + ['Ebond', 'uP_u_0']
		else:
			if BNum == 0:  # MPO EDGE-effect
				nameOrder = ['DuD_uPD_0', 'DaggerDbond_0_1'] + ['DuP_u_0', 'Dbond_0_1']
			elif 0 < BNum < self.qubitNum - 1:
				nameOrder = [f'DaggerDbond_{BNum - 1}_{BNum}', f'DuD_uPD_{BNum}', f'DaggerDbond_{BNum}_{BNum + 1}'] + \
				            [f'Dbond_{BNum - 1}_{BNum}', f'DuP_u_{BNum}', f'Dbond_{BNum}_{BNum + 1}']
			elif BNum == self.qubitNum - 1:
				nameOrder = [f'DaggerDbond_{BNum - 1}_{BNum}', f'DuD_uPD_{BNum}', 'DaggerEbond'] + \
				            [f'Dbond_{BNum - 1}_{BNum}', f'DuP_u_{BNum}', 'Ebond']
			elif BNum == self.qubitNum:
				_BNum = BNum - self.qubitNum
				nameOrder = ['DaggerEbond', f'uD_uPD_{_BNum}', f'Daggerbond_{_BNum}_{_BNum + 1}'] + \
				            ['Ebond', f'uP_u_{_BNum}', f'bond_{_BNum}_{_BNum + 1}']
			elif 2 * self.qubitNum - 1 > BNum > self.qubitNum:
				_BNum = BNum - self.qubitNum
				nameOrder = [f'Daggerbond_{_BNum - 1}_{_BNum}', f'uD_uPD_{_BNum}', f'Daggerbond_{_BNum}_{_BNum + 1}'] + \
				            [f'bond_{_BNum - 1}_{_BNum}', f'uP_u_{_BNum}', f'bond_{_BNum}_{_BNum + 1}']
			else:
				nameOrder = [f'Daggerbond_{self.qubitNum - 2}_{self.qubitNum - 1}', f'uD_uPD_{self.qubitNum - 1}'] + \
				            [f'bond_{self.qubitNum - 2}_{self.qubitNum - 1}', f'uP_u_{self.qubitNum - 1}']
		#
		_MMatrix = self._reshapeTensor2Matrix(_Node=MNode, _nameOrder=nameOrder, _method='M')

		self.MMatrix = _MMatrix
		return _MMatrix

	def contractNMap(self, BNum: int):
		""" Contract the NMap """
		contractorNDict, contractorN = deepcopy(self.maps.NMap), []
		contractorNDict['uPDMPO'] = [_node for _i, _node in enumerate(contractorNDict['uPDMPO']) if _i != BNum]

		for _value in contractorNDict.values():
			contractorN += _value
		#
		NNode = tn.contractors.auto(contractorN, ignore_edge_order=True)
		EdgeName2AxisName(NNode)
		#
		if self.qubitNum == 1:
			assert BNum < 2, 'BNum should be 0 or 1'
			if BNum == 0:
				self.nameOrderN = ['DuD_uPD_0', 'DaggerEbond'] + ['DuD_uPD_tr_0']
			else:
				self.nameOrderN = ['DaggerEbond', 'uD_uPD_0'] + ['uD_uPD_tr_0']
		else:
			if BNum == 0:
				self.nameOrderN = ['DuD_uPD_0', 'DaggerDbond_0_1'] + ['DuD_uPD_tr_0']
			elif 0 < BNum < self.qubitNum - 1:
				self.nameOrderN = [f'DaggerDbond_{BNum - 1}_{BNum}', f'DuD_uPD_{BNum}',
				                   f'DaggerDbond_{BNum}_{BNum + 1}'] + \
				                  [f'DuD_uPD_tr_{BNum}']
			elif BNum == self.qubitNum - 1:
				self.nameOrderN = [f'DaggerDbond_{BNum - 1}_{BNum}', f'DuD_uPD_{BNum}', 'DaggerEbond'] + \
				                  [f'DuD_uPD_tr_{BNum}']
			elif BNum == self.qubitNum:
				_BNum = BNum - self.qubitNum
				self.nameOrderN = ['DaggerEbond', f'uD_uPD_{_BNum}', f'Daggerbond_{_BNum}_{_BNum + 1}'] + \
				                  [f'uD_uPD_tr_{_BNum}']
			elif 2 * self.qubitNum - 1 > BNum > self.qubitNum:
				_BNum = BNum - self.qubitNum
				self.nameOrderN = [f'Daggerbond_{_BNum - 1}_{_BNum}', f'uD_uPD_{_BNum}',
				                   f'Daggerbond_{_BNum}_{_BNum + 1}'] + \
				                  [f'uD_uPD_tr_{_BNum}']
			else:
				self.nameOrderN = [f'Daggerbond_{self.qubitNum - 2}_{self.qubitNum - 1}',
				                   f'uD_uPD_{self.qubitNum - 1}'] + \
				                  [f'uD_uPD_tr_{self.qubitNum - 1}']

		_NMatrix = self._reshapeTensor2Matrix(_Node=NNode, _nameOrder=self.nameOrderN, _method='N')

		self.NMatrix = _NMatrix
		return _NMatrix

	def calculateTensorB(self, BNum: int):
		""" Calculate BMap """
		self.contractMMap(BNum=BNum), self.contractNMap(BNum=BNum)
		_bTensor = tc.linalg.solve(self.MMatrix, self.NMatrix).reshape(self.NNodeShape)
		self.bTensor = _bTensor
		return _bTensor

	def updateProcess(self, epoch: int):
		for _epoch in tqdm(range(epoch)):
			for BNum in range(2 * self.qubitNum):
				_bTensor = self.calculateTensorB(BNum=BNum)
				self.maps.updateMMap(bTensor=_bTensor, BNum=BNum, fixedNameOrder=self.nameOrderN)
				self.maps.updateNMap(bTensor=_bTensor, BNum=BNum, fixedNameOrder=self.nameOrderN)

	def getUPMPO(self):
		""" Get the uPDMPO """
		uPMPO = deepcopy(self.maps.MMap['uPMPO'])
		for _num in range(2 * self.qubitNum):
			if _num < self.qubitNum:
				for _name in uPMPO[_num].axis_names:
					if 'tr' in _name:
						uPMPO[_num][_name].disconnect(f'Dphysics_{_num}', f'Dphysics_{_num}')
					elif 'uP_u_' in _name:
						uPMPO[_num][_name].disconnect(f'Dinner_{_num}', f'Dinner_{_num}')
					elif 'bond' in _name and 'P' not in _name:
						uPMPO[_num][_name].name = 'P' + _name
					else:
						pass
			else:
				_num_ = _num - self.qubitNum
				for _name in uPMPO[_num].axis_names:
					if 'tr' in _name:
						uPMPO[_num][_name].disconnect(f'physics_{_num_}', f'physics_{_num_}')
					elif 'uP_u_' in _name:
						uPMPO[_num][_name].disconnect(f'inner_{_num_}', f'inner_{_num_}')
					elif 'bond' in _name and 'P' not in _name:
						uPMPO[_num][_name].name = 'P' + _name
					else:
						pass
		EdgeName2AxisName(uPMPO)
		return uPMPO