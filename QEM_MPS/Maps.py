"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import copy
import tensornetwork as tn
from superOperator import SuperOperator
from Library.tools import EdgeName2AxisName, generate_random_string_without_duplicate

tn.set_default_backend("pytorch")


class UPMPO(object):
	""" get the uPMPO from uMPO-shapeLike"""

	def __init__(self, uMPO: list[tn.AbstractNode] or SuperOperator):
		self.uPMPO = copy.deepcopy(uMPO)
		self._substitute2RandomTensors()

	def _substitute2RandomTensors(self):
		""" substitute the UTensorMPO with random tensors """
		for _i, _node in enumerate(self.uPMPO):
			_shape = _node.shape
			_node.name = _node.name.replace('U', 'UP')
			_node.tensor = tc.randn(size=_shape, dtype=tc.complex128)


class Maps(object):
	def __init__(self, superOperatorMPO: list[tn.AbstractNode], uPMPO: list[tn.AbstractNode]):
		self.uMPO_Ori = superOperatorMPO
		self.uPMPO_Ori = uPMPO
		self.qubitNum = int(len(self.uMPO_Ori) / 2)

		self.uDMPO_Ori = self._getConjugate(self.uMPO_Ori)
		self.uPDMPO_Ori = self._getConjugate(self.uPMPO_Ori)

		self.MMap, self.NMap = None, None

		self.MMAP()
		self.NMAP()

	@staticmethod
	def _getConjugate(MPO: list[tn.AbstractNode]) -> list[tn.AbstractNode]:
		""" get the conjugate of MPOs """
		MPO = copy.deepcopy(MPO)
		for _node in MPO:
			_node.tensor = _node.tensor.conj()
			_node.name = 'Dagger' + _node.name
			for _i in range(len(_node.axis_names)):
				if 'bond' in _node[_i].name and 'Dagger' not in _node[_i].name:
					_node[_i].name = 'Dagger' + _node[_i].name
		EdgeName2AxisName(MPO)
		return MPO

	@staticmethod
	def findElementsInListWithOrder(inList, orders: list[str]):
		idxList = []
		for element in orders:
			idxList.append(inList.index(element))
		return idxList

	def _reorderTensor(self, tensor: tc.Tensor, _NodeAxisName: list[str], _order: list[str]) -> tc.Tensor:
		for _i, _name in enumerate(_NodeAxisName):
			if 'bond' in _name and 'Dagger' not in _name:
				_NodeAxisName[_i] = 'Dagger' + _NodeAxisName[_i]
			elif 'tr' in _name:
				_NodeAxisName[_i] = [_name_ for _name_ in _order if 'tr' in _name_][0]  # Hard-fix may cause problem
			elif 'tr' not in _name and 'bond' not in _name:
				_NodeAxisName[_i] = [_name_ for _name_ in _order if 'tr' not in _name_ and 'bond' not in _name_][0]
			else:
				pass

		_order = self.findElementsInListWithOrder(_order, _NodeAxisName)
		_randomString = generate_random_string_without_duplicate(len(_order))
		_reorderString = ''.join([_randomString[_i] for _i in _order])

		_tensor = tc.einsum(f'{_randomString} -> {_reorderString}', tensor)

		return _tensor

	def MMAP(self):
		_uMPO, _uPMPO = copy.deepcopy(self.uMPO_Ori), copy.deepcopy(self.uPMPO_Ori)
		_uDMPO, _uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(self.qubitNum):
			# Connect u -> u^\dagger -- D
			tn.connect(_uMPO[_num][f'Dinner_{_num}'], _uDMPO[_num][f'Dinner_{_num}'], name=f'Du_uD_{_num}')
			# Connect u' -> u -- D
			tn.connect(_uPMPO[_num][f'Dinner_{_num}'], _uMPO[_num][f'Dphysics_{_num}'], name=f'DuP_u_{_num}')
			# Connect u^\dagger -> u'^\dagger -- D
			tn.connect(_uDMPO[_num][f'Dphysics_{_num}'], _uPDMPO[_num][f'Dinner_{_num}'], name=f'DuD_uPD_{_num}')
			# Connect u' -> u'^\dagger -- D // TRACE Edge
			tn.connect(_uPMPO[_num][f'Dphysics_{_num}'], _uPDMPO[_num][f'Dphysics_{_num}'], name=f'DuP_uPD_tr_{_num}')

		for _num in range(self.qubitNum):
			_num_ = self.qubitNum + _num
			# Connect u -> u^\dagger
			tn.connect(_uMPO[_num_][f'inner_{_num}'], _uDMPO[_num_][f'inner_{_num}'], name=f'u_uD_{_num}')
			# Connect u' -> u
			tn.connect(_uPMPO[_num_][f'inner_{_num}'], _uMPO[_num_][f'physics_{_num}'], name=f'uP_u_{_num}')
			# Connect u^\dagger -> u'^\dagger
			tn.connect(_uDMPO[_num_][f'physics_{_num}'], _uPDMPO[_num_][f'inner_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(_uPMPO[_num_][f'physics_{_num}'], _uPDMPO[_num_][f'physics_{_num}'], name=f'uP_uPD_tr_{_num}')

		for _list in [_uMPO, _uPMPO, _uDMPO, _uPDMPO]:
			EdgeName2AxisName(_list)
		MMAP = {'uMPO': _uMPO, 'uPMPO': _uPMPO, 'uDMPO': _uDMPO, 'uPDMPO': _uPDMPO}
		self.MMap = MMAP

	def NMAP(self):
		uDMPO, uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(self.qubitNum):
			# Connect u^\dagger -> u'^\dagger -- D
			tn.connect(uDMPO[_num][f'Dphysics_{_num}'], uPDMPO[_num][f'Dinner_{_num}'], name=f'DuD_uPD_{_num}')
			# Connect u' -> u'^\dagger -- D // TRACE Edge
			tn.connect(uDMPO[_num][f'Dinner_{_num}'], uPDMPO[_num][f'Dphysics_{_num}'], name=f'DuD_uPD_tr_{_num}')

		for _num in range(self.qubitNum):
			_num_ = self.qubitNum + _num
			# Connect u^\dagger -> u'^\dagger
			tn.connect(uDMPO[_num_][f'physics_{_num}'], uPDMPO[_num_][f'inner_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(uDMPO[_num_][f'inner_{_num}'], uPDMPO[_num_][f'physics_{_num}'], name=f'uD_uPD_tr_{_num}')

		for _list in [uDMPO, uPDMPO]:
			EdgeName2AxisName(_list)
		NMAP = {'uDMPO': uDMPO, 'uPDMPO': uPDMPO}
		self.NMap = NMAP

	def updateMMap(self, bTensor: tc.Tensor, BNum: int, fixedNameOrder: list[str]):
		self.MMap['uPMPO'][BNum].tensor = self._reorderTensor(bTensor, self.MMap['uPMPO'][BNum].axis_names,
		                                                      fixedNameOrder)
		self.MMap['uPDMPO'][BNum].tensor = self._reorderTensor(bTensor.conj(), self.MMap['uPDMPO'][BNum].axis_names,
		                                                       fixedNameOrder)
		EdgeName2AxisName([self.MMap['uPMPO'][BNum], self.MMap['uPDMPO'][BNum]])    # Hard-reset for fixing prob in _re

	def updateNMap(self, bTensor: tc.Tensor, BNum: int, fixedNameOrder: list[str]):
		self.NMap['uPDMPO'][BNum].tensor = self._reorderTensor(bTensor.conj(), self.NMap['uPDMPO'][BNum].axis_names,
		                                                       fixedNameOrder)
		EdgeName2AxisName([self.NMap['uPDMPO'][BNum]])  # Hard-reset for fixing prob in _reorderTensor


if __name__ == '__main__':
	from Library.realNoise import czExp_channel

	realNoiseTensor = czExp_channel(
		'/Users/weiguo_ma/Python_Program/Quantum_error_mitigation/Noisy quantum circuit with MPDO/data/chi/chi1.mat')

	from Library.NoiseChannel import NoiseChannel
	from Library.AbstractGate import AbstractGate

	noise = NoiseChannel()
	dpcNoiseTensor = noise.dpCTensor
	apdeNoiseTensor = noise.apdeCTensor

	abX = AbstractGate().y().gate.tensor
	XTensor = tc.einsum('ijk, jl -> ilk', dpcNoiseTensor, abX)

	uMPO = SuperOperator(abX, noisy=False).superOperatorMPO
	uPMPO = UPMPO(uMPO=uMPO).uPMPO

	maps = Maps(superOperatorMPO=uMPO, uPMPO=uPMPO)

	# maps = Maps(superOperatorMPO=uMPO, uPMPO=uPMPO)
	# mMap = maps.MMap
	# nMap = maps.NMap
	#
	# print(mMap['uMPO'][0].edges)
	# print(mMap['uMPO'][0].axis_names)