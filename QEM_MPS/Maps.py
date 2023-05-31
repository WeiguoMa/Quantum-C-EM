"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import copy
import tensornetwork as tn
from Library.tools import EdgeName2AxisName

class UPMPO(object):
	""" get the uPMPO from uMPO-shapeLike"""
	def __init__(self, uMPO: list[tn.AbstractNode]):
		self.uPMPO = copy.deepcopy(uMPO)
		self._substitute2RandomTensors()

	def _substitute2RandomTensors(self):
		""" substitute the UTensorMPO with random tensors """
		for _i, _node in enumerate(self.uPMPO):
			_shape = _node.shape
			_node.name = f'UP_{_i}'
			_node.tensor = tc.randn(size=_shape)


class Maps(object):
	def __init__(self, superOperatorMPO: list[tn.AbstractNode], uPMPO: list[tn.AbstractNode]):
		self.uMPO_Ori = superOperatorMPO
		self.uPMPO_Ori = uPMPO

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
				if 'bond_' in _node[_i].name and 'Dagger' not in _node[_i].name:
					_node[_i].name = 'Dagger' + _node[_i].name
		EdgeName2AxisName(MPO)
		return MPO

	def MMAP(self):
		_uMPO, _uPMPO = copy.deepcopy(self.uMPO_Ori), copy.deepcopy(self.uPMPO_Ori)
		_uDMPO, _uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(len(_uMPO)):
			# Connect u -> u^\dagger
			tn.connect(_uMPO[_num][f'inner_{_num}'], _uDMPO[_num][f'physics_{_num}'], name=f'u_uD_{_num}')
			# Connect u' -> u
			tn.connect(_uPMPO[_num][f'inner_{_num}'], _uMPO[_num][f'physics_{_num}'], name=f'uP_u_{_num}')
			# Connect u^\dagger -> u'^\dagger
			tn.connect(_uDMPO[_num][f'inner_{_num}'], _uPDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(_uPMPO[_num][f'physics_{_num}'], _uPDMPO[_num][f'inner_{_num}'], name=f'uP_uPD_tr_{_num}')

		for _list in [_uMPO, _uPMPO, _uDMPO, _uPDMPO]:
			EdgeName2AxisName(_list)
		MMAP = {'uMPO': _uMPO, 'uPMPO': _uPMPO, 'uDMPO': _uDMPO, 'uPDMPO': _uPDMPO}
		self.MMap = MMAP

	def NMAP(self):
		uDMPO, uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(len(uDMPO)):
			# Connect u^\dagger -> u'^\dagger
			tn.connect(uDMPO[_num][f'inner_{_num}'], uPDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(uPDMPO[_num][f'inner_{_num}'], uDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_tr_{_num}')

		for _list in [uDMPO, uPDMPO]:
			EdgeName2AxisName(_list)
		NMAP = {'uDMPO': uDMPO, 'uPDMPO': uPDMPO}
		self.NMap = NMAP

	def updateMMap(self):
		pass

	def updateNMap(self):
		pass