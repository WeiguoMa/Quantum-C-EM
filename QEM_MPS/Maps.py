"""
Author: weiguo_ma
Time: 05.19.2023
Contact: weiguo.m@iphy.ac.cn
"""
import torch as tc
import copy
import tensornetwork as tn
from Library.tools import EdgeName2AxisName


class Maps(object):
	def __init__(self, superOperatorMPO: list[tn.AbstractNode]):
		self.uMPO_Ori = superOperatorMPO
		self.uPMPO_Ori = self._uPMPO(superOperatorMPO)

		self.uDMPO_Ori = self._getConjugate(self.uMPO_Ori)
		self.uPDMPO_Ori = self._getConjugate(self.uPMPO_Ori)

		self.MMap = None
		self.NMAP = None

	@staticmethod
	def _uPMPO(_uMPO: list[tn.AbstractNode]) -> list[tn.AbstractNode]:
		""" get the uPMPO from uMPO-shapeLike"""
		def __substitute2RandomTensors(_uMPO_: list[tn.AbstractNode]):
			""" substitute the UTensorMPO with random tensors """
			for _node in _uMPO_:
				_shape = _node.shape
				_node.tensor = tc.randn(size=_shape)

		# Process the uMPO -> uPMPO
		__substitute2RandomTensors(_uMPO)
		return _uMPO

	@staticmethod
	def _getConjugate(MPO: list[tn.AbstractNode]) -> list[tn.AbstractNode]:
		""" get the conjugate of MPOs """
		MPO = copy.deepcopy(MPO)
		for _node in MPO:
			_node.tensor = _node.tensor.conj()
			_node.name = _node.name + '_conj'
			for _name in _node.axis_names:
				if 'bond_' in _name:
					_node.name = _node.name + '_conj'
		return EdgeName2AxisName(MPO)

	def MMap(self) -> dict:
		uMPO, uPMPO = copy.deepcopy(self.uMPO_Ori), copy.deepcopy(self.uPMPO_Ori)
		uDMPO, uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(len(uMPO)):
			# Connect u -> u^\dagger
			tn.connect(uMPO[_num][f'inner_{_num}'], uPMPO[_num][f'physics_{_num}'], name=f'u_uD_{_num}')
			# Connect u' -> u
			tn.connect(uPMPO[_num][f'inner_{_num}'], uMPO[_num][f'physics_{_num}'], name=f'uP_u_{_num}')
			# Connect u^\dagger -> u'^\dagger
			tn.connect(uDMPO[_num][f'inner_{_num}'], uPDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(uPMPO[_num][f'physics_{_num}'], uPDMPO[_num][f'inner_{_num}'], name=f'uP_uPD_tr_{_num}')

		MMAP = {'uMPO': uMPO, 'uPMPO': uPMPO, 'uDMPO': uDMPO, 'uPDMPO': uPDMPO}
		self.MMap = MMAP

		return MMAP

	def NMAP(self) -> dict:
		uDMPO, uPDMPO = copy.deepcopy(self.uDMPO_Ori), copy.deepcopy(self.uPDMPO_Ori)

		for _num in range(len(uDMPO)):
			# Connect u^\dagger -> u'^\dagger
			tn.connect(uDMPO[_num][f'inner_{_num}'], uPDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_{_num}')
			# Connect u' -> u'^\dagger // TRACE Edge
			tn.connect(uPDMPO[_num][f'inner_{_num}'], uDMPO[_num][f'physics_{_num}'], name=f'uD_uPD_tr_{_num}')

		NMAP = {'uDMPO': uDMPO, 'uPDMPO': uPDMPO}
		self.NMAP = NMAP

		return NMAP

	def updateMAPNode(self):
		pass