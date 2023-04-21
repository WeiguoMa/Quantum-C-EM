"""
Author: weiguo_ma
Time: 04.21.2023
Contact: weiguo.m@iphy.ac.cn
"""
from abc import ABC
import torch as tc
import tensornetwork as tn
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Type, Union, \
	overload, Sequence, Iterable


class AbstractQubit(tn.AbstractNode, ABC):
	def __init__(self, tensor: tc.Tensor, name: str = None):
		super().__init__(tensor, name=name)
