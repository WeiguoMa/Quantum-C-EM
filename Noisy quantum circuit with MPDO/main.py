"""
Author: weiguo_ma
Time: 04.30.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn
import numpy as np

import Library.tools as tools
from Library.ADCircuits import TensorCircuit
from Library.AbstractGate import AbstractGate

# # Ignore warnings from tensornetwork package when using pytorch backend for svd
# warnings.filterwarnings("ignore")
# # Bugs fixed in tensornetwork package, which is 'torch.svd, torch.qr' -> 'torch.linalg.svd, torch.linalg.qr'

tn.set_default_backend("pytorch")

# Basic information of circuit
qnumber = 5
ideal_circuit = False   # or True
crossTalk = False    # or False
noiseType = 'realNoise'		# or 'realNoise' or 'idealNoise'
chiFilename = './data/chi/chi1.mat'
chi, kappa = 4, 4

# Establish a quantum circuit
circuit = TensorCircuit(qn=qnumber, ideal=ideal_circuit, noiseType=noiseType,
                        chiFilename=chiFilename, crossTalk=crossTalk, chi=chi, kappa=kappa, chip='worst4Test')

circuit.add_gate(AbstractGate().h(), [0])
circuit.add_gate(AbstractGate().cnot(), [0, 1])
circuit.add_gate(AbstractGate().cnot(), [1, 2])
circuit.add_gate(AbstractGate().cnot(), [2, 3])
circuit.add_gate(AbstractGate(_lastTrunc=True).cnot(), [3, 4])

print(circuit)


# Generate an initial quantum state
state = tools.create_ket0Series(qnumber)
state = circuit(state, state_vector=False, reduced_index=[])

# Calculate probability distribution
prob_dict = tools.density2prob(state, tolerant=5e-4)

# plot probability distribution
tools.plot_histogram(prob_dict, title=f'"{noiseType}" Probability Distribution')