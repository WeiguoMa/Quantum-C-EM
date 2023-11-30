"""
Author: weiguo_ma
Time: 04.30.2023
Contact: weiguo.m@iphy.ac.cn
"""
import tensornetwork as tn

import Library.tools as tools
from Library.QuantumCircuit import TensorCircuit

tn.set_default_backend("pytorch")

# Basic information of circuit
qnumber = 5
ideal_circuit = False  # or True
noiseType = 'realNoise'  # or 'realNoise' or 'idealNoise'

chiFileNames = {
    'CZ': {'01': './data/chi/czDefault.mat', '12': './data/chi/czDefault.mat', '23': './data/chi/czDefault.mat',
           '34': './data/chi/czDefault.mat'},
    'CP': {}
}
chi, kappa = 4, 4

# Establish a quantum circuit
circuit = TensorCircuit(qn=qnumber, ideal=ideal_circuit, noiseType=noiseType,
                        chiFileDict=chiFileNames, chi=chi, kappa=kappa, chip='worst4Test', device='cpu')

circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.cnot(2, 3)
circuit.cnot(3, 4)

# Set TensorNetwork Truncation
circuit.truncate()

print(circuit)

# Generate an initial quantum state
state = tools.create_ket0Series(qnumber)
state = circuit(state, state_vector=False, reduced_index=[])

# Calculate probability distribution
prob_dict = tools.density2prob(state, tol=5e-4)  # Set _dict=False to return a np.array

# plot probability distribution
tools.plot_histogram(prob_dict, title=f'"{noiseType}" Probability Distribution')
