# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimator,BackendSampler

# General imports
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_distribution

# Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler, Session, Options

# SciPy minimizer routine
from scipy.optimize import minimize
from circuit_knitting.cutting.cutqc import cut_circuit_wires
# rustworkx graph library
import rustworkx as rx
from rustworkx.visualization import mpl_draw
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401
from circuit_knitting.cutting.cutqc import evaluate_subcircuits
# Use local versions of the primitives by default.
service = None
from circuit_knitting.cutting.cutqc import (
    reconstruct_full_distribution,
)
from qiskit.result import ProbDistribution
# Set the Sampler and runtime options




def function(quantumcircuit,parameters,optimizer,max_cuts,max_subcircuit_width,num_subcircuit):
    qc = transpile(quantumcircuit,  basis_gates=['u3', 'cx'], optimization_level=3)
    qc= qc.assign_parameters(parameters)
    cuts = cut_circuit_wires(circuit = qc, method="automatic",
                             max_subcircuit_width=max_subcircuit_width,
                             max_cuts=max_cuts,
                             num_subcircuit=num_subcircuit)
    
    num_qubits = qc.num_qubits
    
    options = Options(execution={"shots": 1000})

    backend_names = ['ibmq_qasm_simulator']* cuts['num_subcircuits']

    subcircuit_instance_probabilities = evaluate_subcircuits(cuts)

    reconstructed_probabilities = reconstruct_full_distribution(
    qc, subcircuit_instance_probabilities, cuts)

    reconstructed_distribution = {
    i: prob for i, prob in enumerate(reconstructed_probabilities)}

    # Represent states as bitstrings (instead of ints)
    reconstructed_dict_bitstring = ProbDistribution(
        data=reconstructed_distribution
    ).binary_probabilities(num_bits=num_qubits)

    
