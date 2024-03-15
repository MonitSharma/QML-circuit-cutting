# useful additional packages
import matplotlib.pyplot as plt
import networkx as nx
import time
from qiskit_optimization.applications import Maxcut
# General imports
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz

from qiskit import transpile
from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_distribution
#from qiskit.quantum_info import SparsePauliOp
# Qiskit Runtime
# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_ibm_runtime import Estimator, Sampler, Session, Options

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.quantum_info import PauliList
from circuit_knitting.cutting import partition_problem
from circuit_knitting.cutting import generate_cutting_experiments
from qiskit_aer.primitives import Sampler

from circuit_knitting.cutting import reconstruct_expectation_values


def get_expectation(circuit,str,observable):



    def execute_circ(params):
        circuit_basis = transpile(circuit, basis_gates=['u3', 'cx'], optimization_level=3)
        
        circuit_basis = circuit_basis.assign_parameters(params)
        partitioned_problem = partition_problem(
        circuit=circuit_basis, partition_labels=str, observables=observable
        )
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables
        bases = partitioned_problem.bases

        subexperiments, coefficients = generate_cutting_experiments(circuits=subcircuits, observables=subobservables, num_samples=np.inf)

        # Set up a Qiskit Aer Sampler primitive for each circuit partition
        samplers = {
            label: Sampler(run_options={"shots": 2000}) for label in subexperiments.keys()
        }

        # Retrieve results from each partition's subexperiments
        results = {
            label: sampler.run(subexperiments[label]).result()
            for label, sampler in samplers.items()
        }

        reconstructed_expvals = reconstruct_expectation_values(
            results,
            coefficients,
            subobservables,
        )
            
        return reconstructed_expvals[0]
    return execute_circ
    





