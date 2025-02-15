import ctypes
import numpy as np

class SolverInterface:
    def __init__(self):
        self.dll = ctypes.CDLL("../sparse-mixed-precision-solver/build/shared_library/Release/SolverDLL.dll")
        self._setup_function()

    def _setup_function(self):
        # Define the function signature
        self.dll.solve_iterative_refinement.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # A_data
            ctypes.POINTER(ctypes.c_int),     # A_indices
            ctypes.POINTER(ctypes.c_int),     # A_indptr
            ctypes.POINTER(ctypes.c_double),  # b
            ctypes.POINTER(ctypes.c_double),  # x (output)
            ctypes.c_int,                     # n
            ctypes.c_double,                  # tolerance
            ctypes.c_int                      # max_iterations
        ]
        self.dll.solve_iterative_refinement.restype = ctypes.c_int

    def solve(self, A_data, A_indices, A_indptr, b, n, tolerance=1e-8, max_iterations=50):
        # Convert inputs to ctypes-compatible formats
        x = np.zeros(n, dtype=np.float64)
        result = self.dll.solve_iterative_refinement(
            A_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            A_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
            tolerance,
            max_iterations
        )
        return result, x