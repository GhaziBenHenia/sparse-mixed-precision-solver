import numpy as np
from scipy.sparse import csr_matrix, diags, rand
from scipy.sparse.linalg import spsolve, cg
from solver_interface import SolverInterface
import timeit
import sys

# imports for other solvers
try:
    import pyamg
except ImportError:
    pyamg = None

try:
    from sklearn.linear_model import lsqr
except ImportError:
    lsqr = None

def generate_diagonally_dominant_matrix(n, density=0.01, random_seed=None):
    """Generate a diagonally dominant sparse matrix of size n x n"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    A = rand(n, n, density=density, format='csr', random_state=random_seed)
    A.data = A.data - 0.5  # Values between -0.5 and 0.5
    
    # Ensure diagonal dominance
    diagonal = np.abs(A).sum(axis=1).A1 + 1.0
    A = A + diags(diagonal - A.diagonal(), 0, shape=(n, n), format='csr')
    
    return A

def benchmark_solvers(sizes=[100, 500, 1000], density=0.01, 
                     num_trials=5, num_repeats=10, tol=1e-6):
    """Benchmark iterative refinement solver against multiple numerical solvers"""
    iterative_refinement_solver = SolverInterface()
    results = []

    # Define available solvers
    solvers = {
        'IR solver': lambda A, b: iterative_refinement_solver.solve(#IR Solver : iterative refinement solver
            A.data.astype(np.float64),
            A.indices.astype(np.int32),
            A.indptr.astype(np.int32),
            b, A.shape[0]
        ),
        'SciPy spsolve': lambda A, b: (0, spsolve(A, b)),
        'SciPy CG': lambda A, b: (0, cg(A, b, atol=tol)[0])
    }

    # optional solvers
    if pyamg is not None:
        def pyamg_solver(A, b):
            ml = pyamg.ruge_stuben_solver(A)
            return (0, ml.solve(b, tol=tol))
        solvers['PyAMG'] = pyamg_solver

    if lsqr is not None:
        solvers['sklearn LSQR'] = lambda A, b: (0, lsqr(A, b, atol=tol)[0])

    for n in sizes:
        print(f"\nBenchmarking matrix size: {n}x{n}")
        solver_results = {name: {'times': [], 'abs_errors': [], 'rel_errors': []} 
                        for name in solvers}

        for trial in range(num_trials):
            # test system
            A = generate_diagonally_dominant_matrix(n, density, trial)
            x_true = np.ones(n)
            b = A.dot(x_true)
            x_true_norm = np.linalg.norm(x_true)

            for solver_name, solver_func in solvers.items():
                try:
                    # Initial solve for error calculation
                    result_code, x_sol = solver_func(A, b)
                    if result_code != 0:
                        print(f"{solver_name} failed trial {trial+1} with code {result_code}")
                        continue

                    # Calculate both errors
                    abs_error = np.linalg.norm(x_sol - x_true)
                    rel_error = abs_error / x_true_norm if x_true_norm != 0 else float('inf')

                    # Timing with timeit.repeat
                    def time_wrapper(_solver=solver_func, _A=A, _b=b):
                        _solver(_A, _b)
                    
                    timer = timeit.repeat(
                        stmt=time_wrapper,
                        number=1,
                        repeat=num_repeats
                    )
                    elapsed_times = np.array(timer)
                    
                    # Store results
                    solver_results[solver_name]['times'].append(np.mean(elapsed_times))
                    solver_results[solver_name]['abs_errors'].append(abs_error)
                    solver_results[solver_name]['rel_errors'].append(rel_error)

                except Exception as e:
                    print(f"{solver_name} failed on size {n}: {str(e)}")
                    continue

        # Aggregate results for this matrix size
        size_results = {'n': n}
        for solver_name, data in solver_results.items():
            if data['times']:
                size_results[solver_name] = {
                    'time_mean': np.mean(data['times']),
                    'time_std': np.std(data['times']),
                    'abs_error_mean': np.mean(data['abs_errors']),
                    'abs_error_std': np.std(data['abs_errors']),
                    'rel_error_mean': np.mean(data['rel_errors']),
                    'rel_error_std': np.std(data['rel_errors'])
                }
        results.append(size_results)

        # Summary table
        print(f"\nResults for n = {n}:")
        header = f"{'Solver':<15} | {'Time (mean±std)':<25} | {'Absolute Error (mean±std)':<30} | {'Relative Error (mean±std)':<30}"
        print(header)
        print("-" * len(header))
        for name, data in size_results.items():
            if name == 'n':
                continue
            time_str = f"{data['time_mean']:.4f} ± {data['time_std']:.4f}"
            abs_str = f"{data['abs_error_mean']:.2e} ± {data['abs_error_std']:.2e}"
            rel_str = f"{data['rel_error_mean']:.2e} ± {data['rel_error_std']:.2e}"
            print(f"{name:<15} | {time_str:<25} | {abs_str:<30} | {rel_str:<30}")

    return results

def plot_results(results):
    """Generate performance comparison plots"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed - skipping plots")
        return

    # Time performance plot
    plt.figure(figsize=(12, 6))
    for solver in results[0].keys():
        if solver == 'n':
            continue
        sizes = [r['n'] for r in results]
        times = [r[solver]['time_mean'] for r in results]
        plt.plot(sizes, times, 'o-', label=solver)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Solver Performance Comparison (Time)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error plots
    error_types = ['abs', 'rel']
    titles = ['Absolute Error', 'Relative Error']
    for err_type, title in zip(error_types, titles):
        plt.figure(figsize=(12, 6))
        for solver in results[0].keys():
            if solver == 'n':
                continue
            sizes = [r['n'] for r in results]
            errors = [r[solver][f'{err_type}_error_mean'] for r in results]
            plt.plot(sizes, errors, 'o-', label=solver)
        plt.xlabel('Matrix Size')
        plt.ylabel(f'{title} (||x - x_true||{"/||x_true||" if err_type=="rel" else ""})')
        plt.title(f'Solver Accuracy Comparison ({title})')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Configuration
    matrix_sizes = [100, 200, 500, 1000, 5000]
    matrix_density = 0.01  # 1% non-zero elements
    trials = 5
    repeats = 10

    # Run benchmark
    results = benchmark_solvers(
        sizes=matrix_sizes,
        density=matrix_density,
        num_trials=trials,
        num_repeats=repeats
    )

    # Generate plots
    plot_results(results)