#include "Solver.hpp"
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>

SOLVER_API int solve_iterative_refinement(
    const double* A_data, const int* A_indices, const int* A_indptr,
    const double* b, double* x, int n, double tolerance, int max_iterations
) {
    try {
        if (n <= 0 || !A_data || !A_indices || !A_indptr || !b || !x) return -1;

        SparseMatrixCSR A(
            std::vector<double>(A_data, A_data + A_indptr[n]),
            std::vector<int>(A_indices, A_indices + A_indptr[n]),
            std::vector<int>(A_indptr, A_indptr + n + 1),
            n
        );

        // Single-precision solve
        Eigen::SparseMatrix<float> A_float;
        A.toEigenSparseMatrix(A_float);
        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver_single;
        solver_single.compute(A_float);
        if (solver_single.info() != Eigen::Success) {
            std::cerr << "Single-precision solver failed!" << std::endl;
            return -2;
        }

        Eigen::VectorXf b_float = Eigen::Map<const Eigen::VectorXd>(b, n).cast<float>();
        Eigen::VectorXf x_float = solver_single.solve(b_float);

        // Convert to double precision
        Eigen::SparseMatrix<double> A_double;
        A.toEigenSparseMatrix(A_double);
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_double;
        solver_double.compute(A_double);
        if (solver_double.info() != Eigen::Success) {
            std::cerr << "Double-precision solver failed!" << std::endl;
            return -3;
        }

        Eigen::VectorXd x_current = x_float.cast<double>();

        // Iterative refinement loop
        for (int iter = 0; iter < max_iterations; ++iter) {
            Eigen::VectorXd residual = Eigen::Map<const Eigen::VectorXd>(b, n) - A_double * x_current;
            if (residual.norm() < tolerance) break;

            x_current += solver_double.solve(residual);
        }

        Eigen::Map<Eigen::VectorXd>(x, n) = x_current;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -4;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return -4;
    }
}
