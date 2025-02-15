#include <gtest/gtest.h>
#include "Solver.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>


TEST(SolverTest, SmallSystem) {
    double A_data[] = { 4.0, 1.0, 1.0, 3.0 };
    int A_indices[] = { 0, 1, 0, 1 };
    int A_indptr[] = { 0, 2, 4 };
    double b[] = { 1.0, 2.0 };
    double x[2] = { 0.0 };
    double tolerance = 1e-8;
    int max_iterations = 50;

    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b, x, 2, max_iterations, tolerance);
    ASSERT_EQ(status, 0);
    EXPECT_NEAR(x[0], 1.0 / 11.0, 1e-6);  // ~0.0909
    EXPECT_NEAR(x[1], 7.0 / 11.0, 1e-6);   // ~0.6364
}

TEST(SolverTest, InvalidInput) {
    double x[2];
    double tolerance = 1e-8;
    int max_iterations = 50;
    int status = solve_iterative_refinement(nullptr, nullptr, nullptr, nullptr, x, 2, tolerance, max_iterations);
    ASSERT_EQ(status, -1);
}

TEST(SolverTest, LargeDiagonallyDominantSystem) {
    const int n = 100; // Test with a 100x100 system
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    double tolerance = 1e-8;
    int max_iterations = 50;

    // Generate a diagonally dominant sparse matrix
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        // Diagonal element: Ensure dominance
        triplets.emplace_back(i, i, n * 2.0);
        // Off-diagonal elements: Add -1 to adjacent columns
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Extract CSR data from Eigen matrix
    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    // Generate exact solution (random vector)
    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    // Solve using your iterative refinement solver
    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(
        A_data, A_indices, A_indptr,
        b.data(), x_computed.data(), n,
        tolerance, max_iterations
    );

    // Check results
    ASSERT_EQ(status, 0); // Ensure solver succeeded
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-6); // Expect small error
}

TEST(SolverTest, NonDiagonallyDominantSystem) {
    const int n = 50; // Moderate size
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    double tolerance = 1e-8;
    int max_iterations = 50;

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
        triplets.emplace_back(i, i, 1.5); // Weak diagonal dominance
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_computed.data(), n, tolerance, max_iterations);

    ASSERT_EQ(status, 0);
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-4);
}

TEST(SolverTest, LargeSparseSystem) {
    const int n = 10000;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    double tolerance = 1e-8;
    int max_iterations = 50;

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_computed.data(), n, tolerance, max_iterations);

    ASSERT_EQ(status, 0);
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-4);
}

TEST(SolverTest, NonSymmetricSparseMatrix) {
    const int n = 50;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    double tolerance = 1e-8;
    int max_iterations = 50;

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0 + (i % 5) * 0.5); // More variation
        if (i > 0) triplets.emplace_back(i, i - 1, -0.7 + (i % 2) * 0.2);
        if (i < n - 2) triplets.emplace_back(i, i + 2, -0.4 + (i % 3) * 0.1);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_computed.data(), n, tolerance, max_iterations);

    ASSERT_EQ(status, 0);
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-6);
}

TEST(SolverTest, SymmetricPositiveDefiniteMatrix) {
    const int n = 50;
    double tolerance = 1e-8;
    int max_iterations = 50;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0);
        if (i > 0) {
            triplets.emplace_back(i, i - 1, -1.0);
            triplets.emplace_back(i - 1, i, -1.0); // Ensure symmetry
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_computed.data(), n, tolerance, max_iterations);

    ASSERT_EQ(status, 0);
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-6);
}

TEST(SolverTest, HighlyIllConditionedMatrix) {
    const int n = 10; // Small size (Hilbert matrices become unstable quickly)
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    double tolerance = 1e-8;
    int max_iterations = 50;

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double value = 1.0 / (i + j + 1);
            if (std::abs(value) > 1e-12) {
                triplets.emplace_back(i, j, value);
            }
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    std::vector<double> x_computed(n, 0.0);
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_computed.data(), n, tolerance, max_iterations);

    ASSERT_EQ(status, 0);
    Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x_computed.data(), n);
    double error = (x_eigen - x_expected).norm();
    EXPECT_LT(error, 1e-2); // Expect larger error due to ill-conditioning
}

TEST(SolverTest, CompareWithEigenSolvers) {
    const int n = 100;
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    double tolerance = 1e-8;
    int max_iterations = 50;

    // Generate a matrix
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 4.0 + (i % 5)); // Diagonal dominance
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Exact solution and RHS
    Eigen::VectorXd x_expected = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = A * x_expected;

    // Extract CSR data for the custum solver
    const double* A_data = A.valuePtr();
    const int* A_indices = A.innerIndexPtr();
    const int* A_indptr = A.outerIndexPtr();

    // Solve using your iterative refinement solver
    std::vector<double> x_custom(n, 0.0);
    auto start_custom = std::chrono::high_resolution_clock::now();
    int status = solve_iterative_refinement(A_data, A_indices, A_indptr, b.data(), x_custom.data(), n, tolerance, max_iterations);
    auto end_custom = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(status, 0);

    // Solve using Eigen's SparseLU (direct solver)
    Eigen::VectorXd x_sparselu(n);
    auto start_sparselu = std::chrono::high_resolution_clock::now();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_sparselu;
    solver_sparselu.compute(A);
    if (solver_sparselu.info() == Eigen::Success) {
        x_sparselu = solver_sparselu.solve(b);
    }
    else {
        FAIL() << "Eigen::SparseLU failed to factorize the matrix.";
    }
    auto end_sparselu = std::chrono::high_resolution_clock::now();

    // Solve using Eigen's BiCGSTAB (iterative solver)
    Eigen::VectorXd x_bicgstab(n);
    auto start_bicgstab = std::chrono::high_resolution_clock::now();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_bicgstab;
    solver_bicgstab.compute(A);
    if (solver_bicgstab.info() == Eigen::Success) {
        x_bicgstab = solver_bicgstab.solve(b);
    }
    else {
        FAIL() << "Eigen::BiCGSTAB failed to solve the system.";
    }
    auto end_bicgstab = std::chrono::high_resolution_clock::now();

    // Solve using Eigen::ConjugateGradient (for SPD matrices)
    // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver_cg; // Uncomment if SPD

    // Calculate errors
    double error_custom = (Eigen::Map<Eigen::VectorXd>(x_custom.data(), n) - x_expected).norm();
    double error_sparselu = (x_sparselu - x_expected).norm();
    double error_bicgstab = (x_bicgstab - x_expected).norm();

    // Report results
    auto duration_custom = std::chrono::duration_cast<std::chrono::milliseconds>(end_custom - start_custom).count();
    auto duration_sparselu = std::chrono::duration_cast<std::chrono::milliseconds>(end_sparselu - start_sparselu).count();
    auto duration_bicgstab = std::chrono::duration_cast<std::chrono::milliseconds>(end_bicgstab - start_bicgstab).count();

    std::cout << "\n=== Solver Comparison ===\n";
    std::cout << "Custom Solver (Mixed Precision):\n  Error: " << error_custom << "\n  Time: " << duration_custom << " ms\n";
    std::cout << "Eigen::SparseLU (Direct):\n  Error: " << error_sparselu << "\n  Time: " << duration_sparselu << " ms\n";
    std::cout << "Eigen::BiCGSTAB (Iterative):\n  Error: " << error_bicgstab << "\n  Time: " << duration_bicgstab << " ms\n";
    std::cout << "=========================\n";

    // Ensure errors are within tolerance
    EXPECT_LT(error_custom, 1e-6);
    EXPECT_LT(error_sparselu, 1e-10); // Direct solvers are exact
    EXPECT_LT(error_bicgstab, 1e-6);
}



