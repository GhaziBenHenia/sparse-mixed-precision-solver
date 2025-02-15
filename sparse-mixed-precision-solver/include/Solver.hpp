#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
#if defined(SOLVER_DLL_EXPORTS)
#define SOLVER_API __declspec(dllexport)  // Export for DLL
#elif defined(SOLVER_STATIC)
#define SOLVER_API  // Empty for static library
#else
#define SOLVER_API __declspec(dllimport)  // Import for external use
#endif
#else
#define SOLVER_API  // Empty on non-Windows
#endif

class SparseMatrixCSR {
public:
    SparseMatrixCSR(const std::vector<double>& data,
        const std::vector<int>& indices,
        const std::vector<int>& indptr,
        int n);

    void toEigenSparseMatrix(Eigen::SparseMatrix<double>& eigenMat) const;
    void toEigenSparseMatrix(Eigen::SparseMatrix<float>& eigenMat) const;
    int size() const;

private:
    std::vector<double> data_;
    std::vector<int> indices_, indptr_;
    int size_;
};

#ifdef __cplusplus
extern "C" {
#endif

    SOLVER_API int solve_iterative_refinement(
        const double* A_data,
        const int* A_indices,
        const int* A_indptr,
        const double* b,
        double* x,
        int n,
        double tolerance,// = 1e-8,
        int max_iterations// = 50
    );

#ifdef __cplusplus
}
#endif