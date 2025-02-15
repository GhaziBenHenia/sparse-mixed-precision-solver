#include "Solver.hpp"
#include <Eigen/Sparse>
#include <stdexcept>

SparseMatrixCSR::SparseMatrixCSR(const std::vector<double>& data,
    const std::vector<int>& indices,
    const std::vector<int>& indptr,
    int n)
    : data_(data), indices_(indices), indptr_(indptr), size_(n) {
    if (indptr.size() != static_cast<size_t>(n + 1)) {
        throw std::invalid_argument("Invalid indptr size");
    }
    for (size_t i = 1; i < indptr.size(); ++i) {
        if (indptr[i] < indptr[i - 1]) {
            throw std::invalid_argument("Indptr must be non-decreasing");
        }
    }
    for (int idx : indices) {
        if (idx < 0 || idx >= n) throw std::invalid_argument("Invalid column index");
    }
}

int SparseMatrixCSR::size() const { return size_; }

void SparseMatrixCSR::toEigenSparseMatrix(Eigen::SparseMatrix<double>& eigenMat) const {
    eigenMat.resize(size_, size_);
    eigenMat.reserve(indptr_.back());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(data_.size());
    for (int i = 0; i < size_; ++i) {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j) {
            triplets.emplace_back(i, indices_[j], data_[j]);
        }
    }
    eigenMat.setFromTriplets(triplets.begin(), triplets.end());
}

void SparseMatrixCSR::toEigenSparseMatrix(Eigen::SparseMatrix<float>& eigenMat) const {
    eigenMat.resize(size_, size_);
    eigenMat.reserve(indptr_.back());
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(data_.size());
    for (int i = 0; i < size_; ++i) {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j) {
            triplets.emplace_back(i, indices_[j], static_cast<float>(data_[j]));
        }
    }
    eigenMat.setFromTriplets(triplets.begin(), triplets.end());
}
