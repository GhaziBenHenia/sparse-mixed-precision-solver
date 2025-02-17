# **Sparse Linear Solver with Mixed-Precision Iterative Refinement**

## **Introduction**
This project implements a **mixed-precision iterative refinement solver for sparse linear systems**, inspired by:  
**Carson, E., & Higham, N. J. (2018).** [*Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions*](https://doi.org/10.1137/17M1140819).  

### **Key Features**
- **Mixed-Precision Algorithm**:  
  - **Single-precision (`float`) LU factorization** for speed.  
  - **Double-precision (`double`) iterative refinement** for improved accuracy.
  - Adapts the iterative refinement approach, simplifying it to **two precisions (single/double) instead of three (half/single/double)** to make the implementation more straightforward for this project.

- **Sparse Matrix Support**:  
  - Uses **Compressed Sparse Row (CSR)** format for memory efficiency.  

- **High-Performance Implementation**:  
  - Written in **C++** and compiled as a **DLL** for optimized computation.  
  - **Python interface** for testing and benchmarking against SciPy solvers.  

---

## **Project Structure**
```
sparse-mixed-precision-solver/  
├── include/                  # Header files (Public API)
│   └── Solver.hpp            # DLL export macros and function declarations
├── src/                      # C++ source files
│   ├── SparseMatrix.cpp      # CSR matrix class
│   └── Solver.cpp            # Mixed-precision solver logic
├── test/                     # Unit tests
│   └── test_Solver.cpp       # C++ test cases using Google Test
├── build.ps1                 # PowerShell script to build the project with CMake and run tests
├── thirdparty_setup.ps1      # Installs dependencies (Eigen, Google Test)
└── CMakeLists.txt            # CMake configuration  

Benchmark/  
├── Benchmarks.py             # Python script to compare solver performance  
├── solver_interface.py       # ctypes wrapper for calling the solver DLL  
└── requirements.txt          # Python dependencies for benchmarks  
```

---

## **Dependencies**
This project relies on the following libraries:

- **Eigen**: C++ template library for linear algebra.  
- **Google Test**: Unit testing framework for C++.  
- **Python (Optional)**: Required for benchmarking (uses SciPy and NumPy).  

---

## **Installation and Setup**

### **Prerequisites**
Ensure you have the following installed:
- **Visual Studio** (for C++ compilation)
- **CMake** (for project building)
- **Python** (for benchmarking)

### **Setup Instructions**

#### **1. Clone the repository**
```sh
git clone https://github.com/GhaziBenHenia/sparse-mixed-precision-solver.git
cd sparse-mixed-precision-solver
```

#### **2. Install third-party dependencies**
```sh
./thirdparty_setup.ps1
```

#### **3. Build the project and run tests**
```sh
./build.ps1
```

#### **4. Run Python Benchmarks (Optional)**
##### **Set up a virtual environment:**
```sh
cd Benchmark/
python -m venv venv
```
##### **Activate the virtual environment:**
  ```sh
  .\venv\Scripts\Activate
  ```
##### **Install required dependencies:**
```sh
pip install -r requirements.txt
```
##### **Run the benchmark script:**
```sh
python Benchmarks.py
```

---

## **Benchmark Results**

---

## **Conclusion & Limitations of this Project**
- **Advantages**:


- **Limitations**:


---

## **Citation**
```bibtex
@article{Carson2018,
  author = {Carson, Erin and Higham, Nicholas J.},
  title = {Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions},
  journal = {SIAM Journal on Scientific Computing},
  volume = {40},
  number = {2},
  pages = {A817-A847},
  year = {2018},
  doi = {10.1137/17M1140819},
}
