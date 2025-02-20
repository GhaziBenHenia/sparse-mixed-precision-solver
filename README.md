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
```
Benchmarking matrix size: 100x100

Results for n = 100:
Solver          | Time (mean±std)           | Absolute Error (mean±std)      | Relative Error (mean±std)
-------------------------------------------------------------------------------------------------------------
IR solver       | 0.0001 ± 0.0000           | 7.26e-16 ± 3.03e-17            | 7.26e-17 ± 3.03e-18
SciPy spsolve   | 0.0001 ± 0.0001           | 7.33e-16 ± 1.30e-16            | 7.33e-17 ± 1.30e-17
SciPy CG        | 0.0194 ± 0.0015           | 8.35e-02 ± 8.13e-02            | 8.35e-03 ± 8.13e-03
PyAMG           | 0.0030 ± 0.0008           | 6.20e-07 ± 9.86e-07            | 6.20e-08 ± 9.86e-08

Benchmarking matrix size: 200x200

Results for n = 200:
Solver          | Time (mean±std)           | Absolute Error (mean±std)      | Relative Error (mean±std)
-------------------------------------------------------------------------------------------------------------
IR solver       | 0.0011 ± 0.0002           | 1.42e-15 ± 1.10e-16            | 1.01e-16 ± 7.75e-18
SciPy spsolve   | 0.0009 ± 0.0007           | 2.22e-15 ± 1.98e-16            | 1.57e-16 ± 1.40e-17
SciPy CG        | 0.0419 ± 0.0030           | 3.90e+00 ± 2.43e+00            | 2.76e-01 ± 1.72e-01
PyAMG           | 0.0035 ± 0.0007           | 6.87e-06 ± 3.03e-06            | 4.86e-07 ± 2.14e-07

Benchmarking matrix size: 500x500

Results for n = 500:
Solver          | Time (mean±std)           | Absolute Error (mean±std)      | Relative Error (mean±std)     
-------------------------------------------------------------------------------------------------------------
IR solver       | 0.0186 ± 0.0035           | 3.17e-15 ± 1.33e-16            | 1.42e-16 ± 5.95e-18
SciPy spsolve   | 0.0079 ± 0.0005           | 1.45e-14 ± 7.34e-16            | 6.49e-16 ± 3.28e-17
SciPy CG        | 0.1285 ± 0.0160           | 3.53e+01 ± 7.97e+00            | 1.58e+00 ± 3.57e-01
PyAMG           | 0.0070 ± 0.0007           | 9.77e-08 ± 2.19e-08            | 4.37e-09 ± 9.79e-10

Benchmarking matrix size: 1000x1000

Results for n = 1000:
Solver          | Time (mean±std)           | Absolute Error (mean±std)      | Relative Error (mean±std)     
-------------------------------------------------------------------------------------------------------------
IR solver       | 0.2837 ± 0.0867           | 5.82e-15 ± 6.04e-17            | 1.84e-16 ± 1.91e-18
SciPy spsolve   | 0.1249 ± 0.0309           | 3.97e-14 ± 9.70e-16            | 1.26e-15 ± 3.07e-17
SciPy CG        | 0.4898 ± 0.1165           | 5.60e+01 ± 4.09e+00            | 1.77e+00 ± 1.29e-01
PyAMG           | 0.0181 ± 0.0038           | 9.57e-08 ± 1.19e-08            | 3.03e-09 ± 3.75e-10
```
---

## Benchmark Summary  
Compared the mixed-precision **Iterative Refinement (IR) Solver** with SciPy solvers (`spsolve`, CG) and **PyAMG** on sparse linear systems.

- **IR Solver achieves the highest accuracy** across all cases.  
- **SciPy `spsolve` is faster** than IR for larger matrices but has slightly higher errors.  
- **PyAMG is the fastest** but sacrifices accuracy, making it suitable for approximate solutions.  
- **SciPy CG is the slowest and least accurate**.  

## Testing Limitations & Future Work  
The solver was **not tested** on all possible cases, including **different matrix structures and problem conditions**, such as a **diagonally dominant random sparse matrix**. Further testing is needed to evaluate performance on **ill-conditioned and larger sparse systems**.  

Additionally, this project does **not fully implement the advanced techniques** described in the article. It serves as an **introductory example** of mixed-precision iterative refinement. The article explores more sophisticated methods, including:  
- **Three-precision computation (half, single, double)** instead of two.  
- **GMRES preconditioning** for handling ill-conditioned systems.  
- **Explicit scaling techniques** to prevent floating-point underflow/overflow.  

For future improvements, I am working on **adding half-precision support, GMRES-based correction solving, and advanced error analysis** to align more closely with the article’s methodology.  

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
