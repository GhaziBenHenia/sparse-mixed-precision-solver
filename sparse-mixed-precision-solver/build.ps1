# Remove the build directory if it exists
if (Test-Path build) {
    Remove-Item -Recurse -Force build
}

if (Test-Path out) {
    Remove-Item -Recurse -Force out
}

if (Test-Path Testing) {
    Remove-Item -Recurse -Force Testing
}

# Create a new build directory
mkdir build -Force

# Navigate to the build directory
cd build

# Configure the project using CMake for Visual Studio 2022
cmake .. -G "Visual Studio 17 2022"

# Build the project in Release mode
cmake --build . --config Release

# Navigate to the Release directory
cd shared_library/Release

# Run the test executable
.\test_Solver.exe 

#example to run a specific test
#.\test_Solver.exe --gtest_filter=SolverTest.SmallSystem

cd ../../..