# Create 'thirdparty' directory if it doesn't exist
if (!(Test-Path -Path "thirdparty")) {
    Write-Host "Creating thirdparty directory..."
    New-Item -ItemType Directory -Path "thirdparty" | Out-Null
}

# Navigate to the 'thirdparty' directory
Set-Location -Path "thirdparty"

# Clone Eigen manually if not already present
if (!(Test-Path "eigen")) {
    Write-Host "Cloning Eigen..."
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git eigen
} else {
    Write-Host "Eigen already exists, skipping..."
}

# Clone GoogleTest manually if not already present
if (!(Test-Path "googletest")) {
    Write-Host "Cloning GoogleTest..."
    git clone --depth 1 https://github.com/google/googletest.git googletest
} else {
    Write-Host "GoogleTest already exists, skipping..."
}

# Navigate back to the original directory
Set-Location -Path ..

# Initialize and update submodules (if they exist in .gitmodules)
Write-Host "Initializing and updating submodules..."
git submodule update --init --recursive

Write-Host "Submodules setup complete!" -ForegroundColor Green
