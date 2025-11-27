cmake_minimum_required(VERSION 3.4...3.18)
project(bloomwisard)

# Use the Python executable provided by setup.py
set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11 subdirectory from pip installation
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
)

if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 cmake directory")
endif()

find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

# Add include directories
include_directories(include)

# Create the Python module
# Note: bind.cc includes all other source files via bind.h
pybind11_add_module(bloomwisard 
    src/bind.cc
)

# Set C++ standard
target_compile_features(bloomwisard PRIVATE cxx_std_11)

# Set optimization flags
target_compile_options(bloomwisard PRIVATE -O3 -Wall)

# Install the module
install(TARGETS bloomwisard DESTINATION .)
