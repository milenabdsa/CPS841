cc=g++

# Defina o caminho completo para o executável Python no formato Windows
PYTHON_EXE=C:/Users/Pedro2/AppData/Local/Programs/Python/Python39/python.exe
PYTHON_DIR=C:/Users/Pedro2/AppData/Local/Programs/Python/Python39

# 1. Use $(shell ...) para obter os CFLAGS do pybind11
PYBIND_INCLUDES=$(shell $(PYTHON_EXE) -m pybind11 --includes)

# 2. Use $(shell ...) para obter o sufixo da extensão
exec_py=$(shell $(PYTHON_EXE) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 3. Defina a variável flags sem static linking, usa geração de DLL com exports
flags=-O3 -Wall -shared -std=c++11 $(PYBIND_INCLUDES) -Iinclude

code=src/bind.cc
bin=wnn

all:
	$(cc) $(flags) $(code) -o $(bin)$(exec_py)