cc=clang++

# Python do venv
PYTHON_EXE=/Users/pedrohenriqueteixeira/Poli/CPS841/projeto_malwisard/venv/bin/python3

# 1. Use $(shell ...) para obter os CFLAGS do pybind11
PYBIND_INCLUDES=$(shell $(PYTHON_EXE) -m pybind11 --includes)

# 2. Use $(shell ...) para obter o sufixo da extensão
exec_py=$(shell $(PYTHON_EXE) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 3. Flags para macOS
flags=-O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $(PYBIND_INCLUDES) -Iinclude

code=src/bind.cc
bin=wnn

all:
	$(cc) $(flags) $(code) -o $(bin)$(exec_py)

clean:
	rm -f $(bin)$(exec_py) $(bin)*.so