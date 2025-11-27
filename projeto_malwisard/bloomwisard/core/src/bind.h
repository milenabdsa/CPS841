#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>

using namespace std;

#include "murmur3.c"
#include "bitarray.c"

#include "discriminator.cc"
#include "dict_discriminator.cc"
#include "bloom_discriminator.cc"

#include "dict_wisard.cc"
#include "wisard.cc"
#include "bloomwisard.cc"
