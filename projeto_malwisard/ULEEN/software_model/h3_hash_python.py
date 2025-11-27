#!/usr/bin/false

################################################################################
# h3_hash_python.py
# Pure Python implementation of H3 hash (fallback for systems without C++ compiler)
#
# This is slower than the C++ version but works without compilation
################################################################################

import torch

def h3_hash(inp, hash_vals):
    """
    Pure Python implementation of H3 hash function
    
    Args:
        inp: A 2D tensor (dxb) consisting of d b-bit values to be hashed,
             expressed as bitvectors
        hash_vals: A 2D tensor (hxb) consisting of h sets of b int64s, representing
                   random constants to compute h unique hashes
    Returns:
        A tuple containing a 2D tensor (dxh) of int64s, representing the results 
        of the h hash functions on the d input values
    """
    device = inp.device
    
    # Garantir que hash_vals está em int64
    hash_vals = hash_vals.to(dtype=torch.int64)
    
    # Choose between hash values and 0 based on input bits
    # inp: (d, b), hash_vals: (h, b)
    # We want to select hash_vals when inp bit is 1, else 0
    # Converter inp para int64 para a operação einsum
    inp_int = inp.to(dtype=torch.int64)
    selected_entries = torch.einsum("hb,db->bdh", hash_vals, inp_int)
    
    # Perform XOR reduction along the b dimension
    reduction_result = torch.zeros(
        (inp.size(0), hash_vals.size(0)),
        dtype=torch.int64,
        device=device
    )
    
    for i in range(hash_vals.size(1)):
        reduction_result.bitwise_xor_(selected_entries[i])
    
    return (reduction_result,)
