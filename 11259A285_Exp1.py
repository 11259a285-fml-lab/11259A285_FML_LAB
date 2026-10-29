# EXPERIMENT 1: WORKING WITH NUMPY ARRAYS

import numpy as np

# 1. Program to print NumPy version
print("1. NUMPY VERSION")
print("NumPy version:", np.__version__)

# 2. Program to print different types of arrays

# A. One-dimensional array
print("\n2. ONE-DIMENSIONAL ARRAY")
x = np.arange(2, 6).reshape(4)
print(x)

# B. Two-dimensional array
print("\n3. TWO-DIMENSIONAL ARRAY")
x = np.arange(2, 10).reshape(2, 4)
print(x)

# C. Three-dimensional array
print("\n4. THREE-DIMENSIONAL ARRAY")
x = np.arange(24).reshape(4, 3, 2)
print(x)