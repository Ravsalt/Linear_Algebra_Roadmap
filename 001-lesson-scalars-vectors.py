"""
001 - Scalars and Vectors in Linear Algebra for Machine Learning

This lesson introduces the fundamental building blocks of linear algebra:
- Scalars: Single numbers (like temperature, age, or a single feature)
- Vectors: Ordered arrays of numbers (like feature vectors in ML)
"""

# Scalars in Python
# A scalar is just a single number (integer or float)
temperature = 36.5  # A scalar representing body temperature in Celsius
age = 25           # A scalar representing age in years

# Vectors in Python (using lists and NumPy)
# A vector is an ordered array of numbers
# In ML, a vector might represent features of a data point
import numpy as np

# A simple vector using Python list
height_weight = [175, 68]  # [height in cm, weight in kg]

# Using NumPy (more common in ML)
features = np.array([0.5, 1.2, 3.8])  # A 3D feature vector

# Accessing vector elements
print("First element of features:", features[0])  # Indexing starts at 0
print("Length of features vector:", len(features))

# Vector operations
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# Element-wise addition
vector_sum = vector_a + vector_b  # [1+4, 2+5, 3+6] = [5, 7, 9]
print("\nVector addition:", vector_sum)

# Scalar multiplication
scalar = 2
scaled_vector = scalar * vector_a  # [2*1, 2*2, 2*3] = [2, 4, 6]
print("Scalar multiplication:", scaled_vector)
