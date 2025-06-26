# Basic Concepts in Linear Algebra

## Scalars, Vectors, and Matrices

### Scalars
A scalar is a single number (real or complex). In Python, we can represent scalars as simple numbers:

```python
# Scalar examples
a = 5
b = 3.14
c = 2 + 3j  # Complex number
```

### Vectors
A vector is an ordered list of numbers. In Python, we use NumPy arrays to represent vectors:

```python
import numpy as np

# Column vector
v = np.array([1, 2, 3])
print(f"Vector v: {v}")
print(f"Shape: {v.shape}")

# Row vector
v_row = v.reshape(1, -1)
print(f"\nRow vector: {v_row}")
print(f"Shape: {v_row.shape}")
```

### Matrices
A matrix is a 2D array of numbers. In NumPy:

```python
# 2x3 matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")
```

## Vector Operations

### Addition and Subtraction
Vectors of the same size can be added or subtracted element-wise:

$$
\mathbf{u} \pm \mathbf{v} = \begin{bmatrix} u_1 \pm v_1 \\ u_2 \pm v_2 \\ \vdots \\ u_n \pm v_n \end{bmatrix}
$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Vector addition
w = u + v  # array([5, 7, 9])

# Vector subtraction
z = u - v  # array([-3, -3, -3])
```

### Scalar Multiplication
Multiply every element of a vector by a scalar:

$$
c\mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \\ \vdots \\ c v_n \end{bmatrix}
$$

```python
# Scalar multiplication
result = 2 * u  # array([2, 4, 6])
```

## Matrix Operations

### Matrix Addition
Matrices of the same dimensions can be added element-wise:

$$
A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} & \dots \\
                          a_{21}+b_{21} & a_{22}+b_{22} & \dots \\
                          \vdots & \vdots & \ddots \end{bmatrix}
$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B  # array([[ 6,  8], [10, 12]])
```

### Matrix Multiplication
Matrix multiplication is more complex than element-wise multiplication. For matrices A (m×n) and B (n×p):

$$
(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}
$$

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Using @ operator for matrix multiplication
C = A @ B  # array([[19, 22], [43, 50]])

# Alternative using np.matmul
D = np.matmul(A, B)  # Same as above
```

### Transpose
Transposing a matrix swaps its rows and columns:

$$
A^T_{ij} = A_{ji}
$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Transpose
A_transpose = A.T  # or np.transpose(A)
print(f"Original shape: {A.shape}")
print(f"Transposed shape: {A_transpose.shape}")
```

## Special Matrices

### Identity Matrix
A square matrix with ones on the diagonal and zeros elsewhere:

$$
I = \begin{bmatrix} 1 & 0 & \dots & 0 \\
                      0 & 1 & \dots & 0 \\
                      \vdots & \vdots & \ddots & \vdots \\
                      0 & 0 & \dots & 1 \end{bmatrix}
$$

```python
# 3x3 identity matrix
I = np.eye(3)
print(I)
```

### Diagonal Matrix
A matrix where all off-diagonal elements are zero:

```python
# Create diagonal matrix from a vector
diag = np.diag([1, 2, 3])
print(diag)
```

## Practical Example: Linear Combination

A linear combination of vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$ is an expression of the form:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n
$$

```python
# Define vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Coefficients
c1, c2 = 2, 3

# Linear combination
result = c1 * v1 + c2 * v2  # array([2, 3])
```

## Practice Exercises

1. Create a 2x3 matrix and its transpose. What are their shapes?

2. Implement matrix multiplication without using NumPy's built-in functions.

3. Given vectors $\mathbf{u} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$, compute $2\mathbf{u} - 3\mathbf{v}$.

4. Create a function that checks if a matrix is symmetric (A = A^T).

---

Next: [Systems of Linear Equations →](03-systems-linear-equations.md)
