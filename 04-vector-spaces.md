# Vector Spaces

## Introduction to Vector Spaces

A vector space (or linear space) over a field $F$ (usually $\mathbb{R}$ or $\mathbb{C}$) is a set $V$ equipped with two operations:

1. **Vector addition**: $\mathbf{u} + \mathbf{v} \in V$ for all $\mathbf{u}, \mathbf{v} \in V$
2. **Scalar multiplication**: $a\mathbf{v} \in V$ for all $a \in F$ and $\mathbf{v} \in V$

These operations must satisfy the vector space axioms (associativity, commutativity, distributivity, etc.).

## Examples of Vector Spaces

### 1. $\mathbb{R}^n$
The set of all n-tuples of real numbers with component-wise addition and scalar multiplication.

```python
import numpy as np

# Vectors in R^3
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
v_sum = v1 + v2  # array([5, 7, 9])


# Scalar multiplication
scaled = 2 * v1  # array([2, 4, 6])
```

### 2. Matrix Spaces
All $m \times n$ matrices with matrix addition and scalar multiplication.

```python
# Set of 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])

# Matrix addition
C = A + B  # array([[1, 3], [4, 4]])


# Scalar multiplication
D = 3 * A  # array([[3, 6], [9, 12]])
```

### 3. Function Spaces
The set of all real-valued functions defined on an interval $[a, b]$ with pointwise addition and scalar multiplication.

```python
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def g(x):
    return x**2

# Function addition
def f_plus_g(x):
    return f(x) + g(x)


# Plot the functions
x = np.linspace(-2, 2, 100)
plt.plot(x, f(x), label='f(x) = sin(x)')
plt.plot(x, g(x), label='g(x) = x²')
plt.plot(x, f_plus_g(x), label='f(x) + g(x)')
plt.legend()
plt.grid(True)
plt.title('Function Space Example')
plt.show()
```

## Subspaces

A subspace $W$ of a vector space $V$ is a subset of $V$ that is itself a vector space with the operations inherited from $V$.

### Subspace Test
A subset $W$ of $V$ is a subspace if and only if:
1. $\mathbf{0} \in W$
2. $\mathbf{u} + \mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$
3. $a\mathbf{u} \in W$ for all $a \in F$ and $\mathbf{u} \in W$

### Example: Subspace of $\mathbb{R}^3$

The plane $x + y + z = 0$ is a subspace of $\mathbb{R}^3$.

```python
# Check if a vector is in the plane x + y + z = 0
def is_in_plane(v):
    return np.isclose(np.sum(v), 0)

v1 = np.array([1, -1, 0])   # In the plane
v2 = np.array([1, 1, 1])    # Not in the plane

print(f"v1 is in plane: {is_in_plane(v1)}")
print(f"v2 is in plane: {is_in_plane(v2)}")
```

## Span and Linear Independence

### Span
The span of a set of vectors $S = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$ is the set of all linear combinations:

$$
\text{span}(S) = \{a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \dots + a_n\mathbf{v}_n \mid a_i \in F\}
$$

```python
def span(vectors, coefficients):
    """Compute a linear combination of vectors."""
    return sum(c * v for c, v in zip(coefficients, vectors))

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Any vector in the xy-plane can be written as a linear combination of v1 and v2
coeffs = [2, 3]  # 2*v1 + 3*v2
result = span([v1, v2], coeffs)  # array([2, 3, 0])
```

### Linear Independence
A set of vectors is **linearly independent** if no vector in the set can be written as a linear combination of the others.

```python
def is_linearly_independent(vectors):
    """Check if vectors are linearly independent using the rank."""
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)

# Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
print(f"Independent: {is_linearly_independent([v1, v2, v3])}")

# Linearly dependent vectors
v4 = np.array([1, 1, 0])
print(f"Independent: {is_linearly_independent([v1, v2, v4])}")
```

## Basis and Dimension

### Basis
A basis for a vector space $V$ is a linearly independent set that spans $V$.

### Standard Basis for $\mathbb{R}^3$
$$
\mathbf{e}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix}, \quad
\mathbf{e}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix}, \quad
\mathbf{e}_3 = \begin{bmatrix}0\\0\\1\end{bmatrix}
$$

```python
# Standard basis for R^3
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Any vector in R^3 can be written as a linear combination of e1, e2, e3
v = np.array([2, -1, 5])
coeffs = [2, -1, 5]  # v = 2*e1 - e2 + 5*e3
```

### Dimension
The dimension of a vector space is the number of vectors in any basis for the space.

## Row Space, Column Space, and Null Space

### Column Space
```python
def column_space(A):
    """Find a basis for the column space of A."""
    _, pivots = np.linalg.qr(A)
    return A[:, pivots]

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Column space basis
col_basis = column_space(A)
print("Column space basis:")
print(col_basis)
```

### Null Space
```python
from scipy.linalg import null_space

# Find null space of A
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

null_basis = null_space(A)
print("Null space basis (columns):")
print(null_basis)
```

## Practice Exercises

1. Determine if the following sets are subspaces of $\mathbb{R}^3$:
   a) All vectors of the form $\begin{bmatrix}x\\y\\0\end{bmatrix}$
   b) All vectors with $x + y + z = 1$
   c) All vectors with $x \leq y \leq z$

2. Find a basis for the subspace of $\mathbb{R}^4$ spanned by:
   $$
   \begin{bmatrix}1\\2\\3\\4\end{bmatrix}, 
   \begin{bmatrix}2\\3\\4\\5\end{bmatrix}, 
   \begin{bmatrix}3\\4\\5\\6\end{bmatrix}
   $$

3. Determine if the following vectors are linearly independent:
   $$
   \begin{bmatrix}1\\2\\3\end{bmatrix}, 
   \begin{bmatrix}4\\5\\6\end{bmatrix}, 
   \begin{bmatrix}7\\8\\9\end{bmatrix}
   $$

4. Find the dimension of the null space of:
   $$
   \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}
   $$

---

Next: [Linear Transformations →](05-linear-transformations.md)
