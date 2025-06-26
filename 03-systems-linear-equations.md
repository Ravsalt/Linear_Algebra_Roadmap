# Systems of Linear Equations

## Introduction

A system of linear equations is a collection of one or more linear equations involving the same set of variables. For example:

$$
\begin{cases}
2x + 3y = 8 \\
4x - y = 2
\end{cases}
$$

## Matrix Representation

Any system of linear equations can be written in matrix form as $A\mathbf{x} = \mathbf{b}$, where:
- $A$ is the coefficient matrix
- $\mathbf{x}$ is the vector of variables
- $\mathbf{b}$ is the constant terms vector

For the example above:

$$
\begin{bmatrix} 2 & 3 \\ 4 & -1 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix} =
\begin{bmatrix} 8 \\ 2 \end{bmatrix}
$$

## Solving Systems with NumPy

### Using `numpy.linalg.solve`

NumPy provides a function to solve systems of linear equations:

```python
import numpy as np

# Coefficient matrix
A = np.array([
    [2, 3],
    [4, -1]
])

# Constants vector
b = np.array([8, 2])


# Solve the system
x = np.linalg.solve(A, b)
print(f"Solution: x = {x[0]:.2f}, y = {x[1]:.2f}")
```

### Example: Electrical Circuit Analysis

Consider this simple circuit with two loops:

```
   R1      R3
A-----VVV-----B
|     R2      |
+-----VVV-----+
|             |
V1            V2
```

Using Kirchhoff's voltage law, we get:

$$
\begin{cases}
(R_1 + R_2)I_1 - R_2I_2 = V_1 \\
-R_2I_1 + (R_2 + R_3)I_2 = V_2
\end{cases}
$$

Let's solve for the currents with specific values:

```python
# Given values (in ohms and volts)
R1, R2, R3 = 10, 20, 15
V1, V2 = 12, 15

# Coefficient matrix
A = np.array([
    [R1 + R2, -R2],
    [-R2, R2 + R3]
])

# Constants vector
b = np.array([V1, V2])


# Solve for currents I1 and I2
I = np.linalg.solve(A, b)
print(f"I1 = {I[0]:.4f} A, I2 = {I[1]:.4f} A")
```

## Gaussian Elimination

Gaussian elimination is a method for solving systems of linear equations by transforming the augmented matrix to row echelon form.

### Manual Example

Solve the system:

$$
\begin{cases}
x + 2y + z = 1 \\
2x + y + 2z = -2 \\
3x + y + z = -1
\end{cases}
$$

### Using NumPy for Row Operations

```python
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Create augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Elimination
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

# Example system
A = np.array([
    [1, 2, 1],
    [2, 1, 2],
    [3, 1, 1]
], dtype=float)

b = np.array([1, -2, -1], dtype=float)

x = gaussian_elimination(A, b)
print(f"Solution: x = {x[0]:.2f}, y = {x[1]:.2f}, z = {x[2]:.2f}")
```

## Solution Types

A system of linear equations can have:

1. **Unique Solution**: Exactly one solution (consistent and independent)
2. **No Solution**: Inconsistent system (parallel planes that never meet)
3. **Infinite Solutions**: Dependent equations (same plane represented multiple times)

### Checking Solution Types

```python
def check_system(A, b):
    """Check the type of solution for the system Ax = b."""
    rank_A = np.linalg.matrix_rank(A)
    augmented = np.hstack([A, b.reshape(-1, 1)])
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    n = A.shape[1]  # Number of variables
    
    if rank_A == rank_augmented:
        if rank_A == n:
            return "Unique solution"
        else:
            return "Infinite solutions"
    else:
        return "No solution"

# Example 1: Unique solution
A1 = np.array([[1, 2], [3, 4]])
b1 = np.array([5, 11])
print(f"System 1: {check_system(A1, b1)}")

# Example 2: No solution
A2 = np.array([[1, 2], [1, 2]])
b2 = np.array([3, 4])
print(f"System 2: {check_system(A2, b2)}")

# Example 3: Infinite solutions
A3 = np.array([[1, 2], [2, 4]])
b3 = np.array([3, 6])
print(f"System 3: {check_system(A3, b3)}")
```

## Applications

### Curve Fitting

Find a quadratic polynomial \(y = ax^2 + bx + c\) that passes through three points:

```python
# Points: (1,1), (2,3), (3,7)
A = np.array([
    [1, 1, 1],  # a(1)^2 + b(1) + c = 1
    [4, 2, 1],  # a(2)^2 + b(2) + c = 3
    [9, 3, 1]   # a(3)^2 + b(3) + c = 7
])

b = np.array([1, 3, 7])

# Solve for coefficients
coeffs = np.linalg.solve(A, b)
print(f"Polynomial: y = {coeffs[0]:.1f}x² + {coeffs[1]:.1f}x + {coeffs[2]:.1f}")
```

## Practice Exercises

1. Solve the following system using both matrix inversion and `numpy.linalg.solve`:
   $$\begin{cases}
   3x + 2y = 7 \\
   -x +  y = 1
   \end{cases}$$

2. Implement a function that performs back substitution on an upper triangular system.

3. For what values of \(k\) does the following system have:
   - No solution
   - Exactly one solution
   - Infinitely many solutions
   
   $$\begin{cases}
   x + 2y + z = 3 \\
   2x + 5y - z = -4 \\
   3x - 2y - kz = 11
   \end{cases}$$

---

Next: [Vector Spaces →](04-vector-spaces.md)
