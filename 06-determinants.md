# Determinants in Linear Algebra with Python

## Introduction to Determinants

The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix and the linear transformation it represents.

### Key Properties of Determinants

1. **Geometric Interpretation**: The absolute value of the determinant gives the scale factor by which area/volume changes under the transformation.
2. **Invertibility**: A matrix is invertible if and only if its determinant is non-zero.
3. **Multiplicative Property**: $\det(AB) = \det(A)\det(B)$
4. **Transpose Property**: $\det(A) = \det(A^T)$

## Calculating Determinants in Python

### Using NumPy

```python
import numpy as np

# Create a 2x2 matrix
A = np.array([
    [3, 1],
    [1, 2]
])

# Calculate determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A:.2f}")

# For larger matrices
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

det_B = np.linalg.det(B)
print(f"Determinant of B: {det_B:.2f}")
```

## Visualizing Determinants

### 2D Transformation Example

Let's visualize how the determinant relates to area scaling in 2D transformations.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_transformation(T, title):
    # Original square vertices
    square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    
    # Apply transformation
    transformed = (T @ square.T).T
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.add_patch(Polygon(square, alpha=0.5, color='blue'))
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_title('Original Unit Square')
    ax1.set_aspect('equal')
    
    # Transformed
    ax2.add_patch(Polygon(transformed, alpha=0.5, color='red'))
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_title(f'Transformed (det = {np.linalg.det(T):.2f})')
    ax2.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example transformations
T_scale = np.array([
    [1.5, 0],
    [0, 0.8]
])

T_shear = np.array([
    [1, 0.5],
    [0.3, 1]
])

# Plot transformations
plot_transformation(T_scale, "Scaling Transformation")
plot_transformation(T_shear, "Shear Transformation")
```

## Properties of Determinants

### 1. Effect of Row Operations

```python
def demonstrate_row_operations():
    # Original matrix
    A = np.array([
        [1, 2],
        [3, 4]
    ])
    
    # Row swap (changes sign of determinant)
    A_swap = A[[1, 0], :]  # Swap rows 0 and 1
    
    # Row scaling (scales determinant)
    A_scale = A.copy()
    A_scale[0, :] *= 2  # Scale first row by 2
    
    # Row addition (doesn't change determinant)
    A_add = A.copy()
    A_add[1, :] += 2 * A[0, :]  # Add 2*row1 to row2
    
    # Calculate determinants
    print(f"Original det: {np.linalg.det(A):.2f}")
    print(f"After row swap: {np.linalg.det(A_swap):.2f}")
    print(f"After scaling row: {np.linalg.det(A_scale):.2f}")
    print(f"After row addition: {np.linalg.det(A_add):.2f}")

demonstrate_row_operations()
```

### 2. Determinant of Products and Inverses

```python
def demonstrate_det_properties():
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)
    
    # Property: det(AB) = det(A)det(B)
    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    det_AB = np.linalg.det(A @ B)
    
    print(f"det(A) * det(B): {det_A * det_B:.6f}")
    print(f"det(AB): {det_AB:.6f}")
    print(f"Are they equal? {np.isclose(det_A * det_B, det_AB, rtol=1e-10)}")
    
    # Property: det(A⁻¹) = 1/det(A)
    if not np.isclose(det_A, 0):
        A_inv = np.linalg.inv(A)
        det_A_inv = np.linalg.det(A_inv)
        print(f"\ndet(A⁻¹): {det_A_inv:.6f}")
        print(f"1/det(A): {1/det_A:.6f}")
        print(f"Are they equal? {np.isclose(det_A_inv, 1/det_A, rtol=1e-10)}")

demonstrate_det_properties()
```

## Applications of Determinants

### 1. Solving Linear Systems (Cramer's Rule)

Cramer's Rule provides an explicit formula for the solution of a system of linear equations with as many equations as unknowns.

```python
def cramers_rule(A, b):
    """
    Solve the system Ax = b using Cramer's Rule.
    Returns the solution vector x.
    """
    n = len(b)
    det_A = np.linalg.det(A)
    
    if np.isclose(det_A, 0):
        raise ValueError("Matrix is singular. Cramer's Rule cannot be applied.")
    
    x = np.zeros(n)
    
    for i in range(n):
        # Create a copy of A and replace the i-th column with b
        Ai = A.copy()
        Ai[:, i] = b
        
        # Calculate determinant of Ai
        det_Ai = np.linalg.det(Ai)
        
        # Calculate x_i
        x[i] = det_Ai / det_A
    
    return x

# Example system
A = np.array([
    [3, 1, -2],
    [1, -1, 4],
    [2, 0, 3]
])

b = np.array([7, -2, 5])

# Solve using Cramer's Rule
x_cramer = cramers_rule(A, b)
print(f"Solution using Cramer's Rule: {x_cramer}")

# Verify with numpy's solver
x_np = np.linalg.solve(A, b)
print(f"Solution using np.linalg.solve: {x_np}")
print(f"Are the solutions equal? {np.allclose(x_cramer, x_np)}")
```

### 2. Area and Volume Calculations

```python
def triangle_area(vertices):
    """
    Calculate the area of a triangle given its vertices.
    vertices: 3x2 array where each row is a vertex (x, y)
    """
    # Create a matrix with vectors v2-v1 and v3-v1
    v1, v2, v3 = vertices
    A = np.column_stack((v2 - v1, v3 - v1))
    
    # Area is half the absolute value of the determinant
    return 0.5 * abs(np.linalg.det(A))

# Triangle vertices (x, y)
triangle = np.array([
    [0, 0],
    [4, 0],
    [0, 3]
])

area = triangle_area(triangle)
print(f"Area of the triangle: {area}")

# Plot the triangle
plt.figure()
tri = plt.Polygon(triangle, fill=None, edgecolor='blue')
plt.gca().add_patch(tri)
plt.xlim(-1, 5)
plt.ylim(-1, 4)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.title(f'Triangle with Area = {area}')
plt.show()
```

## Advanced: Computing Determinants from First Principles

### Recursive Implementation (Laplace Expansion)

```python
def recursive_det(A):
    """
    Compute the determinant of a square matrix using recursive Laplace expansion.
    Note: This is for educational purposes only. Use np.linalg.det for production.
    """
    n = A.shape[0]
    
    # Base case: 1x1 matrix
    if n == 1:
        return A[0, 0]
    
    # Base case: 2x2 matrix
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    det = 0
    for j in range(n):
        # Create submatrix by removing first row and j-th column
        submatrix = np.delete(np.delete(A, 0, 0), j, 1)
        
        # Recursive call with sign alternation
        det += ((-1) ** j) * A[0, j] * recursive_det(submatrix)
    
    return det

# Test with a small matrix
C = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

det_recursive = recursive_det(C)
det_np = np.linalg.det(C)

print(f"Recursive determinant: {det_recursive}")
print(f"NumPy determinant: {det_np}")
print(f"Are they equal? {np.isclose(det_recursive, det_np)}")
```

## Performance Considerations

For large matrices, the recursive method is very inefficient. Let's compare the performance with NumPy's implementation:

```python
import time

def time_det_calculation(method, A, n_trials=10):
    """Time the determinant calculation using the given method."""
    start = time.time()
    for _ in range(n_trials):
        det = method(A)
    end = time.time()
    return (end - start) / n_trials

# Test with different matrix sizes
sizes = [2, 5, 10, 20, 50, 100]
times_recursive = []
times_np = []

for size in sizes:
    # Generate a random matrix
    A = np.random.rand(size, size)
    
    # Time recursive implementation (only for small matrices)
    if size <= 10:
        time_rec = time_det_calculation(recursive_det, A)
        times_recursive.append(time_rec)
    else:
        times_recursive.append(float('nan'))
    
    # Time NumPy's implementation
    time_np = time_det_calculation(np.linalg.det, A)
    times_np.append(time_np)
    
    print(f"Size {size}x{size}: Recursive={times_recursive[-1]:.6f}s, NumPy={time_np:.6f}s")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes[:len(times_recursive)], times_recursive, 'o-', label='Recursive')
plt.plot(sizes, times_np, 's-', label='NumPy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix Size (n×n)')
plt.ylabel('Time (seconds)')
plt.title('Determinant Calculation Performance')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

## Practice Exercises

1. **Determinant Properties**: Create a function that verifies the following properties for random matrices:
   - det(AB) = det(A)det(B)
   - det(Aᵀ) = det(A)
   - det(cA) = cⁿ det(A) where n is the size of the matrix

2. **Volume of a Parallelepiped**: Write a function that calculates the volume of a parallelepiped defined by three 3D vectors using the determinant.

3. **Eigenvalue Connection**: The determinant of a matrix equals the product of its eigenvalues. Verify this property for several random matrices.

4. **Matrix Inversion**: Implement a function that uses the adjugate matrix and determinant to compute the inverse of a matrix. Compare its performance and accuracy with `np.linalg.inv`.

5. **Permutation Matrices**: Create a function that generates a random permutation matrix and verifies that its determinant is always ±1.

---

Next: [Eigenvalues and Eigenvectors →](07-eigenvalues-eigenvectors.md)
