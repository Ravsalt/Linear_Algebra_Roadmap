# Inner Product Spaces

## Introduction to Inner Products

An inner product on a vector space $V$ over $\mathbb{R}$ or $\mathbb{C}$ is a function that takes two vectors and returns a scalar, satisfying certain properties.

### Definition
For a real vector space, an inner product $\langle \cdot, \cdot \rangle$ satisfies:
1. Linearity in first argument: $\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$
2. Symmetry: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
3. Positive-definiteness: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality iff $\mathbf{v} = \mathbf{0}$

### Dot Product in $\mathbb{R}^n$

```python
import numpy as np

def dot_product(u, v):
    """Compute the dot product of two vectors."""
    return np.dot(u, v)

# Example vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

dot_uv = dot_product(u, v)
print(f"Dot product of u and v: {dot_uv}")
print(f"Using numpy.dot: {np.dot(u, v)}")
```

## Norm and Distance

### Vector Norm
For a vector $\mathbf{v} \in V$, the norm is defined as:
$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

### Distance Between Vectors
$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$$

### Implementation

```python
def vector_norm(v):
    """Compute the Euclidean norm of a vector."""
    return np.linalg.norm(v)  # or np.sqrt(np.dot(v, v))

def distance(u, v):
    """Compute the distance between two vectors."""
    return vector_norm(u - v)

# Example usage
print(f"Norm of u: {vector_norm(u)}")
print(f"Distance between u and v: {distance(u, v)}")
```

## Orthogonality and Orthonormal Bases

### Orthogonal Vectors
Two vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal if $\langle \mathbf{u}, \mathbf{v} \rangle = 0$.

### Orthonormal Basis
A set of vectors is orthonormal if they are all unit vectors and pairwise orthogonal.

### Checking Orthogonality

```python
def are_orthogonal(u, v, tol=1e-10):
    """Check if two vectors are orthogonal."""
    return abs(np.dot(u, v)) < tol

# Example orthogonal vectors
u1 = np.array([1, 0])
u2 = np.array([0, 1])
print(f"Are u1 and u2 orthogonal? {are_orthogonal(u1, u2)}")

# Example non-orthogonal vectors
v1 = np.array([1, 1])
v2 = np.array([1, 2])
print(f"Are v1 and v2 orthogonal? {are_orthogonal(v1, v2)}")
```

## The Gram-Schmidt Process

The Gram-Schmidt process converts a set of linearly independent vectors into an orthonormal set.

### Implementation

```python
def gram_schmidt(vectors, normalize=True):
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors.
    
    Args:
        vectors: List of vectors (as numpy arrays)
        normalize: If True, return orthonormal vectors; else, just orthogonal
        
    Returns:
        List of orthogonal/orthonormal vectors
    """
    basis = []
    for v in vectors:
        w = v.copy()
        for u in basis:
            w = w - np.dot(v, u) * u
        if not np.allclose(w, 0):
            if normalize:
                w = w / np.linalg.norm(w)
            basis.append(w)
    return basis

# Example usage
vectors = [
    np.array([1, 1, 1], dtype=float),
    np.array([1, 0, 1], dtype=float),
    np.array([1, 1, 0], dtype=float)
]

orthonormal_basis = gram_schmidt(vectors)
print("\nOrthonormal basis:")
for i, v in enumerate(orthonormal_basis):
    print(f"v{i+1}:", v)

# Verify orthonormality
print("\nVerification of orthonormality:")
for i in range(len(orthonormal_basis)):
    for j in range(i, len(orthonormal_basis)):
        dot = np.dot(orthonormal_basis[i], orthonormal_basis[j])
        if i == j:
            print(f"<v{i+1}, v{j+1}> = {dot:.2f} (should be ~1.0)")
        else:
            print(f"<v{i+1}, v{j+1}> = {dot:.2f} (should be ~0.0)")
```

## Projections

### Orthogonal Projection

```python
def project(u, v):
    """Project vector v onto vector u."""
    return (np.dot(u, v) / np.dot(u, u)) * u

def orthogonal_projection(v, basis):
    """Project vector v onto the subspace spanned by basis vectors."""
    projection = np.zeros_like(v, dtype=float)
    for u in basis:
        projection += project(u, v)
    return projection

# Example usage
v = np.array([1, 1, 1], dtype=float)
u = np.array([1, 0, 0], dtype=float)
proj = project(u, v)
print(f"\nProjection of {v} onto {u} is {proj}")

# Project onto a plane
basis = [
    np.array([1, 0, 0], dtype=float),
    np.array([0, 1, 0], dtype=float)
]
proj_plane = orthogonal_projection(v, basis)
print(f"Projection of {v} onto the xy-plane is {proj_plane}")
```

## Applications

### Least Squares Approximation

```python
def least_squares(A, b):
    """
    Solve the least squares problem min ||Ax - b||^2.
    Returns the vector x that minimizes the norm.
    """
    # Using the normal equations: A^T A x = A^T b
    return np.linalg.solve(A.T @ A, A.T @ b)

# Example: Fit a line to some data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1.1, 1.9, 3.1, 3.9, 5.2])

# We want to find a line y = mx + b that best fits the data
A = np.column_stack([x_data, np.ones_like(x_data)])
b = y_data

# Solve least squares problem
m, b = least_squares(A, b)
print(f"\nBest fit line: y = {m:.3f}x + {b:.3f}")

# Plot the results
import matplotlib.pyplot as plt
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, m * x_data + b, 'r-', label=f'Best fit: y = {m:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Least Squares Line Fitting')
plt.show()
```

## Practice Exercises

1. **Orthogonal Vectors**
   a) Find two non-zero vectors in $\mathbb{R}^3$ that are orthogonal to $\mathbf{v} = [1, 2, 3]^T$
   b) Verify your answer using the dot product

2. **Gram-Schmidt Process**
   Apply the Gram-Schmidt process to the vectors:
   $$\mathbf{v}_1 = \begin{bmatrix}1\\1\\0\end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix}1\\0\\1\end{bmatrix}, \mathbf{v}_3 = \begin{bmatrix}0\\1\\1\end{bmatrix}$$
   Verify that the resulting vectors are orthonormal.

3. **Projection**
   Find the orthogonal projection of $\mathbf{v} = [1, 2, 3]^T$ onto the plane defined by:
   $$x + y + z = 0$$
   
4. **Least Squares**
   Given the data points (1,2), (2,3), (3,5), (4,7):
   a) Find the best-fit line using least squares
   b) Calculate the sum of squared errors
   
5. **Function Spaces**
   Consider the vector space of continuous functions on $[-\pi, \pi]$ with inner product:
   $$\langle f, g \rangle = \int_{-\pi}^{\pi} f(x)g(x) dx$$
   Show that the functions $\{1, \sin(x), \cos(x)\}$ are pairwise orthogonal.

---

Next: [Advanced Topics →](09-advanced-topics.md)
