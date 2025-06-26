# Eigenvalues and Eigenvectors

## Introduction

Eigenvalues and eigenvectors are fundamental concepts in linear algebra with wide applications in various fields. They provide deep insights into the nature of linear transformations.

## Definitions

### Eigenvalues and Eigenvectors
For a square matrix $A$, a non-zero vector $\mathbf{v}$ is an eigenvector if:

$$A\mathbf{v} = \lambda \mathbf{v}$$

where $\lambda$ is the corresponding eigenvalue.

### Finding Eigenvalues and Eigenvectors in Python

```python
import numpy as np

def find_eigen(A):
    """
    Find eigenvalues and eigenvectors of a matrix.
    Returns eigenvalues and corresponding eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

# Example matrix
A = np.array([
    [4, 1],
    [2, 3]
])

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = find_eigen(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# Verify Av = λv for the first eigenpair
idx = 0
v = eigenvectors[:, idx].reshape(-1, 1)
Av = A @ v
lambdav = eigenvalues[idx] * v

print("\nVerification:")
print("A * v:", Av.flatten())
print(f"λ * v: {eigenvalues[idx]} * {v.flatten()} =", lambdav.flatten())
```

## Characteristic Polynomial

The characteristic polynomial of a matrix $A$ is given by:

$$p(\lambda) = \det(A - \lambda I) = 0$$

### Finding Characteristic Polynomial

```python
def characteristic_polynomial(A):
    """Compute the coefficients of the characteristic polynomial."""
    n = A.shape[0]
    # For larger matrices, use np.poly(A) or symbolic computation
    return np.poly(A)

# Get characteristic polynomial coefficients
coeffs = characteristic_polynomial(A)
print("Characteristic polynomial coefficients:", coeffs)
```

## Diagonalization

A matrix $A$ is diagonalizable if it can be written as:

$$A = PDP^{-1}$$

where $D$ is a diagonal matrix of eigenvalues and $P$ is a matrix of eigenvectors.

### Diagonalization in Python

```python
def diagonalize(A):
    """
    Diagonalize matrix A if possible.
    Returns P, D such that A = P D P^(-1)
    """
    eigenvalues, P = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    
    # Verify diagonalization
    A_reconstructed = P @ D @ P_inv
    assert np.allclose(A, A_reconstructed), "Diagonalization failed!"
    
    return P, D

# Example usage
A = np.array([
    [4, 1],
    [2, 3]
])

P, D = diagonalize(A)
print("Matrix P (eigenvectors as columns):\n", P)
print("\nDiagonal matrix D (eigenvalues):\n", D)
```

## Applications

### Power Iteration Method

```python
def power_iteration(A, num_iterations=100, epsilon=1e-10):
    """
    Find the dominant eigenvalue and eigenvector using power iteration.
    """
    n = A.shape[0]
    b_k = np.random.rand(n)
    
    for _ in range(num_iterations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k
        
        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm
        
        # Check convergence
        if np.linalg.norm(A @ b_k - b_k1_norm * b_k) < epsilon:
            break
    
    # The eigenvalue is the Rayleigh quotient
    eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)
    
    return eigenvalue, b_k

# Example usage
dominant_eigenvalue, dominant_eigenvector = power_iteration(A)
print("\nDominant eigenvalue:", dominant_eigenvalue)
print("Corresponding eigenvector:", dominant_eigenvector)
```

## Practice Exercises

1. **Eigenvalue Verification**
   Given the matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$:
   a) Find its eigenvalues and eigenvectors by hand
   b) Verify your results using NumPy
   c) Diagonalize the matrix

2. **Power Method Implementation**
   Implement the inverse power method to find the smallest eigenvalue in magnitude of a matrix.

3. **Application: Markov Chains**
   A Markov chain has the transition matrix:
   $$P = \begin{bmatrix} 0.9 & 0.2 \\ 0.1 & 0.8 \end{bmatrix}$$
   a) Find the steady-state distribution (eigenvector for λ=1)
   b) Verify that the other eigenvalue has magnitude less than 1

4. **Symmetric Matrices**
   Show that for a real symmetric matrix:
   a) Eigenvalues are real
   b) Eigenvectors corresponding to distinct eigenvalues are orthogonal

5. **Cayley-Hamilton Theorem**
   Verify the Cayley-Hamilton theorem for the matrix:
   $$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$
   (The theorem states that every square matrix satisfies its own characteristic equation)

---

Next: [Inner Product Spaces →](08-inner-product-spaces.md)
