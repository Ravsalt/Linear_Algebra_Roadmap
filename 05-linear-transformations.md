# Linear Transformations

## Introduction

A linear transformation $T: V \rightarrow W$ between two vector spaces $V$ and $W$ is a function that satisfies:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(c\mathbf{v}) = cT(\mathbf{v})$

for all vectors $\mathbf{u}, \mathbf{v} \in V$ and all scalars $c$.

## Matrix Representation

Every linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be represented by an $m \times n$ matrix $A$ such that:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

### Finding the Matrix of a Linear Transformation

To find the matrix representation of a linear transformation:
1. Apply $T$ to each basis vector of the domain
2. The images form the columns of the matrix

```python
import numpy as np

def find_transformation_matrix(T, n, m):
    """
    Find the matrix representation of a linear transformation T: R^n -> R^m.
    T should be a function that takes a numpy array of shape (n,) and returns a numpy array of shape (m,).
    """
    # Standard basis for R^n
    e = np.eye(n)
    
    # Apply T to each basis vector
    A = np.column_stack([T(e[:, i]) for i in range(n)])
    
    return A

# Example: Rotation by 90 degrees counterclockwise in R^2
def rotation_90(v):
    return np.array([-v[1], v[0]])

# Find the matrix representation
A = find_transformation_matrix(rotation_90, 2, 2)
print("Rotation matrix (90°):")
print(A)

# Verify with a test vector
v = np.array([1, 0])
print(f"\nRotating {v} by 90°:")
print(A @ v)  # Should be [0, 1]
```

## Common Linear Transformations

### 1. Scaling

$$
\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} s_x x \\ s_y y \end{bmatrix}
$$

```python
def scaling(sx, sy):
    return np.array([
        [sx, 0],
        [0, sy]
    ])

# Scale x by 2, y by 0.5
S = scaling(2, 0.5)
print("Scaling matrix:")
print(S)
```

### 2. Rotation

Rotation by angle $\theta$ counterclockwise:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

```python
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ])

# 45 degree rotation
R = rotation(np.pi/4)
print("45° rotation matrix:")
print(R)
```

### 3. Shearing

Horizontal shear:

$$
\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
$$

```python
def horizontal_shear(k):
    return np.array([
        [1, k],
        [0, 1]
    ])
```

## Composition of Transformations

The composition of linear transformations corresponds to matrix multiplication:

$$
T_2 \circ T_1(\mathbf{v}) = T_2(T_1(\mathbf{v})) = B(A\mathbf{v}) = (BA)\mathbf{v}
$$

```python
# Define two transformations
T1 = scaling(2, 3)      # Scale x by 2, y by 3
T2 = rotation(np.pi/4)  # Rotate by 45 degrees

# Compose transformations (apply T1 first, then T2)
T_composed = T2 @ T1

print("Composed transformation matrix:")
print(T_composed)

# Apply to a vector
v = np.array([1, 0])
result = T_composed @ v
print(f"\nTransformed vector: {result}")
```

## Kernel and Image

### Kernel (Null Space)
The kernel of a linear transformation $T: V \rightarrow W$ is the set of all vectors in $V$ that map to the zero vector in $W$:

$$
\text{ker}(T) = \{\mathbf{v} \in V \mid T(\mathbf{v}) = \mathbf{0}\}
$$

### Image (Range)
The image of $T$ is the set of all possible outputs:

$$
\text{im}(T) = \{T(\mathbf{v}) \mid \mathbf{v} \in V\}
$$

### Computing Kernel and Image

```python
def kernel_and_image(A):
    """Find bases for the kernel and image of a matrix A."""
    from scipy.linalg import null_space
    
    # Kernel (null space)
    ker = null_space(A)
    
    # Image (column space)
    _, pivots = np.linalg.qr(A)
    im = A[:, pivots]
    
    return ker, im

# Example matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

ker, im = kernel_and_image(A)
print("Kernel basis (columns):")
print(ker)
print("\nImage basis (columns):")
print(im)
```

## Change of Basis

### Change of Basis Matrix
To change coordinates from basis $B$ to the standard basis, the change of basis matrix $P$ has the basis vectors of $B$ as its columns.

```python
def change_basis_matrix(basis_vectors):
    """Create a change of basis matrix from a list of basis vectors."""
    return np.column_stack(basis_vectors)

# Standard basis for R^2
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# New basis
b1 = np.array([1, 1])
b2 = np.array([-1, 1])

# Change of basis matrix from B to standard
P = change_basis_matrix([b1, b2])
print("Change of basis matrix:")
print(P)

# Convert coordinates from B to standard basis
v_b = np.array([2, 3])  # Vector in B basis
v_std = P @ v_b        # Same vector in standard basis
print(f"\nVector in standard basis: {v_std}")

# Convert from standard basis to B basis
v_b_again = np.linalg.inv(P) @ v_std
print(f"Back to B basis: {v_b_again}")
```

## Applications

### Image Processing
Linear transformations are widely used in image processing for operations like rotation, scaling, and shearing.

```python
import matplotlib.pyplot as plt
from scipy import ndimage

# Load an example image (using scipy's astronaut)
from scipy.misc import face
image = face()

# Apply a shear transformation
shear_matrix = np.array([
    [1, 0.3, 0],
    [0.2, 1, 0]
])

# Apply the transformation
sheared = ndimage.affine_transform(
    image,
    matrix=shear_matrix[:2, :2],
    offset=shear_matrix[:2, 2],
    output_shape=image.shape
)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Original")
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
plt.title("Sheared")
plt.imshow(sheared)
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Computer Graphics
In computer graphics, linear transformations are used to manipulate 3D objects and project them onto 2D screens.

## Practice Exercises

1. Find the matrix representation of the linear transformation \(T: \mathbb{R}^2 \rightarrow \mathbb{R}^2\) that reflects points across the line \(y = x\).

2. Let \(T: \mathbb{R}^3 \rightarrow \mathbb{R}^2\) be defined by \(T(x, y, z) = (2x - y + z, x + 3y - 2z)\). Find the standard matrix of \(T\).

3. Determine if the following transformation is linear:
   $$T\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x + 1\\y\end{pmatrix}$$

4. Given bases $B = \{\begin{bmatrix}1\\1\end{bmatrix}, \begin{bmatrix}-1\\1\end{bmatrix}\}$ and $C = \{\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}1\\1\end{bmatrix}\}$, find the change of basis matrix from $B$ to $C$.

5. Implement a function that composes any number of linear transformations (matrices) in the order they are given.

---

Next: [Determinants →](06-determinants.md)
