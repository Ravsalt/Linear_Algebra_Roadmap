# Introduction to Linear Algebra

## What is Linear Algebra?

Linear algebra is the branch of mathematics concerning linear equations, linear functions, and their representations through matrices and vector spaces. It is fundamental to modern mathematics and its applications in science and engineering.

### Core Concepts

- **Vectors**: Ordered lists of numbers (scalars) that can represent points in space
- **Matrices**: Rectangular arrays of numbers that can represent linear transformations
- **Linear Equations**: Equations of the form $A\mathbf{x} = \mathbf{b}$
- **Vector Spaces**: Collections of vectors that can be added together and multiplied by scalars

## Importance of Linear Algebra

Linear algebra is essential because:

1. **Foundation for Advanced Mathematics**
   - Used in calculus, differential equations, and functional analysis
   - Basis for modern geometry and topology

2. **Scientific Computing**
   - Core of numerical methods
   - Used in computer graphics and image processing

3. **Data Science and Machine Learning**
   - Fundamental for understanding algorithms
   - Used in neural networks and data transformations

## Real-world Applications

### Computer Graphics

```python
import numpy as np

# 2D Rotation matrix
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Rotate a point (1, 0) by 90 degrees
point = np.array([1, 0])
rotated = rotation_matrix(np.pi/2) @ point
print(f"Rotated point: {rotated}")
```

### Data Science

```python
import numpy as np
from sklearn.decomposition import PCA

# Dimensionality reduction using PCA
X = np.random.rand(100, 3)  # 100 samples, 3 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Reduced shape: {X_reduced.shape}")
```

### Physics and Engineering

- Solving systems of linear equations in circuit analysis
- Quantum mechanics (state vectors and operators)
- Structural analysis in civil engineering

## Mathematical Preliminaries

### Basic Notation

- Scalars: $a, b, c \in \mathbb{R}$ or $\mathbb{C}$
- Vectors: $\mathbf{v} \in \mathbb{R}^n$
- Matrices: $A \in \mathbb{R}^{m \times n}$
- Transpose: $A^T$
- Identity matrix: $I$

### Common Operations

1. **Vector Addition**

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

2. **Scalar Multiplication**

$$
c\mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \\ \vdots \\ c v_n \end{bmatrix}
$$

3. **Dot Product**

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$

## Practice Exercises

---

## Learning Linear Algebra Effectively: Mental Models in Action

To master linear algebra, use these proven mental models:

- **First Principles Thinking:**
  - Break down a concept (e.g., matrix multiplication) to its basic rules and build understanding from the ground up.
  - *Prompt:* "What are the most fundamental rules behind this operation?"

- **Inversion:**
  - Instead of asking "How do I solve this problem?" ask "How could I get stuck or make a mistake?" and avoid those pitfalls.
  - *Prompt:* "What would make this problem unsolvable?"

- **Long-Term Thinking:**
  - Focus on building skills that will compound (e.g., understanding vector spaces helps with machine learning later).
  - *Prompt:* "How will mastering this help me in future math or data science work?"

- **Then What? (Second-Order Thinking):**
  - After learning a new method, ask "Then what?" to explore consequences (e.g., "If I use Gaussian elimination, then what happens to the matrix?")
  - *Prompt:* "What is the next step or effect after this calculation?"

- **Feynman Technique:**
  - Teach a linear algebra concept (like determinants) in your own words or to a friend. If you get stuck, identify gaps and review.
  - *Prompt:* "Could I explain this to a 10-year-old? Where would they get confused?"

Use these models as you work through the rest of this tutorial for deeper understanding and better problem-solving.

---

1. Compute the sum of vectors $\mathbf{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$.

2. Write a Python function to compute the dot product of two vectors without using NumPy.

3. Explain in your own words why linear algebra is important in machine learning.

---

Next: [Basic Concepts →](02-basic-concepts.md)
