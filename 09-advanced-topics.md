# Advanced Topics in Linear Algebra

## Singular Value Decomposition (SVD)

### Introduction to SVD
Any real $m \times n$ matrix $A$ can be decomposed as:

$$A = U\Sigma V^T$$

where:
- $U$ is an $m \times m$ orthogonal matrix (left singular vectors)
- $\Sigma$ is an $m \times n$ diagonal matrix with non-negative real numbers on the diagonal (singular values)
- $V^T$ is an $n \times n$ orthogonal matrix (right singular vectors, transposed)

### Computing SVD in Python

```python
import numpy as np
import matplotlib.pyplot as plt

def svd_example():
    # Create a sample matrix
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Reconstruct the matrix
    Sigma = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    
    print("Original matrix:")
    print(A)
    print("\nReconstructed matrix:")
    print(A_reconstructed)
    print("\nSingular values:", S)
    
    return U, S, Vt, A

# Run the example
U, S, Vt, A = svd_example()
```

### Low-Rank Approximation

```python
def low_rank_approximation(A, k):
    """Compute rank-k approximation of matrix A using SVD."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only the first k singular values/vectors
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Reconstruct the approximation
    A_k = U_k @ S_k @ Vt_k
    
    return A_k

# Example: Image compression using SVD
def compress_image(image_path, k_values):
    """Demonstrate image compression using SVD."""
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=float)
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, len(k_values) + 1, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Plot compressed images for different k values
    U, S, Vt = np.linalg.svd(img_array, full_matrices=False)
    
    for i, k in enumerate(k_values, 1):
        # Compute rank-k approximation
        img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # Plot
        plt.subplot(1, len(k_values) + 1, i + 1)
        plt.imshow(img_compressed, cmap='gray')
        plt.title(f'k = {k}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage (uncomment to run with an actual image)
# compress_image('example.jpg', k_values=[1, 5, 20, 50])
```

## Principal Component Analysis (PCA)

### PCA from Scratch

```python
def pca_from_scratch(X, n_components=2):
    """
    Perform PCA on the input data.
    
    Args:
        X: Input data matrix (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        X_pca: Projected data (n_samples, n_components)
        explained_variance: Explained variance ratio
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    
    # Select the top n_components
    components = eigenvectors[:, :n_components]
    
    # Project the data
    X_pca = X_centered @ components
    
    # Calculate explained variance ratio
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance

# Example usage
def pca_demo():
    # Generate sample data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 2]]
    X = np.random.multivariate_normal(mean, cov, 100)
    
    # Apply PCA
    X_pca, explained_variance = pca_from_scratch(X, n_components=2)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    plt.title("Original Data")
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.title(f"After PCA\nExplained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Run the PCA demo
pca_demo()
```

## Matrix Factorization Techniques

### Non-negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF

def nmf_example():
    # Create a sample non-negative matrix
    X = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    # Apply NMF
    n_components = 2
    model = NMF(n_components=n_components, init='random', random_state=42)
    W = model.fit_transform(X)
    H = model.components_
    
    # Reconstruct the matrix
    X_reconstructed = W @ H
    
    print("Original matrix:")
    print(X)
    print("\nReconstructed matrix:")
    print(X_reconstructed)
    
    return W, H

# Run NMF example
W, H = nmf_example()
```

## Sparse Matrices

### Working with Sparse Matrices

```python
from scipy import sparse

def sparse_matrix_example():
    # Create a dense matrix
    dense_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 4]
    ])
    
    # Convert to sparse matrix (CSR format)
    sparse_matrix = sparse.csr_matrix(dense_matrix)
    
    print("Dense matrix:")
    print(dense_matrix)
    
    print("\nSparse matrix (CSR format):")
    print(sparse_matrix)
    
    # Convert back to dense
    dense_again = sparse_matrix.toarray()
    
    print("\nConverted back to dense:")
    print(dense_again)
    
    # Sparse matrix operations
    print("\nSparse matrix multiplication:")
    result = sparse_matrix @ sparse_matrix
    print(result.toarray())

# Run sparse matrix example
sparse_matrix_example()
```

## Applications in Data Science

### Recommender Systems with Matrix Factorization

```python
def simple_recommender():
    # User-item interaction matrix (rows=users, columns=items)
    # Values represent ratings (0 means no rating)
    ratings = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    
    # Apply NMF for matrix factorization
    n_components = 2
    model = NMF(n_components=n_components, init='random', random_state=42)
    user_features = model.fit_transform(ratings)
    item_features = model.components_
    
    # Reconstruct the ratings matrix
    predicted_ratings = user_features @ item_features
    
    print("Original ratings:")
    print(ratings)
    print("\nPredicted ratings:")
    print(predicted_ratings.round(2))
    
    # Make recommendations for user 0
    user_id = 0
    print(f"\nRecommendations for user {user_id}:")
    # Find items not yet rated by the user
    unrated_items = np.where(ratings[user_id] == 0)[0]
    # Get predicted ratings for unrated items
    for item_id in unrated_items:
        print(f"  Item {item_id}: predicted rating = {predicted_ratings[user_id, item_id]:.2f}")

# Run the recommender example
simple_recommender()
```

## Practice Exercises

1. **SVD Properties**
   a) Implement a function to compute the pseudoinverse of a matrix using SVD
   b) Test it on a non-square matrix and verify the Moore-Penrose conditions

2. **PCA Implementation**
   a) Extend the PCA implementation to handle data standardization
   b) Apply it to the Iris dataset and visualize the first two principal components

3. **Matrix Completion**
   Implement a simple matrix completion algorithm using SVD to fill in missing values

4. **Text Analysis with SVD**
   a) Create a term-document matrix from a set of text documents
   b) Apply SVD for latent semantic analysis (LSA)
   c) Find similar documents using the reduced-dimensional representation

5. **Eigenfaces**
   a) Load a dataset of face images
   b) Perform PCA to extract eigenfaces
   c) Reconstruct faces using a subset of principal components

6. **Sparse Coding**
   Implement a simple sparse coding algorithm using the Lasso (L1 regularization)

7. **Non-negative Matrix Factorization**
   a) Apply NMF to the 20 Newsgroups dataset
   b) Interpret the learned components as topics

8. **Dimensionality Reduction Comparison**
   Compare PCA, t-SNE, and UMAP for visualizing high-dimensional data

9. **Kernel PCA**
   Implement Kernel PCA and apply it to non-linearly separable data

10. **Robust PCA**
    Implement the Robust PCA algorithm to separate a matrix into low-rank and sparse components
