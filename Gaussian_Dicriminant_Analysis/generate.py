import numpy as np
import numpy as np

def generate_binary_2d_dataset(n_samples_per_class=50, random_seed=None):
    """
    Generates a 2D binary classification dataset.
    
    Args:
        n_samples_per_class: Number of samples per class
        random_seed: Seed for reproducibility
    
    Returns:
        xs: numpy array of shape (2*n_samples_per_class, 2)
        ys: numpy array of shape (2*n_samples_per_class,)
        mu0, mu1: class means
        cov0, cov1: class covariance matrices
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomly generate means (between 0 and 10)
    mu0 = np.random.uniform(0, 10, size=2)
    mu1 = np.random.uniform(0, 10, size=2)
    
    # Random positive-definite covariance matrices
    A0 = np.random.rand(2, 2)
    cov0 = A0 @ A0.T + np.eye(2)  # ensure positive definite
    A1=A0                         #As of now the Covariance Matrices are same for both the classes.
    # A1 = np.random.rand(2, 2)
    cov1 = A1 @ A1.T + np.eye(2)
    
    # Generate samples
    x0 = np.random.multivariate_normal(mu0, cov0, size=n_samples_per_class)
    x1 = np.random.multivariate_normal(mu1, cov1, size=n_samples_per_class)
    
    # Combine and create labels
    xs = np.vstack((x0, x1))
    ys = np.array([0]*n_samples_per_class + [1]*n_samples_per_class)
    
    # Shuffle dataset
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    xs, ys = xs[indices], ys[indices]
    
    return xs, ys, mu0, mu1, cov0, cov1
