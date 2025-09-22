import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gda_level_curves(x, y, params_plot, num_std=2):
    """
    Plots the level curves (ellipses) of the two Gaussian classes in 2D.
    
    Args:
        x: Input data, shape (m, 2)
        y: Labels, shape (m,)
        params_plot: Output from gda_binary_learn(...)[0] (for plotting)
        num_std: Number of standard deviations to scale ellipse
    """
    phi, mu0, mu1, cov = params_plot
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Plot the data points
    ax.scatter(x[y==0,0], x[y==0,1], c='blue', label='Class 0', alpha=0.5)
    ax.scatter(x[y==1,0], x[y==1,1], c='red', label='Class 1', alpha=0.5)
    
    # Function to draw an ellipse
    def draw_ellipse(mu, cov, ax, color='black'):
        # Eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        # Sort descending
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # Angle of ellipse
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # Width and height = 2 * num_std * sqrt(eigenvalues)
        width, height = 2 * num_std * np.sqrt(vals)
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                          edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)
    
    # Draw ellipses
    draw_ellipse(mu0, cov, ax, color='blue')
    draw_ellipse(mu1, cov, ax, color='red')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('GDA Gaussian Level Curves')
    ax.legend()
    ax.grid(True)
    plt.show()
