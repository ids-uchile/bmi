import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from typing import Dict, Optional
from bmi.benchmark.task import Task


def plot_2d_figure(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    cell_boundaries: Optional[np.ndarray] = None,
    plot_cells: Optional[bool] = False,
    savefig: Optional[bool] = False,
    savedata: Optional[bool] = False,
    ) -> None:

    """Plot a 2D figure with cell boundaries and data points."""

    # Create a new figure and axes
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    
    # If you want to display the cells
    if plot_cells:
        # Get x and y boundaries for the cells
        x_bounds: np.ndarray = cell_boundaries[0]
        y_bounds: np.ndarray = cell_boundaries[1]

        # Plot vertical and horizontal lines for cell boundaries
        ax.vlines(x=x_bounds, ymin=y_bounds[0], ymax=y_bounds[-1], colors="r", linewidth=2)
        ax.hlines(y=y_bounds, xmin=x_bounds[0], xmax=x_bounds[-1], colors="r", linewidth=2)
                                    
    ax.set_xlabel(xlabel="Value of $X_1$")
    ax.set_ylabel(ylabel="Value of $X_2$")
    ax.set_title(title)

    # Scatter plot of data points
    ax.scatter(
        x=X[:, 0],
        y=X[:, 1],
        c=y,
        cmap="viridis",
        marker=".",
        edgecolors="none",
    )

    # Save figure if savefig is True
    if savefig:
        fig.savefig(fname="figure2d.pdf")
    
    # Save data if savedata is True
    if savedata:
        np.save(file="data_x_2d.npy", arr=X)
        np.save(file="data_y_2d.npy", arr=y)
        
    # Adjust layout and display the figure
    fig.tight_layout()
    plt.show(block=True)


def plot_3d_figure(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    cell_boundaries: Optional[np.ndarray] = None,
    savefig: Optional[bool] = False,
    savedata: Optional[bool] = False,
    ) -> None:
    
    """Plots the data for a 3D scenario of synthetic data."""
    
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection="3d")
    
    ax.scatter(
        X[:, 0],
        X[:, cell_boundaries.shape[0] // 2],
        X[:, cell_boundaries.shape[0] - 1],
        c=y,
        cmap="viridis",
        marker=".",
        edgecolors="none",
        alpha=0.5,
    )
    
    ax.set_xlabel(xlabel="Values of $X_1$")
    ax.set_ylabel(ylabel="Values of $X_3$")
    ax.set_zlabel(zlabel="Values of $X_5$")
    ax.set_title(title)
    
    # Save figure if savefig is True
    if savefig:
        fig.savefig(fname="figure3d.pdf")
        
    # Save data if savedata is True
    if savedata:
        np.save(file="data_x_3d.npy", arr=X)
        np.save(file="data_y_3d.npy", arr=y)
        
    fig.tight_layout()
    plt.show(block=True)
    
def plot_spiral(
    X: np.ndarray,
    y: np.ndarray,
    speed: int,
    title: str,
    savefig: Optional[bool] = False,
    savedata: Optional[bool] = False,
    ) -> None:

    """"""

    # Create a new figure and axes
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(6,6))
                                    
    ax.set_xlabel(xlabel="Value of $X_1$")
    ax.set_ylabel(ylabel="Value of $X_2$")
    ax.set_title(f"{title} (Speed {speed})")

    marker = "."
    # Plot the spiral data
    ax.scatter(
        X[0][:, 0],
        X[0][:, 1],
        marker=marker,
        s=1,
        c=y.ravel(),
        rasterized=True,
    )

    # Save figure if savefig is True
    if savefig:
        fig.savefig(fname="figure_spiral.pdf")
    
    # Save data if savedata is True
    if savedata:
        np.save(file="data_x_spiral.npy", arr=X)
        np.save(file="data_y_spiral.npy", arr=y)
        
    # Adjust layout and display the figure
    fig.tight_layout()
    plt.show(block=True)
    

def plot_swissroll(
    distribution: Task,
    scenario: Dict[str, np.ndarray],
    x_swiss: np.ndarray,
    y_swiss: np.ndarray,
    x_sample: np.ndarray,
    dim: int = 0,
    savefig: bool = False,
    savedata: bool = False,
    ) -> None:
    """
    Plot the Swiss roll in 3D.

    Parameters
    ----------
    distribution : Task
        An object representing the distribution to apply the transformation.
    scenario : Dict[str, np.ndarray]
        A dictionary containing scenario data.
    x_swiss : np.ndarray
        An array containing the Swiss roll transformed coordinates.
    y_swiss : np.ndarray
        An array containing the color values for each point in the Swiss roll.
    x_sample : np.ndarray
        An array containing the original sample points.
    dim : int, optional (default=0)
        The dimension along which to plot the Swiss roll.
        
    """
    # Create figure for the 3D plot
    fig_emb = plt.figure(figsize=(8, 6))
    ax_3d = fig_emb.add_subplot(111, projection='3d')

    # Plot the Swiss roll in 3D
    artist = ax_3d.scatter(x_swiss[dim][:,0], x_swiss[dim][:,1], x_sample[:,dim-1], c=y_swiss.ravel())
    artist.set_rasterized(True) # Rasterize the plot for better performance
    ax_3d.grid(False)
    # Turn off the panes (the gray background)
    ax_3d.xaxis.pane.set_edgecolor("k")
    ax_3d.yaxis.pane.set_edgecolor("k")
    ax_3d.zaxis.pane.set_edgecolor("k")
    ax_3d.xaxis._axinfo["juggled"] = (1, 2, 0)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.set_title(f"{distribution.name} {distribution.dim_x}-dimensional x {len(scenario['sym_prob'])} symbols")
    
    # Save figure if savefig is True
    if savefig:
        fig_emb.savefig(fname="figure_swiss.pdf")
    
    # Save data if savedata is True
    if savedata:
        np.save(file="data_x_swiss.npy", arr=x_swiss)
        np.save(file="data_y_swiss.npy", arr=y_swiss)
        
    plt.show(block=True)
    

def swissroll2d_batch(x: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the Swiss roll transformation to a batch of input values with multiple dimensions.

    Parameters
    ----------
    x : jnp.ndarray 
        An array representing numbers in the range [0, 1].

    Returns
    -------
    jnp.ndarray
        Transformed array after applying the Swiss roll transformation.
    """
    def transform(element):
        # Calculate the parameter t for the Swiss roll transformation
        t = 1.5 * jnp.pi * (1 + 2 * element)
        # Return the transformed coordinates normalized by 21.0
        return jnp.stack([t * jnp.cos(t), t * jnp.sin(t)]) / 21.0
    
    # Apply the transformation to each element along the last axis
    x_transformed = jnp.apply_along_axis(transform, 0, x)
    
    result = []
    # Iterate over the last dimension of the input array
    for i in range(x.shape[-1]):
        # Stack the transformed coordinates along the second axis
        dim_stacked = jnp.stack((x_transformed[0][:, i], x_transformed[1][:, i]), axis=1)
        result.append(dim_stacked)
        
    # Convert the list of stacked arrays back to a single array        
    result = jnp.array(result)
    return result


def sample_scale(samples: np.ndarray) -> np.ndarray:
    """
    Scale samples to ensure they are within the range (0, 1).

    Parameters
    ----------
    samples : np.ndarray
        Input samples to be scaled.

    Returns
    -------
    np.ndarray
        Processed samples scaled to be within the range (0, 1).
    """
    epsilon = 1e-6 # Small value to avoid exact 0 or 1
    processed_samples = []

    for i in range(samples.shape[1]):
        # Check if any sample in the current column is outside the range [0, 1]
        has_problems = np.any((samples[:,i] < 0) | (samples[:,i] > 1))
        min_val = np.min(samples[:,i])
        max_val = np.max(samples[:,i])
        if has_problems:
            # Scale samples to the range [0, 1]
            scaled_samples = (samples[:,i] - min_val) / (max_val - min_val)
            # Clip values to ensure they are within (epsilon, 1 - epsilon)
            scaled_samples = np.clip(scaled_samples, epsilon, 1 - epsilon)
            processed_samples.append(scaled_samples[:, np.newaxis])
        elif np.any((samples[:,i] == 0) | (samples[:,i] == 1)):
            # Clip values if any sample is exactly 0 or 1
            clipped_samples = np.clip(samples[:,i], epsilon, 1 - epsilon)
            processed_samples.append(clipped_samples[:, np.newaxis])
        else:
            # No scaling needed, just add the column
            processed_samples.append(samples[:,i][:, np.newaxis])
            
    # Concatenate all arrays 
    processed_samples = np.concatenate(processed_samples, axis=1)
    
    return processed_samples


def update_cell_bounds(
    samples: np.ndarray,
    scenario: Dict[str, np.ndarray],
    ) -> np.ndarray:
    """
    Update the cell boundaries in the 2D space based on the new samples.
    
    Parameters
    ----------
    samples : np.ndarray
        A 2D array of new samples.
    scenario :  Dict[str, np.ndarray]
        The scenario dictionary containing 'cell_bound' to be updated.

    Returns
    -------
    np.ndarray
        Array with new 'cell_bound'.

    Raises
    ------
    ValueError: 
        If the samples array does not have a shape of (n_samples, 2).
        
    """
    if samples.shape[1] != 2:
        raise ValueError("Samples should be a 2D array with shape (n_samples, 2)")
    
    # Determine new bounds
    x_min, x_max = np.min(samples[:, 0]), np.max(samples[:, 0])
    y_min, y_max = np.min(samples[:, 1]), np.max(samples[:, 1])
    
    # Get original bounds
    original_x_bounds = scenario['cell_bound'][0]
    original_y_bounds = scenario['cell_bound'][1]
    
    # Calculate original proportions
    original_x_proportions = np.diff(original_x_bounds) / (original_x_bounds[-1] - original_x_bounds[0])
    original_y_proportions = np.diff(original_y_bounds) / (original_y_bounds[-1] - original_y_bounds[0])
    
    # Calculate new ranges
    new_x_range = x_max - x_min
    new_y_range = y_max - y_min
    
    # Generate new bounds maintaining the original proportions
    new_x_bounds = np.cumsum(np.hstack(([x_min], original_x_proportions * new_x_range)))
    new_y_bounds = np.cumsum(np.hstack(([y_min], original_y_proportions * new_y_range)))
    
    # Update the scenario dictionary
    new_cell_bound = [new_x_bounds, new_y_bounds]
    
    return new_cell_bound



