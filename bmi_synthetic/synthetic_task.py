import numpy as np

from synthetic_sampler import SyntheticSampler
from bmi.benchmark.task import Task


def task_synthetic(
    cell_boundaries: np.ndarray,
    symbol_probabilities: np.ndarray,
    cell_probabilities: np.ndarray,
    ) -> Task:
    """
    Creates a Task object for a synthetic dataset generation task.

     Parameters
    ----------
    cell_boundaries : np.ndarray
        The boundaries of the cells for each dimension.
    symbol_probabilities : np.ndarray
        The probabilities of each symbol.
    cell_probabilities : np.ndarray
        The probabilities of each cell for each symbol.

    Returns
    -------
    Task
        A Task object representing the synthetic dataset generation task.
    """
    # Create a SyntheticSampler object with provided parameters
    sampler = SyntheticSampler(
        cell_boundaries=cell_boundaries,
        symbol_probabilities=symbol_probabilities,
        cell_probabilities=cell_probabilities,
        )
    
    # Calculate the number of symbols 
    y_symbols: int = len(symbol_probabilities)

    # Create and return a Task object with relevant metadata
    return Task(
        sampler=sampler,
        task_id=f"synthetic {sampler.dim_x}x{y_symbols}",
        task_name=f"Synthetic {sampler.dim_x}-dimensional × {y_symbols}-symbols"
        )

