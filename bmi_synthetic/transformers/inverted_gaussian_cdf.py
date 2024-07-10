import numpy as np
from transformers._transformed import TransformedSampler

from typing import Optional
from scipy.stats import norm
from utils import sample_scale
from bmi.benchmark.task import Task

def gaussian_cdf(
    samples: np.ndarray,
    ) -> np.ndarray:
    """
    Transform samples from a uniform distribution to a multivariate normal distribution using the inverse CDF.
    
    Parameters
    ----------
    samples : np.ndarray
        Samples from a uniform distribution.

    Returns
    -------
    np.ndarray
        Transformed samples following a standard normal distribution.

    """
    # Scale samples using sample_scale function
    samples = sample_scale(samples)
    # Transform scaled samples using the inverse CDF of the normal distribution 
    normal_samples = norm.ppf(q=samples)
    
    return normal_samples


def transform_gaussian_cdf_task(
    base_task: Task,
    task_name: Optional[str] = None,
    ) -> Task:
    
    base_sampler = base_task.sampler
    
    sampler = TransformedSampler(
        base_sampler=base_sampler,
        transform_x=gaussian_cdf,
        transform_y=None,
        vectorise=False,
        )
        
    return Task(
        sampler=sampler,
        task_id=f"gaussian_inverted_cdf-{base_task.id}",
        task_name=task_name or f"Gaussian Inverted CDF @  {base_task.name}",
        task_params=base_task.params
        )