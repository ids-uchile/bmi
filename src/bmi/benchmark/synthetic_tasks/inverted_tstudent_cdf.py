import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from samplers._synthetic_transformed import TransformedSampler

from typing import Optional
from scipy.stats import t
from utils.utils_synthetic import sample_scale
from bmi.benchmark.task import Task

def t_student_cdf(
    samples: np.ndarray,
    df: int,
    ) -> np.ndarray:
    """
    Transform samples from a uniform distribution to a Student's t-distribution using the inverse CDF.
    
    Parameters
    ----------
    samples : np.ndarray
        Samples from a uniform distribution.
    df : int
        Degrees of freedom for the t-distribution.

    Returns
    -------
    np.ndarray
        Transformed samples following a Student's t-distribution.
    """
    # Scale samples using sample_scale function
    samples = sample_scale(samples)
    # Transform scaled samples using the inverse CDF of the t-student distribution 
    students_samples = t.ppf(q=samples, df=df)
    
    return students_samples


def transform_student_cdf_task(
    base_task: Task,
    df: int,
    task_name: Optional[str] = None,
    ) -> Task:

    base_sampler = base_task.sampler
    
    def student_with_df(samples: np.ndarray):
        return t_student_cdf(samples=samples, df=df)
   
    sampler = TransformedSampler(
        base_sampler=base_sampler,
        transform_x=student_with_df,
        transform_y=None,
        vectorise=False,
        )
   
    return Task(
        sampler=sampler,
        task_id=f"t-student_inverted_cdf-{base_task.id}",
        task_name=task_name or f"t-Student Inverted CDF @  {base_task.name}",
        task_params=base_task.params
        )