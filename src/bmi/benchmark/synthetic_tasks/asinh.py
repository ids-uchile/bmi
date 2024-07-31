from typing import Optional

import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from samplers._synthetic_transformed import TransformedSampler
from bmi.benchmark.task import Task


def transform_asinh_task(
    base_task: Task,
    task_name: Optional[str] = None,
    ) -> Task:
    
    base_sampler = base_task.sampler

    sampler = TransformedSampler(
        base_sampler=base_sampler,
        transform_x=jnp.arcsinh,
        transform_y=None,
        )

    return Task(
        sampler=sampler,
        task_id=f"asinh-{base_task.id}",
        task_name=task_name or f"Asinh @ {base_task.name}",
        task_params=base_task.params
        )
