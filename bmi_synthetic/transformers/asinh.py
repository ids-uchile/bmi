from typing import Optional

import jax.numpy as jnp

from transformers._transformed import TransformedSampler
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
