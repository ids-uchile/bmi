from typing import Optional

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from samplers._synthetic_transformed import TransformedSampler
from bmi.benchmark.task import Task
from bmi.transforms import half_cube


def transform_half_cube_task(
    base_task: Task,
    task_name: Optional[str] = None,
    ) -> Task:
    
    base_sampler = base_task.sampler

    sampler = TransformedSampler(
        base_sampler=base_sampler,
        transform_x=half_cube,
        transform_y=None,
    )

    return Task(
        sampler=sampler,
        task_id=f"half_cube-{base_task.id}",
        task_name=task_name or f"Half-cube @ {base_task.name}",
        task_params=base_task.params,
    )
