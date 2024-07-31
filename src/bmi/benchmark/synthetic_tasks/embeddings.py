import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from samplers._synthetic_transformed import TransformedSampler

from bmi.benchmark.task import Task
from utils.utils_synthetic import swissroll2d_batch


def transform_swissroll_task(
    base_task: Task,
    task_name: str,  
    ) -> Task:
    
    sampler = TransformedSampler(
        base_sampler=base_task.sampler,
        transform_x=swissroll2d_batch,
        add_dim_x=1,
        vectorise=False
        )

    return Task(
        sampler=sampler,
        task_id=f"swissroll_x-{base_task.id}",
        task_name=task_name,
        task_params=base_task.params,
        )
