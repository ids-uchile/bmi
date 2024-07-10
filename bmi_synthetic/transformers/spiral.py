from typing import Optional

from transformers._transformed import TransformedSampler
from bmi.benchmark.task import Task
import bmi.transforms._rotate as rt


def transform_spiral_task(
    base_task: Task,
    speed: float = 1.0,
    x_dim: int = 2,
    y_dim: int = 3,
    task_name: Optional[str] = None,
    normalize_speed: bool = True
    ) -> Task:

    base_sampler = base_task.sampler
    
    # Generate rotation generators for x and y dimensions
    generator_x = rt.so_generator(x_dim, 0, 1)
    generator_y = rt.so_generator(y_dim, 1, 2)
    
    # Normalize speed by dimensions if specified
    if normalize_speed: 
        speed_x = speed / x_dim
        speed_y = speed / y_dim
    else:
        speed_x = speed
        speed_y = speed
    
    # Create spiral transformations for x and y
    transform_x = rt.Spiral(generator=generator_x, speed=speed_x)
    transform_y = rt.Spiral(generator=generator_y, speed=speed_y)
    
    spiral_sampler = TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform_x,
        transform_y=transform_y,
    )

    return Task(
        sampler=spiral_sampler,
        task_id=f"spiral-{base_task.id}",
        task_name=task_name or f"Spiral @ {base_task.name}",
        task_params=base_task.params
        | {"speed": speed, "normalize_speed": normalize_speed},
    )
