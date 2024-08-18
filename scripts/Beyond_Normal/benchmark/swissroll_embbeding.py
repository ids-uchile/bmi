# Import libraries
import warnings
warnings.filterwarnings("ignore")

from typing import Dict
import numpy as np

import sys
import os

from bmi.transforms import normal_cdf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../src/bmi/benchmark')))

from synthetic_tasks.synthetic import task_synthetic
from synthetic_tasks.normal_cdf import transform_normal_cdf_task as uniform_margins 
from utils.utils_synthetic import swissroll2d_batch



def normal_cdf_multidim(x: np.ndarray) -> np.ndarray:
    """
    Transforms a multidimensional array to have a uniform 
    distribution in the range [0,1].
    """
    return np.vectorize(normal_cdf)(x)



def dim_augmented(num_swiss: int,
                  n_samples: int = 10_000,
                  savedata: bool = False
                  ) -> np.ndarray:
    
    """   
    Generate a dimensionality-augmented dataset using Swissroll embedding and synthetic distribution.
    
    The 'num_swiss' parameter indicates the number of swissrolls applied consecutively. Each swissroll
    doubles the number of dimensions of the input data. For example, if the original data is 5-dimensional,
    applying one swissroll will transform it into 10-dimensional data, applying two swissrolls will result
    in 20-dimensional data, applying 3 swissrolls will result in 40-dimensional data, etc.
    
    Args:
        num_swiss (int): The number of Swiss roll transformations to apply. Must be greater than zero.
        n_samples (int, optional): The number of samples of the data (default is 10_000).
        savedata (bool, optional): If True, the generated data will be saved to a file (default is False). 
        
    Returns:
        np.ndarray: The final reshaped and augmented Swiss roll data.
        
    Raises:
        ValueError: If `num_swiss` is zero.
    """
    # Ensure that the number of Swiss rolls to apply is valid
    if num_swiss == 0:
        raise ValueError("Sorry, the number must be greater than zero.")
    
    # Define a base probability array for the demonstration scenario.
    base_demo_probabilities: np.ndarray = np.array(
        [
            [[0.4, 0.05], [0.3, 0.0], [0.2, 0.0], [0.05, 0.0]],  
            [[0.0, 0.2], [0.0, 0.3], [0.3, 0.0], [0.2, 0.0]],   
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.7], [0.3, 0.0]],   
        ]
    )
    # Create a dictionary to define the 2D scenario for the synthetic distribution
    demo_2d_scenario: Dict[str, np.ndarray] = {
        # Probabilities for each symbol
        "sym_prob": np.array([0.2, 0.5, 0.3]),
        # Cell boundaries in the 2D space
        "cell_bound": np.array(
            [
                np.array([-0.5, 0.5, 1.5, 2.0, 3.5]),  
                np.array([1.0, 1.5, 2.5])
            ],
            dtype="object",
        ),
        # Cell probabilities copied from the base probabilities array
        "cell_prob": np.copy(a=base_demo_probabilities),
        }


    # Initialize an empty numpy array to store the study scenario probabilities
    study_scenario_probabilities: np.ndarray = np.empty(
        shape=base_demo_probabilities.shape[:-1] + (4,) + base_demo_probabilities.shape[-1:]
    )

    # Copy the base demo probabilities into different slices of the new array
    study_scenario_probabilities[..., 0, :] = np.copy(a=base_demo_probabilities)
    study_scenario_probabilities[..., 1, :] = np.copy(a=base_demo_probabilities)
    study_scenario_probabilities[..., 2, :] = np.copy(a=base_demo_probabilities)
    study_scenario_probabilities[..., 3, :] = np.copy(a=base_demo_probabilities)

    # Modify specific slices to zero out certain probabilities
    study_scenario_probabilities[0, :, 1, :] = 0.0
    study_scenario_probabilities[1, :, 2, :] = 0.0
    study_scenario_probabilities[2, :, 3, :] = 0.0

    # Normalize the probabilities across certain dimensions
    study_scenario_probabilities = study_scenario_probabilities / np.reshape(
        a=np.sum(a=study_scenario_probabilities, axis=(1, 2, 3)), newshape=(3,) + (1,) * 3
    )

    # Create a dictionary to define the scenario for the study distribution
    study_scenario: Dict[str, np.ndarray] = {
        # Use the symbol probabilities from the demo 2D scenario
        "sym_prob": demo_2d_scenario["sym_prob"],

        # Define the cell boundaries for the study scenario
        "cell_bound": np.array(
            [
                demo_2d_scenario["cell_bound"][0],  
                np.array([-2.0, 2.0]),  
                np.array([-1.0, 0.0, 0.3, 1.0, 3.0]),  
                np.array([4.0, 4.5]),  
                demo_2d_scenario["cell_bound"][-1],  
            ],
            dtype="object",
        ),
        
        # Reshape the cell probabilities to add new axes for the scenario
        "cell_prob": study_scenario_probabilities[..., np.newaxis, :, np.newaxis, :],
    }
    
    # Create a synthetic distribution object
    synthetic_dist_study = task_synthetic(
    cell_boundaries=study_scenario['cell_bound'],
    symbol_probabilities=study_scenario['sym_prob'],
    cell_probabilities=study_scenario['cell_prob']
    )
        
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(synthetic_dist_study)
    uniform_x_sample, _ = uniform_margins_map.sample(n_samples=n_samples, seed=1234)
    
    # Generate data for a Swiss roll embedding using the uniform samples
    X_swiss = swissroll2d_batch(x=uniform_x_sample)
    x_augmented = X_swiss
    
    # Perform additional Swiss roll transformations based on the 'num_swiss' parameter
    i=0
    while i < num_swiss-1:
        # Reshape and transpose the data for the next Swiss roll embedding
        x_reshaped = x_augmented.transpose(1, 0, 2).reshape(x_augmented.shape[1], x_augmented.shape[0]*x_augmented.shape[2])
        x_unif = normal_cdf_multidim(x_reshaped)
        new_swiss = swissroll2d_batch(x=x_unif)
        x_augmented = new_swiss
        i += 1
    
    # Final reshaping of the augmented data
    x_augmented_reshaped = x_augmented.transpose(1, 0, 2).reshape(x_augmented.shape[1], x_augmented.shape[0]*x_augmented.shape[2])
    
    # Save the generated data to a file if the 'savedata' flag is set to True
    if savedata:
        np.save(file="data_x_swiss.npy", arr=x_augmented_reshaped)
    
    return x_augmented_reshaped

