# %%
# Import libraries
import warnings
warnings.filterwarnings("ignore")

from typing import Dict
import numpy as np

from synthetic_task import task_synthetic
from utils import plot_2d_figure, plot_3d_figure, update_cell_bounds, plot_swissroll, plot_spiral

from transformers.half_cube import transform_half_cube_task as half_cube
from transformers.asinh import transform_asinh_task as asinh
from transformers.normal_cdf import transform_normal_cdf_task as uniform_margins

from transformers.inverted_gaussian_cdf import transform_gaussian_cdf_task as gaussian_cdf
from transformers.inverted_tstudent_cdf import transform_student_cdf_task as student_cdf

from transformers.wiggly import transform_wiggly_task as wiggly
from transformers.embeddings import transform_swissroll_task as swissroll
from transformers.spiral import transform_spiral_task as spiral

# %%
# 2 DIMENSIONAL x 2 SYMBOLS 

# Initialize an empty numpy array with 2 elements to store cell boundaries
singular_2d_bounds: np.ndarray = np.empty(shape=2, dtype="object")

# Assign boundaries for the 2D scenario
singular_2d_bounds[:] = [np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0])]

# Create a dictionary to define the scenario for the synthetic distribution
singular_2d_scenario: Dict[str, np.ndarray] = {
    # Probabilities for each symbol
    "sym_prob": np.array([0.5, 0.5]),
    # Cell boundaries in the 2D space
    "cell_bound": singular_2d_bounds,
    # Probabilities for each cell in the 2D grid, structured as 2x2 matrices
    "cell_prob": np.array([[[0.5, 0.0], [0.0, 0.5]], [[0.0, 0.5], [0.5, 0.0]]])
    }


# 2 DIMENSIONAL x 4 SYMBOLS

# Initialize an empty numpy array with 2 elements to store cell boundaries
simple_2d_bounds: np.ndarray = np.empty(shape=2, dtype="object")
# Assign boundaries for the 2D scenario
simple_2d_bounds[:] = [np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0])]
# Create a dictionary to define the scenario for the synthetic distribution
simple_2d_scenario: Dict[str, np.ndarray] = {
    # Probabilities for each symbol
    "sym_prob": np.array([0.25, 0.25, 0.25, 0.25]),
    # Cell boundaries in the 2D space
    "cell_bound": simple_2d_bounds,
    # Probabilities for each cell in the 2D grid, structured as 2x2 matrices
    "cell_prob": np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],  # First symbol's cell probabilities
            [[0.0, 1.0], [0.0, 0.0]],  # Second symbol's cell probabilities
            [[0.0, 0.0], [1.0, 0.0]],  # Third symbol's cell probabilities
            [[0.0, 0.0], [0.0, 1.0]],  # Fourth symbol's cell probabilities
        ]
    ),
}


# 2 DIMENSIONAL x 3 SYMBOLS

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

# 3 DIMENSIONAL x 8 SYMBOLS

# Initialize an empty numpy array with 3 elements to store cell boundaries
simple_3d_bounds: np.ndarray = np.empty(shape=3, dtype="object")
# Assign the same boundaries for each of the 3 dimensions
simple_3d_bounds[:] = [np.array([-1.0, 0.0, 1.0])] * 3

# Create a dictionary to define the scenario for the synthetic 3D distribution
simple_3d_scenario: Dict[str, np.ndarray] = {
    # Probabilities for each of the 8 symbols, each with equal probability of 0.125
    "sym_prob": np.ones(shape=8) * 0.125,
    # Cell boundaries in the 3D space
    "cell_bound": simple_3d_bounds,
    # Probabilities for each cell in the 3D grid, structured as 2x2x2 matrices
    "cell_prob": np.array(
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],  # First symbol's cell probabilities
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],  # Second symbol's cell probabilities
            [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],  # Third symbol's cell probabilities
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],  # Fourth symbol's cell probabilities
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],  # Fifth symbol's cell probabilities
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],  # Sixth symbol's cell probabilities
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],  # Seventh symbol's cell probabilities
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],  # Eighth symbol's cell probabilities
        ]
    ),
}


# 5 DIMENSIONAL x 3 SYMBOLS

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

scenarios = dict()

scenarios.update({
   "scenario1":singular_2d_scenario,
   "scenario2":simple_2d_scenario,
   "scenario3":demo_2d_scenario,
   "scenario4":simple_3d_scenario,
   "scenario5":study_scenario,
   })


if __name__ == "__main__":
   
    n_samples = 70_000
    SEED = 1234
   
    scenario = scenarios["scenario5"]
   
    # Create a synthetic distribution object
    synthetic_dist = task_synthetic(
      cell_boundaries=scenario['cell_bound'],
      symbol_probabilities=scenario['sym_prob'],
      cell_probabilities=scenario['cell_prob']
      )
   
    # Print elements of the metadata
    for element in synthetic_dist.sampler.get_metadata().items():
       print(f"{element[0]} : {element[1]}")
      
    # Sample 'n_samples' data points from the synthetic distribution 
    x_sample, y_sample = synthetic_dist.sample(n_samples=n_samples, seed=SEED)
   
    # # Plot the 2D figure of the sampled data points
    #    plot_2d_figure(
    #       X=x_sample,
    #       y=y_sample, 
    #       cell_boundaries=scenario['cell_bound'], 
    #       title=synthetic_dist.name,
    #       plot_cells=False,
    #       savefig=False,
    #       savedata=False
    #       )
    
    # Plot the 3D figure of the sampled data points
    # plot_3d_figure(
    #     X=x_sample,
    #     y=y_sample, 
    #     cell_boundaries=scenario['cell_bound'], 
    #     title=synthetic_dist.name,
    #     savefig=False,
    #     savedata=False
    #     )
    #####################################################################
    gaussian_cdf_transf = gaussian_cdf(base_task=synthetic_dist)
    gaussian_x_sample, gaussian_y_sample = gaussian_cdf_transf.sample(n_samples=n_samples, seed=SEED)

    # Plot the 2D figure of the sampled data points
    #    new_cell_bound = update_cell_bounds(samples=gaussian_x_sample, scenario=scenario)
    #    plot_2d_figure(
    #       X=gaussian_x_sample,                  
    #       y=gaussian_y_sample,                  
    #       cell_boundaries=new_cell_bound,       
    #       title=gaussian_cdf_transf.name,       
    #       plot_cells=False,
    #       savefig=False,
    #       savedata=False,                      
    #       )
    
    # Plot the 3D figure of the sampled data points
    plot_3d_figure(
       X=gaussian_x_sample,
       y=gaussian_y_sample, 
       cell_boundaries=scenario['cell_bound'], 
       title=gaussian_cdf_transf.name,
       savefig=False,
       savedata=True
       )
    #####################################################################
    # df = 3 # Degrees of freedom

    # student_cdf_transf = student_cdf(base_task=synthetic_dist, df=df)
    # student_x_sample, student_y_sample = student_cdf_transf.sample(n_samples=n_samples, seed=SEED)

    # Plot the 2D figure of the sampled data points
    #    new_cell_bound = update_cell_bounds(samples=student_x_sample, scenario=scenario)
    #    plot_2d_figure(
    #       X=student_x_sample,                   
    #       y=student_y_sample,                   
    #       cell_boundaries=new_cell_bound,       
    #       title=student_cdf_transf.name,        
    #       plot_cells=False,
    #       savefig=False,
    #       savedata=False,
    #       )
    
    # Plot the 3D figure of the sampled data points
    # plot_3d_figure(
    #    X=student_x_sample,                  
    #    y=student_y_sample,                           
    #    cell_boundaries=scenario['cell_bound'],      
    #    title=student_cdf_transf.name,  
    #    savefig=False,
    #    savedata=False
    #    )
    #####################################################################
    # Apply half-cube mapping
    # half_cube_map = half_cube(base_task=synthetic_dist)
    # half_cube_x_sample, half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)

    # Plot the 2D figure of the sampled data points
    # new_cell_bound = update_cell_bounds(samples=half_cube_x_sample, scenario=scenario)
    # plot_2d_figure(
    #    X=half_cube_x_sample,
    #    y=half_cube_y_sample, 
    #    cell_boundaries=new_cell_bound, 
    #    title=half_cube_map.name,
    #    plot_cells=False,
    #    savefig=False,
    #    savedata=False
    #    )
    
    # Plot the 3D figure of the sampled data points
    # plot_3d_figure(
    #    X=half_cube_x_sample,
    #    y=half_cube_y_sample, 
    #    cell_boundaries=scenario['cell_bound'], 
    #    title=half_cube_map.name,
    #    savefig=False,
    #    savedata=False
    #    )
    #####################################################################
    # Apply asinh mapping 
    # asinh_map = asinh(base_task=synthetic_dist)
    # asinh_x_sample, asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)

    # Plot the 2D figure of the sampled data points
    # new_cell_bound = update_cell_bounds(samples=asinh_x_sample, scenario=scenario)
    # plot_2d_figure(
    #    X=asinh_x_sample,
    #    y=asinh_y_sample, 
    #    cell_boundaries=new_cell_bound, 
    #    title=asinh_map.name,
    #    plot_cells=False,
    #    savefig=False,
    #    savedata=False
    #    )
    # Plot the 3D figure of the sampled data points
    # plot_3d_figure(
    #    X=asinh_x_sample,
    #    y=asinh_y_sample, 
    #    cell_boundaries=scenario['cell_bound'], 
    #    title=asinh_map.name,
    #    savefig=False,
    #    savedata=False
    #    )
    #####################################################################
    # Apply uniform margins transformation
    # uniform_margins_map = uniform_margins(base_task=synthetic_dist)
    # uniform_x_sample, uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)

    # Plot the 2D figure of the sampled data points
    # new_cell_bound = update_cell_bounds(samples=uniform_x_sample, scenario=scenario)
    # plot_2d_figure(
    #    X=uniform_x_sample,
    #    y=uniform_y_sample, 
    #    cell_boundaries=new_cell_bound, 
    #    title=uniform_margins_map.name,
    #    plot_cells=False,
    #    savefig=False,
    #    savedata=False
    #    )
    # Plot the 3D figure of the sampled data points
    # plot_3d_figure(
    #    X=uniform_x_sample,
    #    y=uniform_y_sample, 
    #    cell_boundaries=scenario['cell_bound'], 
    #    title=uniform_margins_map.name,
    #    savefig=False,
    #    savedata=False
    #    )
    #####################################################################
    # Apply wiggly transformation
    #    wiggly_map = wiggly(base_task=synthetic_dist)
    #    wiggly_x_sample, wiggly_y_sample = wiggly_map.sampler.sample_split(n_points=n_samples, rng=SEED, transformer="wiggly")

    # Plot the 2D figure of the sampled data points
    #    new_cell_bound = update_cell_bounds(samples=wiggly_x_sample, scenario=scenario)
    #    plot_2d_figure(
    #       X=wiggly_x_sample,
    #       y=wiggly_y_sample, 
    #       cell_boundaries=new_cell_bound, 
    #       title=wiggly_map.name,
    #       plot_cells=False,
    #       savefig=False,
    #       savedata=False
    #       )
    #####################################################################
    # Generate data for a Swiss roll embedding task
    # swissroll_emb = swissroll(base_task=uniform_margins_map, task_name="Swissroll @ Synthetic")
    # X_swiss, Y_swiss = swissroll_emb.sample(n_samples=n_samples, seed=SEED)

    # print(f"Total dimensions: {X_swiss.shape[0]*X_swiss.shape[-1]}")
    # # Create figure for the 3D plot
    # plot_swissroll(
    #     distribution=swissroll_emb,
    #     scenario=scenario,
    #     x_swiss=X_swiss,  
    #     y_swiss=Y_swiss,
    #     x_sample=uniform_x_sample,
    #     dim=0,
    #     savefig=False,
    #     savedata=False
    #     )
    # ###########################################################################
    # # Generate data for a Swiss roll embedding task
    #    dims = [2, 3]
    #    speed_list = [0.5, 1.0, 1.5]

    #    # Generate spiral data for the given speed
    #    spiral_diff = spiral(base_task=synthetic_dist, task_name="Spiral @ Synthetic", speed=speed_list[1], x_dim=dims[0], y_dim=dims[1])
    #    x_spiral, y_spiral = spiral_diff.sampler.sample_split(n_points=n_samples, rng=SEED, transformer="spiral", spiral_dims=dims)

    #    plot_spiral(
    #       X=x_spiral,
    #       y=y_spiral,
    #       speed=speed_list[1],
    #       title=spiral_diff.name,
    #       savefig=False, 
    #       savedata=False,
    #       )
# %%
