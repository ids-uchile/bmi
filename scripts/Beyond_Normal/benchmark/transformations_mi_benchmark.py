import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bmi

from typing import Dict
from sklearn.feature_selection import mutual_info_regression

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../src/bmi/benchmark')))

from synthetic_tasks.synthetic import task_synthetic

from synthetic_tasks.half_cube import transform_half_cube_task as half_cube
from synthetic_tasks.asinh import transform_asinh_task as asinh
from synthetic_tasks.normal_cdf import transform_normal_cdf_task as uniform_margins 

from synthetic_tasks.inverted_gaussian_cdf import transform_gaussian_cdf_task as gaussian_cdf
from synthetic_tasks.inverted_tstudent_cdf import transform_student_cdf_task as student_cdf



def bench_mi_transformations(title: bool = False,
                             add_noise: bool = False,
                             n_samples: int = 10_000,
                             savefig: bool = False
                             ) -> None:
    """
    Estimates the mutual information of each dimension on transformed synthetic
    data by applying the KSG estimator.
    
    Args:
        title (bool, optional): If you want to add title True if not False (default is False).
        add_noise (bool, optional): If you want to add noise to the target value True if not False (default es False).
        n_samples (int, optional): Number of samples of the data (default is 10_000).
        savefig (bool, optional): If you want to save the figure True otherwise False (default is False).
    """
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

    # Seed for random number generator to ensure reproducibility
    SEED = 1234

    # Create a synthetic distribution object
    synthetic_dist_study = task_synthetic(
    cell_boundaries=study_scenario['cell_bound'],
    symbol_probabilities=study_scenario['sym_prob'],
    cell_probabilities=study_scenario['cell_prob']
    )

    
    # BASE CASE
    
    base_x_sample, base_y_sample = synthetic_dist_study.sample(n_samples=n_samples, seed=SEED)
    # Apply half-cube mapping
    half_cube_map = half_cube(synthetic_dist_study)
    b_half_cube_x_sample, b_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    # Apply asinh mapping
    asinh_map = asinh(synthetic_dist_study)
    b_asinh_x_sample, b_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(synthetic_dist_study)
    b_uniform_x_sample, b_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)

    # GAUSSIAN DISTRIBUTION
    
    gaussian_cdf_transf = gaussian_cdf(base_task=synthetic_dist_study)
    gaussian_x_sample, gaussian_y_sample  = gaussian_cdf_transf.sample(n_samples=n_samples, seed=SEED)
    # Apply half-cube mapping
    half_cube_map = half_cube(gaussian_cdf_transf)
    g_half_cube_x_sample, g_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    # Apply asinh mapping
    asinh_map = asinh(gaussian_cdf_transf)
    g_asinh_x_sample, g_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(gaussian_cdf_transf)
    g_uniform_x_sample, g_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)

    # T-STUDENT DISTRIBUTION
    
    df = 3  # Degrees of freedom 
    student_cdf_transf = student_cdf(base_task=synthetic_dist_study, df=df)
    student_x_sample, student_y_sample = student_cdf_transf.sample(n_samples=n_samples, seed=SEED)
    # Apply half-cube mapping
    half_cube_map = half_cube(student_cdf_transf)
    t_half_cube_x_sample, t_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    # Apply asinh mapping
    asinh_map = asinh(student_cdf_transf)
    t_asinh_x_sample, t_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(student_cdf_transf)
    t_uniform_x_sample, t_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)

    # Dictionary to store MI estimates
    data_cases = {
        "Base case":{
            "Original": [],
            "Half-cube": [],
            "Asinh": [],
            "Uniform margins": []},
        "Gaussian distribution":{
            "Original": [],
            "Half-cube": [],
            "Asinh": [],
            "Uniform margins": []},
        "t-Student distribution":{
            "Original": [],
            "Half-cube": [],
            "Asinh": [],
            "Uniform margins": []}
        }
    
    if not add_noise:
        # Iteration over each dimension
        for j in range(base_x_sample.shape[1]):
            # Mutual Information Estimates for the Base Case
            base_EMI_KSG = mutual_info_regression(base_x_sample[:,j].reshape(-1,1), base_y_sample).item()/np.log(2)
            base_hc_EMI_KSG = mutual_info_regression(b_half_cube_x_sample[:,j].reshape(-1,1), b_half_cube_y_sample).item()/np.log(2)
            base_asinh_EMI_KSG = mutual_info_regression(b_asinh_x_sample[:,j].reshape(-1,1), b_asinh_y_sample).item()/np.log(2)
            base_unif_EMI_KSG = mutual_info_regression(b_uniform_x_sample[:,j].reshape(-1,1), b_uniform_y_sample).item()/np.log(2)
            
            data_cases['Base case']['Original'].append(base_EMI_KSG)
            data_cases['Base case']['Half-cube'].append(base_hc_EMI_KSG)
            data_cases['Base case']['Asinh'].append(base_asinh_EMI_KSG)
            data_cases['Base case']['Uniform margins'].append(base_unif_EMI_KSG)
            
            # Mutual Information Estimates for the Gaussian distribution case
            gaussian_EMI_KSG = mutual_info_regression(gaussian_x_sample[:,j].reshape(-1,1), gaussian_y_sample).item()/np.log(2)
            gaussian_hc_EMI_KSG = mutual_info_regression(g_half_cube_x_sample[:,j].reshape(-1,1), g_half_cube_y_sample).item()/np.log(2)
            gaussian_asinh_EMI_KSG = mutual_info_regression(g_asinh_x_sample[:,j].reshape(-1,1), g_asinh_y_sample).item()/np.log(2)
            gaussian_unif_EMI_KSG = mutual_info_regression(g_uniform_x_sample[:,j].reshape(-1,1), g_uniform_y_sample).item()/np.log(2)
            
            data_cases['Gaussian distribution']['Original'].append(gaussian_EMI_KSG)
            data_cases['Gaussian distribution']['Half-cube'].append(gaussian_hc_EMI_KSG)
            data_cases['Gaussian distribution']['Asinh'].append(gaussian_asinh_EMI_KSG)
            data_cases['Gaussian distribution']['Uniform margins'].append(gaussian_unif_EMI_KSG)
            
            # Mutual Information Estimates for the t-Student distribution case
            student_EMI_KSG = mutual_info_regression(student_x_sample[:,j].reshape(-1,1), student_y_sample).item()/np.log(2)
            student_hc_EMI_KSG = mutual_info_regression(t_half_cube_x_sample[:,j].reshape(-1,1), t_half_cube_y_sample).item()/np.log(2)
            student_asinh_EMI_KSG = mutual_info_regression(t_asinh_x_sample[:,j].reshape(-1,1), t_asinh_y_sample).item()/np.log(2)
            student_unif_EMI_KSG = mutual_info_regression(t_uniform_x_sample[:,j].reshape(-1,1), t_uniform_y_sample).item()/np.log(2)
            
            data_cases['t-Student distribution']['Original'].append(student_EMI_KSG)
            data_cases['t-Student distribution']['Half-cube'].append(student_hc_EMI_KSG)
            data_cases['t-Student distribution']['Asinh'].append(student_asinh_EMI_KSG)
            data_cases['t-Student distribution']['Uniform margins'].append(student_unif_EMI_KSG)
    else:
        # Generate uniform random noise to be added to the target variable
        noise = np.random.uniform(low=-0.1, high=0.1, size=base_y_sample.shape[0])
        # Iteration over each dimension
        for j in range(base_x_sample.shape[1]):
            # Mutual Information Estimates for the Base Case
            base_EMI_KSG = mutual_info_regression(base_x_sample[:,j].reshape(-1,1), (base_y_sample+noise)).item()/np.log(2)
            base_hc_EMI_KSG = mutual_info_regression(b_half_cube_x_sample[:,j].reshape(-1,1), (b_half_cube_y_sample+noise)).item()/np.log(2)
            base_asinh_EMI_KSG = mutual_info_regression(b_asinh_x_sample[:,j].reshape(-1,1), (b_asinh_y_sample+noise)).item()/np.log(2)
            base_unif_EMI_KSG = mutual_info_regression(b_uniform_x_sample[:,j].reshape(-1,1), (b_uniform_y_sample+noise)).item()/np.log(2)
            
            data_cases['Base case']['Original'].append(base_EMI_KSG)
            data_cases['Base case']['Half-cube'].append(base_hc_EMI_KSG)
            data_cases['Base case']['Asinh'].append(base_asinh_EMI_KSG)
            data_cases['Base case']['Uniform margins'].append(base_unif_EMI_KSG)
            
            # Mutual Information Estimates for the Gaussian distribution case
            gaussian_EMI_KSG = mutual_info_regression(gaussian_x_sample[:,j].reshape(-1,1), (gaussian_y_sample+noise)).item()/np.log(2)
            gaussian_hc_EMI_KSG = mutual_info_regression(g_half_cube_x_sample[:,j].reshape(-1,1), (g_half_cube_y_sample+noise)).item()/np.log(2)
            gaussian_asinh_EMI_KSG = mutual_info_regression(g_asinh_x_sample[:,j].reshape(-1,1), (g_asinh_y_sample+noise)).item()/np.log(2)
            gaussian_unif_EMI_KSG = mutual_info_regression(g_uniform_x_sample[:,j].reshape(-1,1), (g_uniform_y_sample+noise)).item()/np.log(2)
            
            data_cases['Gaussian distribution']['Original'].append(gaussian_EMI_KSG)
            data_cases['Gaussian distribution']['Half-cube'].append(gaussian_hc_EMI_KSG)
            data_cases['Gaussian distribution']['Asinh'].append(gaussian_asinh_EMI_KSG)
            data_cases['Gaussian distribution']['Uniform margins'].append(gaussian_unif_EMI_KSG)
            
            # Mutual Information Estimates for the t-Student distribution case
            student_EMI_KSG = mutual_info_regression(student_x_sample[:,j].reshape(-1,1), (student_y_sample+noise)).item()/np.log(2)
            student_hc_EMI_KSG = mutual_info_regression(t_half_cube_x_sample[:,j].reshape(-1,1), (t_half_cube_y_sample+noise)).item()/np.log(2)
            student_asinh_EMI_KSG = mutual_info_regression(t_asinh_x_sample[:,j].reshape(-1,1), (t_asinh_y_sample+noise)).item()/np.log(2)
            student_unif_EMI_KSG = mutual_info_regression(t_uniform_x_sample[:,j].reshape(-1,1), (t_uniform_y_sample+noise)).item()/np.log(2)
            
            data_cases['t-Student distribution']['Original'].append(student_EMI_KSG)
            data_cases['t-Student distribution']['Half-cube'].append(student_hc_EMI_KSG)
            data_cases['t-Student distribution']['Asinh'].append(student_asinh_EMI_KSG)
            data_cases['t-Student distribution']['Uniform margins'].append(student_unif_EMI_KSG)
        
    
        
    # Iterates over the 'data_cases' dictionary and flattens the data into a list.
    data_flat = []
    for key1, subdict in data_cases.items():
        for key2, values in subdict.items():
            combined_key = f"{key1} @ {key2}"
            data_flat.append([combined_key] + values)

    # Create DataFrame
    df = pd.DataFrame(data_flat, columns=['Description', 'Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4', 'Dimension 5'])

    # Estimate the ground truth of the mutual information for each dimension
    ground_truth = [synthetic_dist_study.sampler.get_metadata()[f"mi_{i}"] for i in range(1, base_x_sample.shape[1]+1)]
    # Add the ground truth as the first row in the DataFrame
    df.loc[0] = ["Ground truth", ground_truth[0], ground_truth[1], ground_truth[2], ground_truth[3], ground_truth[4]] 
    
    # Plot a heatmap for each dimension
    df_values = df.set_index('Description')
    fig, axes = plt.subplots(1, len(df_values.columns), figsize=(18, 6), sharey=True)
    for i, column in enumerate(df_values.columns):
        diffs = df_values[[column]].copy()
        diffs = (diffs - diffs.iloc[0, 0]) # Subtract the ground truth value from the estimated MI values
        # Plot the heatmap for each dimension
        # if i == 1 and not add_noise:
        #     sns.heatmap(diffs, cmap="BuPu",annot=df_values[[column]].values.reshape(-1, 1), ax=axes[i], cbar=True, linewidths=0.5, fmt='.6f')
        # else:
        sns.heatmap(diffs, cmap="BuPu",annot=df_values[[column]].values.reshape(-1, 1), ax=axes[i], cbar=True, linewidths=0.5, fmt='.6f', center=0)
        axes[i].set_ylabel('')
    
    # Add a title if the 'title' argument is True 
    if title:
        if not add_noise:
            fig.suptitle("Kraskov–Stögbauer–Grassberger (KSG) estimation", fontsize=14)
        else:
            fig.suptitle("Kraskov–Stögbauer–Grassberger (KSG) estimation with noise", fontsize=14)
    
    # Save the figure as a PDF if savefig is set to True.
    if savefig:
        fig.savefig(fname="compare_EMI_transforms.pdf")
        
    plt.tight_layout()
    plt.show()





def bench_joint_mi_transformations(ground_truth: np.ndarray,
                                   title: bool = False,
                                   n_samples: int = 10_000,
                                   savefig: bool = False
                                   ) -> None:
    """
    Estimates the joint mutual information of clear and noisy 
    transformed synthetic data by applying the KSG estimator.
    
    Args:
        ground_truth (np.ndarray): Ground truth values of mutual information for comparison.
        title (bool, optional): If you want to add title True if not False (default is False).
        n_samples (int, optional): Number of samples of the data (default is 10_000).
        savefig (bool, optional): If you want to save the figure True otherwise False (default is False).
    """
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

    # Seed for random number generator to ensure reproducibility
    SEED = 1234

    # Create a synthetic distribution object
    synthetic_dist_study = task_synthetic(
    cell_boundaries=study_scenario['cell_bound'],
    symbol_probabilities=study_scenario['sym_prob'],
    cell_probabilities=study_scenario['cell_prob']
    )
    
    # Generate uniform random noise to be added to the target variable
    noise = np.random.uniform(low=-0.1, high=0.1, size=n_samples)
     
    # Kraskov, Stögbauer and Grassberger estimator
    estimator = bmi.estimators.KSGEnsembleFirstEstimator()
    
    # Mutual Information Neural Estimator
    # estimator = bmi.estimators.MINEEstimator()
    
    # BASE CASE
    
    # Apply half-cube mapping
    half_cube_map = half_cube(synthetic_dist_study)
    b_half_cube_x_sample, b_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    base_hc_EMI = estimator.estimate(b_half_cube_x_sample, b_half_cube_y_sample.reshape(-1,1))/np.log(2)
    base_hc_EMI_noise = estimator.estimate(b_half_cube_x_sample, (b_half_cube_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply asinh mapping
    asinh_map = asinh(synthetic_dist_study)
    b_asinh_x_sample, b_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    base_asinh_EMI = estimator.estimate(b_asinh_x_sample, b_asinh_y_sample.reshape(-1,1))/np.log(2)
    base_asinh_EMI_noise = estimator.estimate(b_asinh_x_sample, (b_asinh_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(synthetic_dist_study)
    b_uniform_x_sample, b_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)
    base_unif_EMI = estimator.estimate(b_uniform_x_sample, b_uniform_y_sample.reshape(-1,1))/np.log(2)
    base_unif_EMI_noise = estimator.estimate(b_uniform_x_sample, (b_uniform_y_sample+noise).reshape(-1,1))/np.log(2)
    
    
    # GAUSSIAN DISTRIBUTION
    
    gaussian_cdf_transf = gaussian_cdf(base_task=synthetic_dist_study)
    gaussian_x_sample, gaussian_y_sample  = gaussian_cdf_transf.sample(n_samples=n_samples, seed=SEED)
    gaussian_EMI = estimator.estimate(gaussian_x_sample, gaussian_y_sample.reshape(-1,1))/np.log(2)
    gaussian_EMI_noise = estimator.estimate(gaussian_x_sample, (gaussian_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply half-cube mapping
    half_cube_map = half_cube(gaussian_cdf_transf)
    g_half_cube_x_sample, g_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    gaussian_hc_EMI = estimator.estimate(g_half_cube_x_sample, g_half_cube_y_sample.reshape(-1,1))/np.log(2)
    gaussian_hc_EMI_noise = estimator.estimate(g_half_cube_x_sample, (g_half_cube_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply asinh mapping
    asinh_map = asinh(gaussian_cdf_transf)
    g_asinh_x_sample, g_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    gaussian_asinh_EMI = estimator.estimate(g_asinh_x_sample, g_asinh_y_sample.reshape(-1,1))/np.log(2)
    gaussian_asinh_EMI_noise = estimator.estimate(g_asinh_x_sample, (g_asinh_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(gaussian_cdf_transf)
    g_uniform_x_sample, g_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)
    gaussian_unif_EMI = estimator.estimate(g_uniform_x_sample, g_uniform_y_sample.reshape(-1,1))/np.log(2)
    gaussian_unif_EMI_noise = estimator.estimate(g_uniform_x_sample, (g_uniform_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # T-STUDENT DISTRIBUTION
    
    df = 3  # Degrees of freedom 
    student_cdf_transf = student_cdf(base_task=synthetic_dist_study, df=df)
    student_x_sample, student_y_sample = student_cdf_transf.sample(n_samples=n_samples, seed=SEED)
    student_EMI = estimator.estimate(student_x_sample, student_y_sample.reshape(-1,1))/np.log(2)
    student_EMI_noise = estimator.estimate(student_x_sample, (student_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply half-cube mapping
    half_cube_map = half_cube(student_cdf_transf)
    t_half_cube_x_sample, t_half_cube_y_sample = half_cube_map.sample(n_samples=n_samples, seed=SEED)
    student_hc_EMI = estimator.estimate(t_half_cube_x_sample, t_half_cube_y_sample.reshape(-1,1))/np.log(2)
    student_hc_EMI_noise = estimator.estimate(t_half_cube_x_sample, (t_half_cube_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply asinh mapping
    asinh_map = asinh(student_cdf_transf)
    t_asinh_x_sample, t_asinh_y_sample = asinh_map.sample(n_samples=n_samples, seed=SEED)
    student_asinh_EMI = estimator.estimate(t_asinh_x_sample, t_asinh_y_sample.reshape(-1,1))/np.log(2)
    student_asinh_EMI_noise = estimator.estimate(t_asinh_x_sample, (t_asinh_y_sample+noise).reshape(-1,1))/np.log(2)
    
    # Apply uniform margins transformation
    uniform_margins_map = uniform_margins(student_cdf_transf)
    t_uniform_x_sample, t_uniform_y_sample = uniform_margins_map.sample(n_samples=n_samples, seed=SEED)
    student_unif_EMI = estimator.estimate(t_uniform_x_sample, t_uniform_y_sample.reshape(-1,1))/np.log(2)
    student_unif_EMI_noise = estimator.estimate(t_uniform_x_sample, (t_uniform_y_sample+noise).reshape(-1,1))/np.log(2)

    # Dictionary to store MI estimates
    data_cases = {
        "Base case":{
            "Half-cube": base_hc_EMI,
            "Asinh": base_asinh_EMI,
            "Uniform margins": base_unif_EMI},
        "Gaussian distribution":{
            "Original": gaussian_EMI,
            "Half-cube": gaussian_hc_EMI,
            "Asinh": gaussian_asinh_EMI,
            "Uniform margins": gaussian_unif_EMI},
        "t-Student distribution":{
            "Original": student_EMI,
            "Half-cube": student_hc_EMI,
            "Asinh": student_asinh_EMI,
            "Uniform margins": student_unif_EMI}
        }
    
    # Dictionary to store MI estimates with noise
    data_cases_noise = {
        "Base case":{
            "Half-cube": base_hc_EMI_noise,
            "Asinh": base_asinh_EMI_noise,
            "Uniform margins": base_unif_EMI_noise},
        "Gaussian distribution":{
            "Original": gaussian_EMI_noise,
            "Half-cube": gaussian_hc_EMI_noise,
            "Asinh": gaussian_asinh_EMI_noise,
            "Uniform margins": gaussian_unif_EMI_noise},
        "t-Student distribution":{
            "Original": student_EMI_noise,
            "Half-cube": student_hc_EMI_noise,
            "Asinh": student_asinh_EMI_noise,
            "Uniform margins": student_unif_EMI_noise}
        }
    
    
    # Flatten the dictionaries to create the names of bars and values
    labels = []
    clean_values = []
    noise_values = []

    for case, methods in data_cases.items():
        for method, value in methods.items():
            combined_key = f"{case} @ {method}"
            labels.append(combined_key)
            clean_values.append(value)
            noise_values.append(data_cases_noise[case][method])
    
    # Invert order of labels and values
    labels = labels[::-1]
    clean_values = clean_values[::-1]
    noise_values = noise_values[::-1]
    
    y = np.arange(len(labels)) # Location of the labels
    width = 0.3 # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))  # Crear un gráfico con tamaño ajustado
    bars_clean = ax.barh(y - width/2, clean_values, width, label='Clean')
    bars_noise = ax.barh(y + width/2, noise_values, width, label='Noise')
    # Add the horizontal line at ground truth
    ax.axvline(x=ground_truth, color='gray', linestyle='--', linewidth=1, label=f'Ground truth: {ground_truth}')

    # ax.set_ylabel('Transforms')
    ax.set_xlabel('Mutual Information')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)  
    ax.legend()
    
    # Add a title if the 'title' argument is True 
    if title:
        ax.set_title('Kraskov–Stögbauer–Grassberger (KSG) estimation of joint mutual information')

    def add_values_to_bars(bars):
        """Function to add values to bars"""
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval, bar.get_y() + bar.get_height()/2, round(xval, 2), 
                    ha='left', va='center')

    # Add the values to each set of bars
    add_values_to_bars(bars_clean)
    add_values_to_bars(bars_noise)
    
    # Save the figure as a PDF if savefig is set to True.
    if savefig:
        fig.savefig(fname="compare_joint_EMI_transforms.pdf")

    plt.tight_layout()
    plt.show()