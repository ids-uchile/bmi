import numpy as np
import pandas as pd
import bmi
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression


def compare_EMI(X: np.ndarray,
                y: np.ndarray,
                ground_truth: np.ndarray,
                add_noise: bool = False,
                title: bool = False,
                savefig: bool = False
                ) -> None:
    """
    Compare several estimators of mutual information (EMI) across multiple dimensions of the data.
    The estimators include Histogram, Canonical Correlation Analysis (CCA), Kraskov, Stögbauer, and Grassberger (KSG), 
    and Mutual Information Neural Estimator (MINEE). The function also computes the EMI with added noise to the target.

    Args:
        X (np.ndarray): Input data with multiple dimensions.
        y (np.ndarray): Target data.
        ground_truth (np.ndarray): Ground truth values of mutual information for comparison.
        add_noise (bool, optional): If you want to add noise to the target value True if not False. Default is False.
        title (bool, optional): If you want to add title True if not False. Default is False.
        savefig (bool, optional): If you want to save the figure True otherwise False. Default is False.
    """
    
    # Generate uniform random noise to be added to the target variable
    noise = np.random.uniform(low=-0.1, high=0.1, size=y.shape[0])
    # Get the number of dimensions (features) in the input data
    n_dims = X.shape[1]
    
    # Store all estimations on a dictionary
    EMIS = {'Histogram': [],
            'CCA': [],
            'KSG': [],
            'MINE': []}
    
    # Iterate over each dimension in X
    if not add_noise: 
        for i in range(n_dims):
            
            # Histogram estimator
            estimator = bmi.estimators.HistogramEstimator(n_bins_x=200, n_bins_y=200)
            EMI_HIST = estimator.estimate(X[:,i].reshape(-1,1), y.reshape(-1,1))/np.log(2)
            
            # Canonical correlation analysis estimator
            estimator = bmi.estimators.CCAMutualInformationEstimator()
            EMI_CCA = estimator.estimate(X[:,i].reshape(-1,1), y.reshape(-1,1))/np.log(2)
            
            # Kraskov, Stögbauer and Grassberger estimator
            EMI_KSG = mutual_info_regression(X[:,i].reshape(-1,1), y).item()/np.log(2)
            
            # Mutual information neural estimator
            estimator = bmi.estimators.MINEEstimator()
            EMI_MINE = estimator.estimate(X[:,i].reshape(-1,1), y.reshape(-1,1))/np.log(2)
            
            # Append the computed EMI values to their corresponding lists
            EMIS['Histogram'].append(EMI_HIST)
            EMIS['CCA'].append(EMI_CCA)
            EMIS['KSG'].append(EMI_KSG)
            EMIS['MINE'].append(EMI_MINE)

    else:
        for i in range(n_dims):
            
            # Histogram estimator
            estimator = bmi.estimators.HistogramEstimator(n_bins_x=200, n_bins_y=200)
            EMI_HIST_noise = estimator.estimate(X[:,i].reshape(-1,1), (y+noise).reshape(-1,1))/np.log(2)
            
            # Canonical correlation analysis estimator
            estimator = bmi.estimators.CCAMutualInformationEstimator()
            EMI_CCA_noise = estimator.estimate(X[:,i].reshape(-1,1), (y+noise).reshape(-1,1))/np.log(2)
            
            # Kraskov, Stögbauer and Grassberger estimator
            EMI_KSG_noise = mutual_info_regression(X[:,i].reshape(-1,1), (y+noise)).item()/np.log(2)
            
            # Mutual information neural estimator
            estimator = bmi.estimators.MINEEstimator()
            EMI_MINE_noise = estimator.estimate(X[:,i].reshape(-1,1), (y+noise).reshape(-1,1))/np.log(2)
            
            # Append the computed EMI values to their corresponding lists
            EMIS['Histogram'].append(EMI_HIST_noise)
            EMIS['CCA'].append(EMI_CCA_noise)
            EMIS['KSG'].append(EMI_KSG_noise)
            EMIS['MINE'].append(EMI_MINE_noise)
            
    # Iterates over the estimations dictionary and flattens the data into a list
    data_flat = []
    for key, values in EMIS.items():
        data_flat.append([key]+values)
    
    # Create DataFrame
    df = pd.DataFrame(data_flat, columns=['Estimator', 'Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4', 'Dimension 5'])
    # Add the ground truth as the first row in the DataFrame
    ground_truth_df = pd.DataFrame([["Ground truth", ground_truth[0], ground_truth[1], ground_truth[2], ground_truth[3], ground_truth[4]]], 
                               columns=df.columns)
    df = pd.concat([ground_truth_df, df], ignore_index=True)
    df_values = df.set_index('Estimator')
    
    fig, axes = plt.subplots(1, len(df_values.columns), figsize=(18, 6), sharey=True)
    for i, column in enumerate(df_values.columns):
        diffs = df_values[[column]].copy()
        diffs = (diffs - diffs.iloc[0, 0]) # Subtract the ground truth value from the estimated MI values
        # Plot the heatmap for each dimension
        sns.heatmap(diffs, cmap="BuPu",annot=df_values[[column]].values.reshape(-1, 1), ax=axes[i], cbar=True, linewidths=0.5, fmt='.6f', center=0)
        axes[i].set_ylabel('')
    
    # Add a title if the 'title' argument is True 
    if title:
        if not add_noise:
            fig.suptitle("Comparision of estimators of mutual information from clean data", fontsize=16)
        else:
            fig.suptitle("Comparision of estimators of mutual information from data with noise", fontsize=16)
    
    # Save the figure as a PDF if savefig is set to True.
    if savefig:
        fig.savefig(fname="compare_EMI.pdf")
        
    plt.tight_layout()
    plt.show()


def compare_joint_EMI(X: np.ndarray,
                      y: np.ndarray,
                      ground_truth: np.ndarray,
                      title: bool = False,
                      savefig: bool = False
                      ) -> None:
    """
    Compare several estimators of the joint mutual information.
    The estimators include Histogram, Canonical Correlation Analysis (CCA), Kraskov, Stögbauer, and Grassberger (KSG), 
    and Mutual Information Neural Estimator (MINEE). The function also computes the EMI with added noise to the target.

    Args:
        X (np.ndarray): Input data with multiple dimensions.
        y (np.ndarray): Target data.
        ground_truth (np.ndarray): Ground truth values of mutual information for comparison.
        title (bool, optional): If you want to add title True if not False. Default is False.
        savefig (bool, optional): If you want to save the figure True otherwise False. Default is False.
    """
    
    # Generate uniform random noise to be added to the target variable
    noise = np.random.uniform(low=-0.1, high=0.1, size=y.shape[0])
    
    # Histogram estimator
    estimator = bmi.estimators.HistogramEstimator(n_bins_x=30, n_bins_y=30)
    EMI_HIST = estimator.estimate(X, y.reshape(-1,1))/np.log(2)
    EMI_HIST_noise = estimator.estimate(X, (y+noise).reshape(-1,1))/np.log(2)
    
    # Canonical correlation analysis estimator
    estimator = bmi.estimators.CCAMutualInformationEstimator()
    EMI_CCA = estimator.estimate(X, y.reshape(-1,1))/np.log(2)
    EMI_CCA_noise = estimator.estimate(X, (y+noise).reshape(-1,1))/np.log(2)

    # Kraskov, Stögbauer and Grassberger estimator
    estimator = bmi.estimators.KSGEnsembleFirstEstimator()
    EMI_KSG = estimator.estimate(X, y.reshape(-1,1))/np.log(2)
    EMI_KSG_noise = estimator.estimate(X, (y+noise).reshape(-1,1))/np.log(2)
    
    # Mutual information neural estimator
    estimator = bmi.estimators.MINEEstimator()
    EMI_MINEE = estimator.estimate(X, y.reshape(-1,1))/np.log(2)
    EMI_MINEE_noise = estimator.estimate(X, (y+noise).reshape(-1,1))/np.log(2)
    
    # Store all estimations on a dictionary
    EMIS = {'clean': {'Histogram': EMI_HIST,
                        'CCA': EMI_CCA,
                        'KSG': EMI_KSG,
                        'MINE': EMI_MINEE},
              'noise': {'Histogram': EMI_HIST_noise,
                        'CCA': EMI_CCA_noise,
                        'KSG': EMI_KSG_noise,
                        'MINE': EMI_MINEE_noise}}
    
    # Create the DataFrame
    df = pd.DataFrame(EMIS)
    
    # Create the grouped bar chart
    labels = df.index  # 'Histogram', 'CCA', 'KSG', 'MINEE'
    clean_values = df['clean'].values  # Values without noise
    noise_values = df['noise'].values  # Values with noise

    x = np.arange(len(labels))  # Location of the labels
    width = 0.3  # Width of the bars

    fig, ax = plt.subplots()
    bars_clean = ax.bar(x - width/2, clean_values, width, label='Clean')
    bars_noise = ax.bar(x + width/2, noise_values, width, label='Noise')
    # Add the horizontal line at ground truth
    ax.axhline(y=ground_truth, color='gray', linestyle='--', linewidth=1, label=f'Ground truth: {ground_truth}')

    ax.set_xlabel('Estimator')
    ax.set_ylabel('Mutual Information')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add a title if the 'title' argument is True 
    if title:
        ax.set_title("Comparision of estimator of joint mutual information")

    def add_values_to_bars(bars):
        """Function to add values to bars"""
        for bar in bars:
            yval = bar.get_height() # Gets the height of the bar (value)
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                    ha='center', va='bottom')  # Add the value above the bar

    # Add the values to each set of bars
    add_values_to_bars(bars_clean)
    add_values_to_bars(bars_noise)
    
    # Save the figure as a PDF if savefig is set to True.
    if savefig:
        fig.savefig(fname="compare_joint_EMI.pdf")
    
    plt.tight_layout()
    plt.show()