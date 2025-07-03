import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

 # Function to compute and display statistics
def show_statistics(ax, data, metric_name):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    min_val = np.min(data)
    max_val = np.max(data)

    # Annotate statistics on the plot
    stats_text = (
        f'Mean: {mean:.2f}\n'
        f'Median: {median:.2f}\n'
        f'Std Dev: {std_dev:.2f}\n'
        f'25th Percentile: {q1:.2f}\n'
        f'75th Percentile: {q3:.2f}\n'
        f'Min: {min_val:.2f}\n'
        f'Max: {max_val:.2f}'
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))

def plot_metrics_distribution(df, save_path="plots", plot_name:str="metrics_distribution_with_stats"):
    """
    Function to create and save the plot for the distribution of the metrics Lift, Confidence, and Support.
    df: DataFrame containing the columns 'Support', 'Confidence' and 'Lift'.
    save_path: Directory to save the image.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set a more stylish theme for seaborn
    sns.set_palette("Set2")  # Use a nice palette of colors
    sns.set(style="whitegrid", font_scale=1.2)  # Increase font scale for better readability
    
    # Create the figure
    plt.figure(figsize=(12, 10))  # Increased height for more space between plots
    
    
    # Distribution of Support
    ax1 = plt.subplot(3, 1, 1)
    sns.histplot(df['Support'], kde=True, color='blue', bins=50, ax=ax1)
    ax1.set_title('Distribution of Support')
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Density')
    show_statistics(ax1, df['Support'], 'Support')
    
    # Distribution of Confidence
    ax2 = plt.subplot(3, 1, 2)
    sns.histplot(df['Confidence'], kde=True, color='green', bins=50, ax=ax2)
    ax2.set_title('Distribution of Confidence')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Density')
    show_statistics(ax2, df['Confidence'], 'Confidence')
    
    # Distribution of Lift
    ax3 = plt.subplot(3, 1, 3)
    sns.histplot(df['Lift'], kde=True, color='red', bins=50, ax=ax3)
    ax3.set_title('Distribution of Lift')
    ax3.set_xlabel('Lift')
    ax3.set_ylabel('Density')
    show_statistics(ax3, df['Lift'], 'Lift')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{save_path}/{plot_name}.jpg", dpi=300)

    # Close the figure to avoid overlap on future plots
    plt.close()
