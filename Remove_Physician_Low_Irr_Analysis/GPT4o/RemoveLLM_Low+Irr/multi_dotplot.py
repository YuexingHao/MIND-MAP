import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Models for all panels
models = ['Physician\nTrainee', 'GPT\n4o', 'Llama\n70B', 'Qwen\n72B', 'Qwen\n14B']

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

# Titles for each subplot
titles = ['(a) JAMA Challenge', '(b) MedBullets', '(c) MMLU', '(d) MedXpertQA']

# Sample data for each dataset (replace with your actual data)
data = {
    'JAMA': {
        'original_acc': [56.0, 80.0, 50.0, 70.0, 54.0],
        'processed_acc': [np.nan, 70.0, 68.7, 60.0, np.nan],
        'differences': [5.9, 10.2, 18.7, 10.0, 1.4],
        'significant': [False, True, False, True, False]
    },
    'MedBullets': {
        'original_acc': [62.0, 74.5, 55.0, 65.0, 50.0],
        'processed_acc': [np.nan, 65.0, 70.0, 55.0, np.nan],
        'differences': [4.5, 9.6, 15.0, 8.2, 0.8],
        'significant': [False, True, True, False, False]
    },
    'MMLU': {
        'original_acc': [58.0, 95.6, 60.0, 75.0, 56.0],
        'processed_acc': [np.nan, 85.0, 75.0, 65.0, np.nan],
        'differences': [6.2, 0.8, 15.0, 12.5, 2.1],
        'significant': [False, False, True, True, False]
    },
    'MedXpertQA': {
        'original_acc': [45.0, 65.0, 48.0, 55.0, 40.0],
        'processed_acc': [np.nan, 60.0, 65.0, 48.0, np.nan],
        'differences': [7.2, 24.8, 17.0, 14.6, 5.2],
        'significant': [False, True, True, False, False]
    }
}

# Dataset names in the order they should appear
datasets = ['JAMA', 'MedBullets', 'MMLU', 'MedXpertQA']

# Plot each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i]
    
    # Set the positions for each model on x-axis
    x_pos = np.arange(len(models))
    
    # Get data for this dataset
    original_acc = data[dataset]['original_acc']
    processed_acc = data[dataset]['processed_acc']
    differences = data[dataset]['differences']
    significant = data[dataset]['significant']
    
    # Plot original accuracy points (darker blue and orange)
    for j, (acc, sig) in enumerate(zip(original_acc, significant)):
        if j == 0:  # Physician Trainee in orange
            ax.scatter(j, acc, s=200, color='#ff7f0e', edgecolors='black', linewidth=1, zorder=3)
        else:  # Others in dark blue
            ax.scatter(j, acc, s=200, color='#1f77b4', edgecolors='black', linewidth=1, zorder=3)
        
        # Add difference annotation
        if j > 0 or j == 0:  # For all points
            ax.annotate(f'+{differences[j]}', 
                       xy=(j, acc), 
                       xytext=(5, 0), 
                       textcoords='offset points',
                       fontsize=9)
        
        # Add significance markers
        if sig:
            ax.text(j, 95, '*', color='red', fontsize=24, ha='center', va='center')
    
    # Plot processed accuracy points (lighter blue)
    for j, acc in enumerate(processed_acc):
        if not np.isnan(acc):
            ax.scatter(j, acc, s=200, color='#aec7e8', edgecolors='black', linewidth=1, zorder=2)
    
    # Customize the plot
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_title(titles[i], fontsize=16)
    
    # Add percentage labels to y-axis
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add y-axis label only for left subplots
    if i % 2 == 0:
        ax.set_ylabel('Accuracy', fontsize=14)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend for the first subplot
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
               markeredgecolor='black', markersize=12, label='Physician Trainee'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markeredgecolor='black', markersize=12, label='LLM Original'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aec7e8', 
               markeredgecolor='black', markersize=12, label='After Processing')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           ncol=3, fontsize=12)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
output_filename = 'multi_dataset_dotplot.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Multi-panel dot plot saved as {output_filename}") 