import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for JAMA Challenge
models = ['Physician\nTrainee', 'GPT\n4o', 'Llama\n70B', 'Qwen\n72B', 'Qwen\n14B']
# Original accuracy values
original_acc = [56.0, 80.0, 50.0, 70.0, 54.0]  
# After some processing accuracy values - lighter shade
processed_acc = [np.nan, 70.0, 68.7, 60.0, np.nan]  
# Difference between processed and original
differences = [5.9, 10.2, 18.7, 10.0, 1.4]  
# Models with statistical significance
significant = [False, True, False, True, False]  

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Set the positions for each model on x-axis
x_pos = np.arange(len(models))

# Plot original accuracy points (darker blue and orange)
for i, (acc, sig) in enumerate(zip(original_acc, significant)):
    if i == 0:  # Physician Trainee in orange
        ax.scatter(i, acc, s=200, color='#ff7f0e', edgecolors='black', linewidth=1, zorder=3)
    else:  # Others in dark blue
        ax.scatter(i, acc, s=200, color='#1f77b4', edgecolors='black', linewidth=1, zorder=3)
    
    # Add difference annotation
    if i > 0 or i == 0:  # For all points
        ax.annotate(f'+{differences[i]}', 
                   xy=(i, acc), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   fontsize=9)
    
    # Add significance markers
    if sig:
        ax.text(i, 95, '*', color='red', fontsize=24, ha='center', va='center')

# Plot processed accuracy points (lighter blue)
for i, acc in enumerate(processed_acc):
    if not np.isnan(acc):
        ax.scatter(i, acc, s=200, color='#aec7e8', edgecolors='black', linewidth=1, zorder=2)

# Customize the plot
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('(a) JAMA Challenge', fontsize=18)

# Add percentage labels to y-axis
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
output_filename = 'jama_challenge_dotplot.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Dot plot saved as {output_filename}") 