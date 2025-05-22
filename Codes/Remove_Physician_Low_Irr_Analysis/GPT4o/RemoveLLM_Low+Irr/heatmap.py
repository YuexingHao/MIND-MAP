import pandas as pd
import matplotlib.pyplot as plt

# Construct the DataFrame
data = {
    'MMLU': [95.6, 0.8, -1.8, -2.4, -9.6],
    'JAMA': [68.5, 10.2, 4.0, 0.7, -16.9],
    'MedBullets': [74.5, 9.6, 2.3, 0.1, -10.5],
    'MedXpertQA': [16.4, 24.8, 24.6, 22.4, -7.7]
}

index = [
    'Original',
    'After Human Low+Irr Removal',
    'After Qwen-72B Low+Irr Removal',
    'After Llama-70B Low+Irr Removal',
    'After GPT-4o SR Low+Irr Removal'
]

df = pd.DataFrame(data, index=index)

# Plot heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(df.values, aspect='auto', cmap='viridis')
plt.colorbar(im, label='Performance Metric')
plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
plt.yticks(range(len(df.index)), df.index)
plt.title('GPT-4o Performance Heatmap Across Conditions')
plt.tight_layout()

# Save the figure as a PNG file
output_filename = 'gpt4o_performance_heatmap.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Heatmap saved as {output_filename}")

# plt.show()  # Commented out to avoid displaying the plot
