import openai

import pandas as pd
import os
import re
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm

# Change directory to the location of the file
os.chdir("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/RemoveLLM_Low+Irr")


# llama70b = pd.read_csv("70B_ContextCite_Removal.csv")

# print(llama70b.head())
# print(len(llama70b))

# def generate_direct_prediction(context, question):
#     """
#     Queries GPT-4o with a clinical vignette (context) and a multiple-choice question (with embedded options).
#     Returns only the predicted answer in the format: '[Letter]: [Answer Text]' (e.g., 'B: Femoral artery murmur').
#     """
#     prompt = f"""
# You are given some context and a multiple-choice question.

# Select the most appropriate answer from the options provided.

# {context}

# {question}

# Provide your response in the following format:\n<answer>Option [letter]</answer>"""

#     try:
#         client = openai.OpenAI(
#             api_key="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"
#         )

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature = 0
#         )

#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         print(f"Error: {e}")
#         return "Error"

# # Create a new column to store the GPT-4o predictions
# llama70b['GPT4o_prediction'] = None

# # Process each row in the DataFrame
# for idx, row in tqdm(llama70b.iterrows(), total=len(llama70b), desc="Processing with GPT-4o"):
#     context = row['70B_After_Removal']
#     question = row['question_options']
    
#     # Skip if any of the required fields are missing
#     if pd.isna(context) or pd.isna(question):
#         llama70b.loc[idx, 'GPT4o_prediction'] = "Missing data"
#         continue
    
#     # Get prediction from GPT-4o
#     prediction = generate_direct_prediction(context, question)
#     llama70b.loc[idx, 'GPT4o_prediction'] = prediction

# # Save the results to a CSV file
# output_file = "llama70b_GPT4o_predictions.csv"
# llama70b.to_csv(output_file, index=False)
# print(f"Results saved to {output_file}")


output_file = "llama70b_GPT4o_predictions.csv"

# Analysis of the predictions
import pandas as pd
import numpy as np
import re

# Load the predictions file
results_df = pd.read_csv(output_file)
print(f"Loaded {len(results_df)} rows from {output_file}")

# Function to extract letter from GPT4o prediction
def extract_answer_letter(prediction):
    if pd.isna(prediction) or prediction == "Error" or prediction == "Missing data":
        return None
    
    # Try to extract from <answer>Option [letter]</answer> format
    match = re.search(r'<answer>Option ([A-E])</answer>', prediction)
    if match:
        return match.group(1)
    
    # Alternative formats
    match = re.search(r'Option ([A-E])', prediction)
    if match:
        return match.group(1)
    
    # Just find any letter
    match = re.search(r'\b([A-E])\b', prediction)
    if match:
        return match.group(1)
    
    return None

# Extract answer letters
results_df['extracted_prediction'] = results_df['GPT4o_prediction'].apply(extract_answer_letter)

# Calculate accuracy
results_df['is_correct'] = results_df['extracted_prediction'] == results_df['answer_df3']

# Overall accuracy
valid_predictions = results_df['extracted_prediction'].notna()
overall_accuracy = results_df.loc[valid_predictions, 'is_correct'].mean() * 100
overall_std = results_df.loc[valid_predictions, 'is_correct'].std() * 100

print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
print(f"Standard Deviation: {overall_std:.2f}%")
print(f"Total valid predictions: {valid_predictions.sum()} ({valid_predictions.sum()/len(results_df)*100:.2f}%)")

# Analysis by data source
print("\nAccuracy by Data Source:")
data_sources = results_df['data_source_corr'].unique()

for source in data_sources:
    source_df = results_df[results_df['data_source_corr'] == source]
    valid_source_predictions = source_df['extracted_prediction'].notna()
    
    if valid_source_predictions.sum() > 0:
        source_accuracy = source_df.loc[valid_source_predictions, 'is_correct'].mean() * 100
        source_std = source_df.loc[valid_source_predictions, 'is_correct'].std() * 100
        source_count = valid_source_predictions.sum()
        
        print(f"{source}: {source_accuracy:.2f}% (±{source_std:.2f}%), n={source_count}")

# Save the analysis results
results_df.to_csv("GPT4o_analysis_results.csv", index=False)
print("\nDetailed analysis saved to GPT4o_analysis_results.csv")