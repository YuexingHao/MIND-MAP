import pandas as pd
import numpy as np
import math

# Function to calculate p-value from chi-square statistic
def chi2_to_pvalue(chi2, df=1):
    """
    Convert chi-square statistic to p-value
    Using a simple approximation for chi-square with df=1
    """
    # Simple approximation for chi-square with df=1
    z = math.sqrt(chi2)
    # Approximation of the standard normal CDF
    p = math.erfc(z / math.sqrt(2)) 
    return p

# Function to calculate standard deviation for binary data
def binary_std(p, n):
    """
    Calculate standard deviation for a binary variable
    p: probability of success (accuracy)
    n: number of samples
    """
    return math.sqrt(p * (1-p) / n)

# Load the CSV file
df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Llama-70B/llama_70b_both_rounds.csv")

# Display basic info
print(f"Total rows in dataset: {len(df)}")
print("First 5 rows:")
print(df.head(5))

# Calculate all four conditions for the entire dataset
condition1 = (df['Round 1'] == df['Extracted_Answer']) & (df['Round 2'] == df['Extracted_Answer'])
count1 = condition1.sum()
print(f"\nOVERALL STATISTICS:")
print(f"Condition 1: Round 1 = Extracted_Answer AND Round 2 = Extracted_Answer: {count1} rows")

condition2 = (df['Round 1'] == df['Extracted_Answer']) & (df['Round 2'] != df['Extracted_Answer'])
count2 = condition2.sum()
print(f"Condition 2: Round 1 = Extracted_Answer AND Round 2 ≠ Extracted_Answer: {count2} rows")

condition3 = (df['Round 1'] != df['Extracted_Answer']) & (df['Round 2'] == df['Extracted_Answer'])
count3 = condition3.sum()
print(f"Condition 3: Round 1 ≠ Extracted_Answer AND Round 2 = Extracted_Answer: {count3} rows")

condition4 = (df['Round 1'] != df['Extracted_Answer']) & (df['Round 2'] != df['Extracted_Answer'])
count4 = condition4.sum()
print(f"Condition 4: Round 1 ≠ Extracted_Answer AND Round 2 ≠ Extracted_Answer: {count4} rows")

# Show percentage of each condition
total = len(df)
print(f"\nPercentages:")
print(f"Condition 1: {count1/total*100:.2f}%")
print(f"Condition 2: {count2/total*100:.2f}%")
print(f"Condition 3: {count3/total*100:.2f}%")
print(f"Condition 4: {count4/total*100:.2f}%")

# Statistical significance testing
print("\n\nSTATISTICAL SIGNIFICANCE TESTING:")

# Create contingency table for test
contingency_table = np.array([[count1, count2], 
                              [count3, count4]])
print("Contingency table [1: correct, 0: incorrect]:")
print("          | Round 2 = 1 | Round 2 = 0 |")
print("----------|------------|------------|")
print(f"Round 1 = 1 | {count1:10d} | {count2:10d} |")
print(f"Round 1 = 0 | {count3:10d} | {count4:10d} |")

# Calculate the difference in accuracy
round1_correct = count1 + count2
round2_correct = count1 + count3
round1_accuracy = round1_correct/total
round2_accuracy = round2_correct/total

# Calculate standard deviations
round1_std = binary_std(round1_accuracy, total)
round2_std = binary_std(round2_accuracy, total)

print(f"\nRound 1 accuracy: {round1_correct}/{total} = {round1_accuracy*100:.2f}% ± {round1_std*100:.2f}%")
print(f"Round 2 accuracy: {round2_correct}/{total} = {round2_accuracy*100:.2f}% ± {round2_std*100:.2f}%")
print(f"Difference: {abs(round2_correct - round1_correct)}/{total} = {abs(round2_accuracy - round1_accuracy)*100:.2f}%")

# Manual implementation of McNemar's test
# The test statistic is (b - c)^2 / (b + c) where b and c are the off-diagonal elements
b = count2  # Round 1 correct, Round 2 incorrect
c = count3  # Round 1 incorrect, Round 2 correct

# Calculate chi-square value with continuity correction
chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
p_value = chi2_to_pvalue(chi2)

print(f"\nManual McNemar's test (comparing Round 1 and Round 2):")
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"p-value: {p_value:.8f}")

if p_value < 0.05:
    print("Result: There is a statistically significant difference between Round 1 and Round 2 accuracy (p < 0.05)")
else:
    print("Result: There is no statistically significant difference between Round 1 and Round 2 accuracy (p ≥ 0.05)")

# Summary table for data sources
print("\n\nACCURACY SUMMARY BY DATA SOURCE:")
print("----------------------------------------------------------------------")
print(f"{'Data Source':<15} | {'Round 1 Accuracy':<25} | {'Round 2 Accuracy':<25}")
print("----------------------------------------------------------------------")

# Analysis by data_source
print("\n\nDETAILED ANALYSIS BY DATA SOURCE:")
data_sources = df['data_source'].unique()

for source in data_sources:
    source_df = df[df['data_source'] == source]
    source_total = len(source_df)
    
    # Calculate conditions for this source
    s_condition1 = (source_df['Round 1'] == source_df['Extracted_Answer']) & (source_df['Round 2'] == source_df['Extracted_Answer'])
    s_count1 = s_condition1.sum()
    
    s_condition2 = (source_df['Round 1'] == source_df['Extracted_Answer']) & (source_df['Round 2'] != source_df['Extracted_Answer'])
    s_count2 = s_condition2.sum()
    
    s_condition3 = (source_df['Round 1'] != source_df['Extracted_Answer']) & (source_df['Round 2'] == source_df['Extracted_Answer'])
    s_count3 = s_condition3.sum()
    
    s_condition4 = (source_df['Round 1'] != source_df['Extracted_Answer']) & (source_df['Round 2'] != source_df['Extracted_Answer'])
    s_count4 = s_condition4.sum()
    
    # Calculate accuracy for this source
    s_round1_correct = s_count1 + s_count2
    s_round2_correct = s_count1 + s_count3
    
    s_round1_accuracy = s_round1_correct/source_total
    s_round2_accuracy = s_round2_correct/source_total
    
    # Calculate standard deviation
    s_round1_std = binary_std(s_round1_accuracy, source_total)
    s_round2_std = binary_std(s_round2_accuracy, source_total)
    
    # Add to summary table
    print(f"{source:<15} | {s_round1_accuracy*100:.2f}% ± {s_round1_std*100:.2f}% ({s_round1_correct}/{source_total}) | {s_round2_accuracy*100:.2f}% ± {s_round2_std*100:.2f}% ({s_round2_correct}/{source_total})")
    
    # Print detailed analysis
    print(f"\nData Source: {source} (Total rows: {source_total})")
    print(f"Condition 1: {s_count1} rows ({s_count1/source_total*100:.2f}%)")
    print(f"Condition 2: {s_count2} rows ({s_count2/source_total*100:.2f}%)")
    print(f"Condition 3: {s_count3} rows ({s_count3/source_total*100:.2f}%)")
    print(f"Condition 4: {s_count4} rows ({s_count4/source_total*100:.2f}%)")
    
    print(f"Round 1 accuracy: {s_round1_correct}/{source_total} = {s_round1_accuracy*100:.2f}% ± {s_round1_std*100:.2f}%")
    print(f"Round 2 accuracy: {s_round2_correct}/{source_total} = {s_round2_accuracy*100:.2f}% ± {s_round2_std*100:.2f}%")
    
    # Perform statistical test
    if (s_count2 + s_count3) > 0:  # Only if there are disagreements
        # Calculate chi-square value with continuity correction
        s_chi2 = (abs(s_count2 - s_count3) - 1)**2 / (s_count2 + s_count3)
        s_p_value = chi2_to_pvalue(s_chi2)
        
        print(f"Chi-squared statistic: {s_chi2:.4f}")
        print(f"p-value: {s_p_value:.8f}")
        
        if s_p_value < 0.05:
            print("Statistically significant difference (p < 0.05)")
        else:
            print("No statistically significant difference (p ≥ 0.05)")
    else:
        print("Insufficient disagreements for statistical testing")

print("\n----------------------------------------------------------------------")

# Save the dataframe to a CSV file
output_path = "/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen/combinedQwen_72B.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved combined dataset to '{output_path}'")
