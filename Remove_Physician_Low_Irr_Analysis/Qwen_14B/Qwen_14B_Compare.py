import pandas as pd
import numpy as np
from scipy import stats

# Function to calculate binary standard deviation
def binary_std(p, n):
    """
    Calculate standard deviation for a binary variable
    p: probability of success (accuracy)
    n: number of samples
    """
    if n == 0 or p == 0:
        return 0
    return (p * (1-p) / n) ** 0.5

# Load the dataframes
print("Loading dataframes...")
df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen_14B/Qwen_14B_predictions_final.csv")  # Recent results
df2 = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen_14B/14B_Round_1_Result.csv")  # Previous results with Round 1

# Display original column information
print(f"Original df columns: {df.columns.tolist()}")
print(f"Original df2 columns: {df2.columns.tolist()}")
print(f"Original df rows: {len(df)}")
print(f"Original df2 rows: {len(df2)}")

# Sample of QA_ID formats from both dataframes
print("\nSample QA_IDs from df:", df['QA_ID'].head(3).tolist())
print("Sample QA_IDs from df2:", df2['QA_ID'].head(3).tolist())

# Create standardized QA_ID columns by removing underscores and spaces
df['QA_ID_std'] = df['QA_ID'].str.replace('_', '').str.replace(' ', '')
df2['QA_ID_std'] = df2['QA_ID'].str.replace('_', '').str.replace(' ', '')

# Create a mapping from df2's standardized QA_ID to the columns we want
round1_mapping = dict(zip(df2['QA_ID_std'], df2['Round 1']))
ground_truth_mapping = dict(zip(df2['QA_ID_std'], df2['Ground_Truth_Answer']))

# Add the columns to df using the standardized QA_ID for lookup
df['Round 2'] = df['QA_ID_std'].map(round1_mapping)
df['Ground_Truth_Answer'] = df['QA_ID_std'].map(ground_truth_mapping)

# Rename the Extracted_Answer column to Round 1
if 'Extracted_Answer' in df.columns:
    df.rename(columns={'Extracted_Answer': 'Round 1'}, inplace=True)

# Count how many matches were found
round2_matched = df['Round 2'].notna().sum()
ground_truth_matched = df['Ground_Truth_Answer'].notna().sum()

print(f"Matched {round2_matched}/{len(df)} records for Round 2 ({round2_matched/len(df)*100:.2f}%)")
print(f"Matched {ground_truth_matched}/{len(df)} records for Ground_Truth_Answer ({ground_truth_matched/len(df)*100:.2f}%)")

# Display a sample of the merged data
print("\nSample of merged data:")
display_columns = ['QA_ID', 'Round 1', 'Round 2', 'Ground_Truth_Answer', 'data_source']
display_columns = [col for col in display_columns if col in df.columns]
print(df[display_columns].head(10))

# Function to perform accuracy analysis and print results
def analyze_accuracy(data, title=""):
    # Calculate accuracy for both rounds where we have Ground_Truth_Answer
    if 'Ground_Truth_Answer' in data.columns and 'Round 1' in data.columns and 'Round 2' in data.columns:
        valid_rows = data.dropna(subset=['Ground_Truth_Answer', 'Round 1', 'Round 2'])
        
        # If no valid rows, return early
        if len(valid_rows) == 0:
            print(f"\n{title} - No valid rows for analysis")
            return
        
        # Calculate Round 1 accuracy
        round1_correct = (valid_rows['Round 1'] == valid_rows['Ground_Truth_Answer']).sum()
        round1_accuracy = round1_correct / len(valid_rows) if len(valid_rows) > 0 else 0
        round1_std = binary_std(round1_accuracy, len(valid_rows))
        
        # Calculate Round 2 accuracy
        round2_correct = (valid_rows['Round 2'] == valid_rows['Ground_Truth_Answer']).sum()
        round2_accuracy = round2_correct / len(valid_rows) if len(valid_rows) > 0 else 0
        round2_std = binary_std(round2_accuracy, len(valid_rows))
        
        print(f"\n{title}ACCURACY METRICS (on {len(valid_rows)} valid rows):")
        print(f"Round 1 accuracy: {round1_correct}/{len(valid_rows)} = {round1_accuracy*100:.2f}% ± {round1_std*100:.2f}%")
        print(f"Round 2 accuracy: {round2_correct}/{len(valid_rows)} = {round2_accuracy*100:.2f}% ± {round2_std*100:.2f}%")
        print(f"Difference (Round 1 - Round 2): {(round1_accuracy - round2_accuracy)*100:+.2f}%")
        
        # Calculate agreement between rounds
        agreement = (valid_rows['Round 1'] == valid_rows['Round 2']).sum()
        agreement_rate = agreement / len(valid_rows) if len(valid_rows) > 0 else 0
        print(f"Agreement between Round 1 and Round 2: {agreement}/{len(valid_rows)} = {agreement_rate*100:.2f}%")

        # Show detailed contingency table
        print("\nContingency table:")
        print("               | Round 1 correct | Round 1 incorrect |")
        print("---------------|----------------|-------------------|")
        
        # Both correct
        both_correct = ((valid_rows['Round 2'] == valid_rows['Ground_Truth_Answer']) &
                        (valid_rows['Round 1'] == valid_rows['Ground_Truth_Answer'])).sum()
        
        # Round 2 correct, Round 1 incorrect
        r2_correct_r1_incorrect = ((valid_rows['Round 2'] == valid_rows['Ground_Truth_Answer']) &
                                  (valid_rows['Round 1'] != valid_rows['Ground_Truth_Answer'])).sum()
        
        # Round 2 incorrect, Round 1 correct
        r2_incorrect_r1_correct = ((valid_rows['Round 2'] != valid_rows['Ground_Truth_Answer']) &
                                  (valid_rows['Round 1'] == valid_rows['Ground_Truth_Answer'])).sum()
        
        # Both incorrect
        both_incorrect = ((valid_rows['Round 2'] != valid_rows['Ground_Truth_Answer']) &
                          (valid_rows['Round 1'] != valid_rows['Ground_Truth_Answer'])).sum()
        
        print(f"Round 2 correct   | {both_correct:14d} | {r2_correct_r1_incorrect:17d} |")
        print(f"Round 2 incorrect | {r2_incorrect_r1_correct:14d} | {both_incorrect:17d} |")
        
        # Print percentages
        total = len(valid_rows)
        print(f"\nPercentages (of {total} total):")
        print(f"Both correct: {both_correct}/{total} = {both_correct/total*100:.2f}%")
        print(f"Round 2 correct, Round 1 incorrect: {r2_correct_r1_incorrect}/{total} = {r2_correct_r1_incorrect/total*100:.2f}%")
        print(f"Round 2 incorrect, Round 1 correct: {r2_incorrect_r1_correct}/{total} = {r2_incorrect_r1_correct/total*100:.2f}%")
        print(f"Both incorrect: {both_incorrect}/{total} = {both_incorrect/total*100:.2f}%")
        
        # Perform McNemar's test if there are disagreements
        if r2_correct_r1_incorrect + r2_incorrect_r1_correct > 0:
            # Calculate chi-square with continuity correction
            chi2 = (abs(r2_correct_r1_incorrect - r2_incorrect_r1_correct) - 1)**2 / (r2_correct_r1_incorrect + r2_incorrect_r1_correct)
            p_value = 1 - stats.chi2.cdf(chi2, 1)  # 1 degree of freedom
            
            print(f"\nMcNemar's test p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("Statistically significant difference (p < 0.05)")
            else:
                print("No statistically significant difference (p ≥ 0.05)")

# Perform overall analysis first
print("\n" + "="*80)
print("OVERALL ANALYSIS")
print("="*80)
analyze_accuracy(df)

# Check if data_source column exists for per-source analysis
if 'data_source' in df.columns:
    print("\n" + "="*80)
    print("ANALYSIS BY DATA SOURCE")
    print("="*80)
    
    # Get unique data sources and sort them
    data_sources = df['data_source'].dropna().unique()
    data_sources.sort()
    
    # Create a summary table for a quick overview
    print("\nSUMMARY TABLE BY DATA SOURCE")
    print(f"{'Data Source':<15} | {'Sample Size':<11} | {'Round 1 Accuracy':<25} | {'Round 2 Accuracy':<25} | {'Difference':<10}")
    print("-" * 95)
    
    # Store detailed results for the summary table
    summary_rows = []
    
    # Analyze each data source separately
    for source in data_sources:
        source_df = df[df['data_source'] == source]
        valid_rows = source_df.dropna(subset=['Ground_Truth_Answer', 'Round 1', 'Round 2'])
        
        if len(valid_rows) > 0:
            # Calculate metrics for summary
            round1_correct = (valid_rows['Round 1'] == valid_rows['Ground_Truth_Answer']).sum()
            round1_accuracy = round1_correct / len(valid_rows)
            round1_std = binary_std(round1_accuracy, len(valid_rows))
            
            round2_correct = (valid_rows['Round 2'] == valid_rows['Ground_Truth_Answer']).sum()
            round2_accuracy = round2_correct / len(valid_rows)
            round2_std = binary_std(round2_accuracy, len(valid_rows))
            
            diff = round1_accuracy - round2_accuracy
            
            # Add to summary table
            print(f"{source:<15} | {len(valid_rows):<11d} | {round1_accuracy*100:.2f}% ± {round1_std*100:.2f}% | {round2_accuracy*100:.2f}% ± {round2_std*100:.2f}% | {diff*100:+.2f}%")
            
            # Add row for DataFrame
            summary_rows.append({
                'data_source': source,
                'sample_size': len(valid_rows),
                'round1_accuracy': round1_accuracy,
                'round1_std': round1_std,
                'round2_accuracy': round2_accuracy,
                'round2_std': round2_std,
                'difference': diff
            })
        
        # Perform detailed analysis for each source
        print("\n" + "-"*80)
        analyze_accuracy(source_df, f"DATA SOURCE: {source} - ")
    
    # Create a DataFrame from summary data and save it separately
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = "/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen_14B/Qwen_14B_accuracy_by_source.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved accuracy summary by data source to {summary_file}")
else:
    print("\nNo 'data_source' column found in the data, skipping per-source analysis.")

# Remove the temporary standardized ID column before saving
df.drop(columns=['QA_ID_std'], inplace=True)

# Save the merged results
output_file = "/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen_14B/Qwen_14B_comparison_results.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved merged results to {output_file}")

