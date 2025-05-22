import pandas as pd
import re
import numpy as np
from scipy import stats

# Function to extract just the letter from format like "<answer>Option D</answer>"
def extract_letter(text):
    if pd.isna(text):
        return None
    
    # Try to match "<answer>Option X</answer>" pattern
    pattern1 = r'<answer>Option\s+([A-Za-z])</answer>'
    match1 = re.search(pattern1, text)
    if match1:
        return match1.group(1)
    
    # Try to match "<answer>X</answer>" pattern
    pattern2 = r'<answer>([A-Za-z])</answer>'
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1)
    
    # Try to match for just the letter (fallback)
    pattern3 = r'Option\s+([A-Za-z])'
    match3 = re.search(pattern3, text)
    if match3:
        return match3.group(1)
    
    # Just return the text if no match is found
    return text

# Function to calculate standard deviation for binary data
def binary_std(p, n):
    """
    Calculate standard deviation for a binary variable
    p: probability of success (accuracy)
    n: number of samples
    """
    if n == 0 or p == 0:
        return 0
    return (p * (1-p) / n) ** 0.5

# Load the CSV file
df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_Round1.csv")
df2 = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_Round2.csv")
df3 = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/merged_2k_with_4k_ID.csv")

# Display the original columns of each dataframe
print("Original columns in df:", df.columns.tolist())
print("Original columns in df2:", df2.columns.tolist())
print("Original columns in df3:", df3.columns.tolist())
print("Original rows in df:", len(df))
print("Original rows in df2:", len(df2))
print("Original rows in df3:", len(df3))

# Create a mapping from df3's 'ID' to df3's '4k_ID'
id_to_4k_id_mapping = dict(zip(df3['ID'], df3['4k_ID']))

# Check if 'ID_corr' exists in df2
if 'ID_corr' in df2.columns:
    # Apply the mapping to add 4k_ID to df2
    df2['4k_ID'] = df2['ID_corr'].map(id_to_4k_id_mapping)
    
    # Show how many 4k_IDs were successfully mapped
    mapped_count = df2['4k_ID'].notna().sum()
    print(f"\nMapped {mapped_count} 4k_IDs to df2 based on matching ID_corr ({mapped_count/len(df2)*100:.2f}%)")
    
    # Show sample of the merged data
    print("\nSample of df2 with added 4k_ID column:")
    print(df2[['ID_corr', '4k_ID']].head(10))
    
    # Now merge df's gpt_direct_prediction to df2 based on matching 4k_ID with QA_ID
    if 'gpt_direct_prediction' in df.columns and 'QA_ID' in df.columns:
        # Rename df's gpt_direct_prediction to Round 1
        df['Round 1'] = df['gpt_direct_prediction']
        
        # Create mapping from df's QA_ID to Round 1
        qa_to_prediction_mapping = dict(zip(df['QA_ID'], df['Round 1']))
        
        # Apply mapping to df2 using the newly added 4k_ID column
        df2['Round 1'] = df2['4k_ID'].map(qa_to_prediction_mapping)
        
        # Rename df2's gpt_direct_prediction to Round 2
        if 'gpt_direct_prediction' in df2.columns:
            df2['Round 2'] = df2['gpt_direct_prediction']
        elif 'answer_corr' in df2.columns:  # Assuming answer_corr is the prediction column in df2
            df2['Round 2'] = df2['answer_corr']
        
        # Extract just the answer letter from Round 1 and Round 2
        df2['Round 1 Letter'] = df2['Round 1'].apply(extract_letter)
        df2['Round 2 Letter'] = df2['Round 2'].apply(extract_letter)
        
        # Extract the letter from the correct answer as well
        if 'answer_corr' in df2.columns:
            df2['Correct Letter'] = df2['answer_corr'].apply(extract_letter)
        
        # Show examples of the extraction
        print("\nExamples of answer letter extraction:")
        print(pd.DataFrame({
            'Round 1': df2['Round 1'].head(5),
            'Round 1 Letter': df2['Round 1 Letter'].head(5),
            'Round 2': df2['Round 2'].head(5), 
            'Round 2 Letter': df2['Round 2 Letter'].head(5),
            'Correct Letter': df2['Correct Letter'].head(5) if 'Correct Letter' in df2.columns else None
        }))
        
        # Show how many predictions were successfully mapped
        pred_mapped_count = df2['Round 1'].notna().sum()
        print(f"\nMapped {pred_mapped_count} Round 1 predictions to df2 based on 4k_ID matching QA_ID ({pred_mapped_count/len(df2)*100:.2f}%)")
        
        # Show sample of the data with both added columns
        print("\nSample of df2 with both rounds of predictions:")
        print(df2[['ID_corr', '4k_ID', 'Round 1 Letter', 'Round 2 Letter', 'Correct Letter' if 'Correct Letter' in df2.columns else 'answer_corr']].head(10))
        
        # Calculate accuracy
        if 'Correct Letter' in df2.columns:
            # Filter out rows where we don't have predictions or correct answers
            valid_rows = df2.dropna(subset=['Round 1 Letter', 'Round 2 Letter', 'Correct Letter'])
            total_valid = len(valid_rows)
            
            # Calculate accuracy for Round 1
            round1_correct = (valid_rows['Round 1 Letter'] == valid_rows['Correct Letter']).sum()
            round1_accuracy = round1_correct / total_valid if total_valid > 0 else 0
            round1_std = binary_std(round1_accuracy, total_valid)
            
            # Calculate accuracy for Round 2
            round2_correct = (valid_rows['Round 2 Letter'] == valid_rows['Correct Letter']).sum()
            round2_accuracy = round2_correct / total_valid if total_valid > 0 else 0
            round2_std = binary_std(round2_accuracy, total_valid)
            
            print("\n" + "="*50)
            print("ACCURACY METRICS")
            print("="*50)
            print(f"Total valid rows with predictions and correct answers: {total_valid}")
            print(f"Round 1 accuracy: {round1_correct}/{total_valid} = {round1_accuracy*100:.2f}% ± {round1_std*100:.2f}%")
            print(f"Round 2 accuracy: {round2_correct}/{total_valid} = {round2_accuracy*100:.2f}% ± {round2_std*100:.2f}%")
            print(f"Accuracy difference (Round 2 - Round 1): {(round2_accuracy - round1_accuracy)*100:.2f}%")
            
            # Calculate agreement between Round 1 and Round 2
            agreement = (valid_rows['Round 1 Letter'] == valid_rows['Round 2 Letter']).sum()
            agreement_rate = agreement / total_valid if total_valid > 0 else 0
            print(f"Agreement between Round 1 and Round 2: {agreement}/{total_valid} = {agreement_rate*100:.2f}%")
            
            # Calculate conditional accuracy
            both_correct = ((valid_rows['Round 1 Letter'] == valid_rows['Correct Letter']) & 
                            (valid_rows['Round 2 Letter'] == valid_rows['Correct Letter'])).sum()
            
            round1_only_correct = ((valid_rows['Round 1 Letter'] == valid_rows['Correct Letter']) & 
                                  (valid_rows['Round 2 Letter'] != valid_rows['Correct Letter'])).sum()
            
            round2_only_correct = ((valid_rows['Round 1 Letter'] != valid_rows['Correct Letter']) & 
                                  (valid_rows['Round 2 Letter'] == valid_rows['Correct Letter'])).sum()
            
            both_incorrect = ((valid_rows['Round 1 Letter'] != valid_rows['Correct Letter']) & 
                             (valid_rows['Round 2 Letter'] != valid_rows['Correct Letter'])).sum()
            
            print("\nConditional Accuracy:")
            print(f"Both rounds correct: {both_correct}/{total_valid} = {both_correct/total_valid*100:.2f}%")
            print(f"Round 1 correct, Round 2 incorrect: {round1_only_correct}/{total_valid} = {round1_only_correct/total_valid*100:.2f}%")
            print(f"Round 1 incorrect, Round 2 correct: {round2_only_correct}/{total_valid} = {round2_only_correct/total_valid*100:.2f}%")
            print(f"Both rounds incorrect: {both_incorrect}/{total_valid} = {both_incorrect/total_valid*100:.2f}%")
            
            # Add statistical significance testing using McNemar's test
            print("\n" + "="*50)
            print("STATISTICAL SIGNIFICANCE TESTING")
            print("="*50)
            
            # Create contingency table for test
            contingency_table = np.array([[both_correct, round1_only_correct], 
                                         [round2_only_correct, both_incorrect]])
            
            print("Contingency table [1: correct, 0: incorrect]:")
            print("          | Round 2 = 1 | Round 2 = 0 |")
            print("----------|------------|------------|")
            print(f"Round 1 = 1 | {both_correct:10d} | {round1_only_correct:10d} |")
            print(f"Round 1 = 0 | {round2_only_correct:10d} | {both_incorrect:10d} |")
            
            # Manual implementation of McNemar's test
            # The test statistic is (b - c)^2 / (b + c) where b and c are the off-diagonal elements
            b = round1_only_correct  # Round 1 correct, Round 2 incorrect
            c = round2_only_correct  # Round 1 incorrect, Round 2 correct
            
            # Calculate chi-square value with continuity correction
            chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
            p_value = 1 - stats.chi2.cdf(chi2, 1)  # 1 degree of freedom
            
            print(f"\nMcNemar's test (comparing Round 1 and Round 2):")
            print(f"Chi-squared statistic: {chi2:.4f}")
            print(f"p-value: {p_value:.8f}")
            
            if p_value < 0.05:
                print("Result: There is a statistically significant difference between Round 1 and Round 2 accuracy (p < 0.05)")
            else:
                print("Result: There is no statistically significant difference between Round 1 and Round 2 accuracy (p ≥ 0.05)")
            
            # Check if there's a data_source column for per-source analysis
            if 'data_source' in valid_rows.columns:
                print("\n" + "="*80)
                print(" "*30 + "ANALYSIS BY DATA SOURCE")
                print("="*80)
                
                data_sources = sorted(valid_rows['data_source'].unique())
                
                # First, print a summary table of all data sources
                print("\nSUMMARY TABLE FOR ALL DATA SOURCES:")
                print(f"{'Data Source':<15} | {'Total Rows':<10} | {'Round 1 Acc':<15} | {'Round 2 Acc':<15} | {'Difference':<10} | {'Significant':<10}")
                print("-" * 85)
                
                source_results = []
                
                for source in data_sources:
                    source_df = valid_rows[valid_rows['data_source'] == source]
                    source_total = len(source_df)
                    
                    # Calculate conditions for this source
                    s_condition1 = (source_df['Round 1 Letter'] == source_df['Correct Letter']) & (source_df['Round 2 Letter'] == source_df['Correct Letter'])
                    s_count1 = s_condition1.sum()
                    
                    s_condition2 = (source_df['Round 1 Letter'] == source_df['Correct Letter']) & (source_df['Round 2 Letter'] != source_df['Correct Letter'])
                    s_count2 = s_condition2.sum()
                    
                    s_condition3 = (source_df['Round 1 Letter'] != source_df['Correct Letter']) & (source_df['Round 2 Letter'] == source_df['Correct Letter'])
                    s_count3 = s_condition3.sum()
                    
                    s_condition4 = (source_df['Round 1 Letter'] != source_df['Correct Letter']) & (source_df['Round 2 Letter'] != source_df['Correct Letter'])
                    s_count4 = s_condition4.sum()
                    
                    # Calculate accuracy for this source
                    s_round1_correct = s_count1 + s_count2
                    s_round2_correct = s_count1 + s_count3
                    s_round1_acc = s_round1_correct / source_total if source_total > 0 else 0
                    s_round2_acc = s_round2_correct / source_total if source_total > 0 else 0
                    s_diff = s_round2_acc - s_round1_acc
                    
                    # Run statistical test
                    s_significant = "N/A"
                    s_p_value = 1.0
                    
                    if (s_count2 + s_count3) > 0:  # Only if there are disagreements
                        # Calculate chi-square value with continuity correction
                        s_chi2 = (abs(s_count2 - s_count3) - 1)**2 / (s_count2 + s_count3)
                        s_p_value = 1 - stats.chi2.cdf(s_chi2, 1)  # 1 degree of freedom
                        s_significant = "Yes" if s_p_value < 0.05 else "No"
                    
                    # Store results for later
                    source_results.append({
                        'source': source,
                        'total': source_total,
                        'count1': s_count1,
                        'count2': s_count2,
                        'count3': s_count3,
                        'count4': s_count4,
                        'round1_acc': s_round1_acc,
                        'round2_acc': s_round2_acc,
                        'difference': s_diff,
                        'p_value': s_p_value,
                        'significant': s_significant
                    })
                    
                    # Print summary row
                    print(f"{source:<15} | {source_total:<10} | {s_round1_acc*100:>6.2f}% ({s_round1_correct}) | {s_round2_acc*100:>6.2f}% ({s_round2_correct}) | {s_diff*100:>+8.2f}% | {s_significant:<10}")
                
                # Now print detailed analysis for each data source
                for result in source_results:
                    source = result['source']
                    source_total = result['total']
                    s_count1 = result['count1']
                    s_count2 = result['count2']
                    s_count3 = result['count3']
                    s_count4 = result['count4']
                    s_round1_acc = result['round1_acc']
                    s_round2_acc = result['round2_acc']
                    s_p_value = result['p_value']
                    s_significant = result['significant']
                    
                    print("\n" + "="*60)
                    print(f" DATA SOURCE: {source} (Total rows: {source_total})")
                    print("="*60)
                    
                    # Print counts
                    print("\nCondition Breakdown:")
                    print(f"Condition 1 (Both rounds correct): {s_count1} rows ({s_count1/source_total*100:.2f}%)")
                    print(f"Condition 2 (Round 1 correct, Round 2 incorrect): {s_count2} rows ({s_count2/source_total*100:.2f}%)")
                    print(f"Condition 3 (Round 1 incorrect, Round 2 correct): {s_count3} rows ({s_count3/source_total*100:.2f}%)")
                    print(f"Condition 4 (Both rounds incorrect): {s_count4} rows ({s_count4/source_total*100:.2f}%)")
                    
                    # Calculate accuracy for this source
                    s_round1_correct = s_count1 + s_count2
                    s_round2_correct = s_count1 + s_count3
                    print(f"\nAccuracy Comparison:")
                    print(f"Round 1 accuracy: {s_round1_correct}/{source_total} = {s_round1_acc*100:.2f}%")
                    print(f"Round 2 accuracy: {s_round2_correct}/{source_total} = {s_round2_acc*100:.2f}%")
                    print(f"Difference: {(s_round2_acc - s_round1_acc)*100:+.2f}%")
                    
                    # Statistical test results
                    print("\nStatistical Significance:")
                    if s_significant != "N/A":
                        print(f"Chi-squared statistic: {(abs(s_count2 - s_count3) - 1)**2 / (s_count2 + s_count3):.4f}" if (s_count2 + s_count3) > 0 else "N/A")
                        print(f"p-value: {s_p_value:.8f}")
                        
                        if s_p_value < 0.05:
                            print("Result: Statistically significant difference (p < 0.05)")
                        else:
                            print("Result: No statistically significant difference (p ≥ 0.05)")
                    else:
                        print("Insufficient disagreements for statistical testing")
            
            # After the existing data source analysis
            
            # Check if there's a data_source_corr column for additional analysis
            if 'data_source_corr' in valid_rows.columns:
                print("\n" + "="*80)
                print(" "*25 + "ANALYSIS BY DATA_SOURCE_CORR")
                print("="*80)
                
                data_sources_corr = sorted(valid_rows['data_source_corr'].unique())
                
                # First, print a summary table of all data sources
                print("\nSUMMARY TABLE FOR ALL DATA_SOURCE_CORR GROUPS:")
                print(f"{'Group':<15} | {'Total Rows':<10} | {'Round 1 Acc ± Std':<20} | {'Round 2 Acc ± Std':<20} | {'Difference':<10} | {'Significant':<10}")
                print("-" * 95)
                
                source_corr_results = []
                
                for source in data_sources_corr:
                    source_df = valid_rows[valid_rows['data_source_corr'] == source]
                    source_total = len(source_df)
                    
                    # Calculate conditions for this source
                    s_condition1 = (source_df['Round 1 Letter'] == source_df['Correct Letter']) & (source_df['Round 2 Letter'] == source_df['Correct Letter'])
                    s_count1 = s_condition1.sum()
                    
                    s_condition2 = (source_df['Round 1 Letter'] == source_df['Correct Letter']) & (source_df['Round 2 Letter'] != source_df['Correct Letter'])
                    s_count2 = s_condition2.sum()
                    
                    s_condition3 = (source_df['Round 1 Letter'] != source_df['Correct Letter']) & (source_df['Round 2 Letter'] == source_df['Correct Letter'])
                    s_count3 = s_condition3.sum()
                    
                    s_condition4 = (source_df['Round 1 Letter'] != source_df['Correct Letter']) & (source_df['Round 2 Letter'] != source_df['Correct Letter'])
                    s_count4 = s_condition4.sum()
                    
                    # Calculate accuracy and standard deviation for this source
                    s_round1_correct = s_count1 + s_count2
                    s_round2_correct = s_count1 + s_count3
                    s_round1_acc = s_round1_correct / source_total if source_total > 0 else 0
                    s_round2_acc = s_round2_correct / source_total if source_total > 0 else 0
                    
                    # Calculate standard deviations
                    s_round1_std = binary_std(s_round1_acc, source_total)
                    s_round2_std = binary_std(s_round2_acc, source_total)
                    
                    s_diff = s_round2_acc - s_round1_acc
                    
                    # Run statistical test
                    s_significant = "N/A"
                    s_p_value = 1.0
                    
                    if (s_count2 + s_count3) > 0:  # Only if there are disagreements
                        # Calculate chi-square value with continuity correction
                        s_chi2 = (abs(s_count2 - s_count3) - 1)**2 / (s_count2 + s_count3)
                        s_p_value = 1 - stats.chi2.cdf(s_chi2, 1)  # 1 degree of freedom
                        s_significant = "Yes" if s_p_value < 0.05 else "No"
                    
                    # Store results for later
                    source_corr_results.append({
                        'source': source,
                        'total': source_total,
                        'count1': s_count1,
                        'count2': s_count2,
                        'count3': s_count3,
                        'count4': s_count4,
                        'round1_acc': s_round1_acc,
                        'round2_acc': s_round2_acc,
                        'round1_std': s_round1_std,
                        'round2_std': s_round2_std,
                        'difference': s_diff,
                        'p_value': s_p_value,
                        'significant': s_significant
                    })
                    
                    # Print summary row with standard deviations
                    print(f"{source:<15} | {source_total:<10} | {s_round1_acc*100:>6.2f}% ± {s_round1_std*100:.2f}% | {s_round2_acc*100:>6.2f}% ± {s_round2_std*100:.2f}% | {s_diff*100:>+8.2f}% | {s_significant:<10}")
                
                # Now print detailed analysis for each data source
                for result in source_corr_results:
                    source = result['source']
                    source_total = result['total']
                    s_count1 = result['count1']
                    s_count2 = result['count2']
                    s_count3 = result['count3']
                    s_count4 = result['count4']
                    s_round1_acc = result['round1_acc']
                    s_round2_acc = result['round2_acc']
                    s_round1_std = result['round1_std']
                    s_round2_std = result['round2_std']
                    s_p_value = result['p_value']
                    s_significant = result['significant']
                    
                    print("\n" + "="*60)
                    print(f" DATA_SOURCE_CORR: {source} (Total rows: {source_total})")
                    print("="*60)
                    
                    # Print counts
                    print("\nCondition Breakdown:")
                    print(f"Condition 1 (Both rounds correct): {s_count1} rows ({s_count1/source_total*100:.2f}%)")
                    print(f"Condition 2 (Round 1 correct, Round 2 incorrect): {s_count2} rows ({s_count2/source_total*100:.2f}%)")
                    print(f"Condition 3 (Round 1 incorrect, Round 2 correct): {s_count3} rows ({s_count3/source_total*100:.2f}%)")
                    print(f"Condition 4 (Both rounds incorrect): {s_count4} rows ({s_count4/source_total*100:.2f}%)")
                    
                    # Calculate accuracy for this source
                    s_round1_correct = s_count1 + s_count2
                    s_round2_correct = s_count1 + s_count3
                    print(f"\nAccuracy Comparison (with standard deviation):")
                    print(f"Round 1 accuracy: {s_round1_correct}/{source_total} = {s_round1_acc*100:.2f}% ± {s_round1_std*100:.2f}%")
                    print(f"Round 2 accuracy: {s_round2_correct}/{source_total} = {s_round2_acc*100:.2f}% ± {s_round2_std*100:.2f}%")
                    print(f"Difference: {(s_round2_acc - s_round1_acc)*100:+.2f}%")
                    
                    # Statistical test results
                    print("\nStatistical Significance:")
                    if s_significant != "N/A":
                        print(f"Chi-squared statistic: {(abs(s_count2 - s_count3) - 1)**2 / (s_count2 + s_count3):.4f}" if (s_count2 + s_count3) > 0 else "N/A")
                        print(f"p-value: {s_p_value:.8f}")
                        
                        if s_p_value < 0.05:
                            print("Result: Statistically significant difference (p < 0.05)")
                        else:
                            print("Result: No statistically significant difference (p ≥ 0.05)")
                    else:
                        print("Insufficient disagreements for statistical testing")
            
            # Save the updated df2 to a new file
            output_path = "/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_Before_After_Results.csv"
            df2.to_csv(output_path, index=False)
            print(f"\nSaved updated df2 to '{output_path}'")
        else:
            print("\nWarning: 'Correct Letter' column not found. Cannot calculate accuracy.")
    else:
        missing_cols = []
        if 'gpt_direct_prediction' not in df.columns:
            missing_cols.append('gpt_direct_prediction')
        if 'QA_ID' not in df.columns:
            missing_cols.append('QA_ID')
        print(f"\nWarning: Could not merge predictions because df is missing these columns: {missing_cols}")
    
    # Save the updated df2 to a new file
    output_path = "/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_Before_After_Results.csv"
    df2.to_csv(output_path, index=False)
    print(f"\nSaved updated df2 to '{output_path}'")
else:
    print("\nWarning: 'ID_corr' column not found in df2. Available columns are:", df2.columns.tolist())