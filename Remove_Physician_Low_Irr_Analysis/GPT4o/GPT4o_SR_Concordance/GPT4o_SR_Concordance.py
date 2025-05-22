import pandas as pd
import re
import numpy as np

# Read the CSV files
df = pd.read_csv('/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_SR_Concordance/MAJORITY_Vote_GPT4o_Self_Reported__Labels.csv')
df2 = pd.read_csv('/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_SR_Concordance/Centaur_1300_FirstRound.csv')
physician_df = pd.read_csv('/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_SR_Concordance/Physician_Selected_Sentence_IDs.csv')

# Check the initial lengths
print(f"Initial df length: {len(df)}")
print(f"Initial df2 length: {len(df2)}")
print(f"Physician df length: {len(physician_df)}")

# Perform a left join to maintain df2's length
# This ensures all records from df2 are kept, and matches with df where possible
result = pd.merge(df2, df, left_on="Origin", right_on="ID", how="left")

# Function to collect column numbers with specific values
def collect_column_numbers(row, value_list):
    matching_columns = []
    # Skip non-categorical columns that we know are in the dataframe
    skip_columns = ['ID', 'Origin', 'high', 'low/irr', 'high_number', 'low_number', 'sen_num_diff', 'match_percentage', 'human_Contain_Match'] 
    for col in row.index:
        if col not in skip_columns:
            if any(val in str(row[col]) for val in value_list):
                # Extract the number from column name (e.g., "label_5" -> "5")
                match = re.search(r'label_(\d+)', col)
                if match:
                    matching_columns.append(match.group(1))
    return ", ".join(matching_columns) if matching_columns else ""

# Add columns for collecting column numbers with relevance categories
result['high'] = result.apply(lambda row: collect_column_numbers(row, ["High Relevance"]), axis=1)
result['low/irr'] = result.apply(lambda row: collect_column_numbers(row, ["Low Relevance", "Irrelevant"]), axis=1)

# Function to count the number of items in a comma-separated string
def count_items(value):
    if not value or pd.isna(value):
        return 0
    return len(value.split(', '))

# Add columns for counting the number of occurrences
result['high_number'] = result['high'].apply(count_items)
result['low_number'] = result['low/irr'].apply(count_items)

# Merge with physician data to get the 'keep_k' column and 'human_sentence_ids'
# Assuming 'Origin' in result matches with a column in physician_df (checking the first few rows)
print("\nFirst few rows of physician_df:")
print(physician_df.head())

# Check for potential join column in physician_df
join_column = None
potential_columns = ['ID', 'Origin', 'id', 'origin']
for col in potential_columns:
    if col in physician_df.columns:
        join_column = col
        print(f"Found join column in physician_df: {join_column}")
        break

if join_column:
    # Make sure 'human_sentence_ids' is in physician_df
    if 'human_sentence_ids' in physician_df.columns:
        # Merge based on the found join column, including human_sentence_ids
        result = pd.merge(result, physician_df[[join_column, 'keep_k', 'human_sentence_ids']], 
                        left_on="Origin", right_on=join_column, how="left")
        
        # If the column names are different, drop the duplicate join column
        if join_column != "Origin":
            result.drop(columns=[join_column], inplace=True)
        
        # Calculate the numeric difference for stats
        result['diff_numeric'] = result['high_number'] - result['keep_k']
        
        # Calculate the difference with a sign for display
        result['sen_num_diff'] = result.apply(
            lambda row: f"+{row['diff_numeric']}" if row['diff_numeric'] > 0 
                      else f"{row['diff_numeric']}", axis=1)
        
        # Function to calculate the match percentage
        def calculate_match_percentage(row):
            if pd.isna(row['high']) or pd.isna(row['human_sentence_ids']):
                return 0.0
            
            # Convert comma-separated strings to sets of numbers
            high_set = set(row['high'].split(', ') if row['high'] else [])
            human_set = set(str(row['human_sentence_ids']).split(', ') if not pd.isna(row['human_sentence_ids']) else [])
            
            # Find matches (intersection)
            matches = high_set.intersection(human_set)
            
            # Calculate percentage based on the maximum number
            denominator = max(row['high_number'], row['keep_k'])
            if denominator == 0:
                return 0.0
            
            return (len(matches) / denominator) * 100
        
        # Function to calculate how many human sentence IDs match with high relevance
        def calculate_human_contain_match(row):
            if pd.isna(row['high']) or pd.isna(row['human_sentence_ids']) or row['keep_k'] == 0:
                return 0.0
            
            # Convert comma-separated strings to sets of numbers
            high_set = set(row['high'].split(', ') if row['high'] else [])
            human_set = set(str(row['human_sentence_ids']).split(', ') if not pd.isna(row['human_sentence_ids']) else [])
            
            # Find matches (intersection)
            matches = high_set.intersection(human_set)
            
            # Calculate percentage based on human keep_k
            return (len(matches) / row['keep_k']) * 100
        
        # Add match percentage columns
        result['match_percentage'] = result.apply(calculate_match_percentage, axis=1)
        result['human_Contain_Match'] = result.apply(calculate_human_contain_match, axis=1)
        
        # Calculate statistics
        diff_mean = result['diff_numeric'].mean()
        diff_std = result['diff_numeric'].std()
        match_mean = result['match_percentage'].mean()
        match_std = result['match_percentage'].std()
        human_match_mean = result['human_Contain_Match'].mean()
        human_match_std = result['human_Contain_Match'].std()
        
        print(f"\nStatistics for sentence number differences:")
        print(f"Mean difference: {diff_mean:.2f}")
        print(f"Standard deviation: {diff_std:.2f}")
        
        print(f"\nStatistics for match percentage (matches / max(high_number, keep_k)):")
        print(f"Mean match percentage: {match_mean:.2f}%")
        print(f"Standard deviation: {match_std:.2f}%")
        
        print(f"\nStatistics for human_Contain_Match (matches / keep_k):")
        print(f"Mean percentage: {human_match_mean:.2f}%")
        print(f"Standard deviation: {human_match_std:.2f}%")
        
        # Calculate match percentage statistics by data source category
        print("\nMatch percentage statistics by data_source_df3 category:")
        
        # For original match percentage
        print("\nFor match_percentage:")
        match_by_source = result.groupby('data_source_df3')['match_percentage'].agg(['mean', 'std', 'count'])
        match_by_source.columns = ['Mean Match %', 'Std Dev', 'Count']
        match_by_source['Mean Match %'] = match_by_source['Mean Match %'].round(2)
        match_by_source['Std Dev'] = match_by_source['Std Dev'].round(2)
        print(match_by_source.sort_values('Mean Match %', ascending=False))
        
        # For human_Contain_Match
        print("\nFor human_Contain_Match:")
        human_match_by_source = result.groupby('data_source_df3')['human_Contain_Match'].agg(['mean', 'std', 'count'])
        human_match_by_source.columns = ['Mean Match %', 'Std Dev', 'Count']
        human_match_by_source['Mean Match %'] = human_match_by_source['Mean Match %'].round(2)
        human_match_by_source['Std Dev'] = human_match_by_source['Std Dev'].round(2)
        print(human_match_by_source.sort_values('Mean Match %', ascending=False))
    else:
        print("Warning: 'human_sentence_ids' column not found in physician_df")
        print("Available columns in physician_df:", physician_df.columns.tolist())
else:
    print("Warning: Could not find appropriate join column in physician_df")

# Verify the final dataframe has the same length as df2
print(f"Final result length: {len(result)}")
print(f"Length preserved: {len(result) == len(df2)}")

# Print the first 5 rows of the result dataframe
print("\nFirst 5 rows of the result dataframe:")
pd.set_option('display.max_columns', 15)  # Increased to show all columns
pd.set_option('display.width', 1000)      # Increase display width
print(result.head())

# Save the result to a new CSV file
result.to_csv('/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/GPT4o/GPT4o_SR_Concordance/GPT4o_SR_Concordance_Result.csv', index=False)

print("Matching completed and result saved to GPT4o_SR_Concordance_Result.csv")
