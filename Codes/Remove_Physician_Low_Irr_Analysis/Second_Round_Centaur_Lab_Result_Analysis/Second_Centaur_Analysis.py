import pandas as pd

# Load the CSV file
df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Second_Round_Centaur_Lab_Result_Analysis/Round2_CentaurLab_Raw_Data.csv")

# Display the first few rows after removing duplicates
print(df.columns)
# print(df.head())
print(len(df))

# Group by 'data_source_corr' and calculate stats for 'Agreement'
agreement_stats = df.groupby('data_source_corr')['Agreement'].agg(['mean', 'std', 'count', 'min', 'max'])

print(agreement_stats)

# Calculate overall stats for 'Agreement'
overall_stats = df['Agreement'].agg(['mean', 'std', 'count', 'min', 'max'])
print("\nOverall Agreement stats:")
print(overall_stats)