import pandas as pd

# Read the dataset from the CSV file without header
df = pd.read_csv('data/processed/dataset_smote.csv', header=None, names=["organization", "description", "part_id"])


# Group by organization
grouped = df.groupby('organization')

# Initialize lists to hold the training and evaluation data
train_data = []
eval_data = []

# Split each group into training and evaluation datasets
for name, group in grouped:
    # Shuffle the group
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Determine split index
    split_index = int(len(group) * 0.8)
    
    # Append the data to respective lists
    train_data.append(group.iloc[:split_index])
    eval_data.append(group.iloc[split_index:])

# Concatenate all training and evaluation data
train_df = pd.concat(train_data)
eval_df = pd.concat(eval_data)

# Save to CSV files with headers
train_df.to_csv('data/processed/training_data.csv', index=False, header=True)

# Delete the last row
eval_df = eval_df.iloc[:-1]
eval_df.to_csv('data/evaluation/evaluation_data.csv', index=False, header=True)


