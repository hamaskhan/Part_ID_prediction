import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'training_data.csv'
df = pd.read_csv(file_path)

# Get the first three unique organizations
first_three_orgs = df['organization'].unique()[:3]

# Filter the dataset for the first three organizations
filtered_df = df[df['organization'].isin(first_three_orgs)]

# Group by organization and part_id and count the occurrences
org_part_counts = filtered_df.groupby(['organization', 'part_id']).size().reset_index(name='counts')

# Set the plot size
plt.figure(figsize=(12, 8))

# Create a bar plot
sns.barplot(x='organization', y='counts', hue='part_id', data=org_part_counts)

# Add labels and title
plt.xlabel('Organization')
plt.ylabel('Count of Part ID')
plt.title('Count of Part IDs for First Three Organizations')
plt.xticks(rotation=90)
plt.legend(title='Part ID')

# Show the plot
plt.tight_layout()
plt.show()
