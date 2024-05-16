import pandas as pd

# Path to the final CSV file
csv_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/labels_tensorflow/labels_modified_v2.csv'

data = pd.read_csv(csv_path, delimiter=',')

# Display the initial rows to understand the structure
initial_rows = data.head(220)
print("Initial Rows:\n", initial_rows)

# Drop the first two rows which are unnecessary
data_cleaned = data.drop([0, 1])

# Manually set the correct headers
headers = ['ID', 'Quadrant', 'Pitting', 'Leukonichie', 'Pete rosii lunula', 'Aspect sfaramicios/crumbling', 'Onicoliza', 'Hemoragii aschie', 'Pata de ulei', 'Hiperkertoza', 'Scor NAPSI']
data_cleaned.columns = headers + data_cleaned.columns[len(headers):].tolist()

# Keep only the relevant columns
data_cleaned = data_cleaned[headers]

# Propagate 'ID' values forward
data_cleaned['ID'] = data_cleaned['ID'].ffill()

# Drop rows where 'Quadrant' is NaN
data_cleaned = data_cleaned.dropna(subset=['Quadrant'])

# Reset index
data_cleaned.reset_index(drop=True, inplace=True)

# Verify the cleaned data
cleaned_rows = data_cleaned.head(220)
print("Cleaned Rows:\n", cleaned_rows)
# Save the cleaned data for future use
data_cleaned.to_csv(csv_path, index=False)
