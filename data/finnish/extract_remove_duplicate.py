import pandas as pd

# Function to remove adjacent duplicates from a string
def remove_adjacent_duplicates(s):
    """ Remove adjacent duplicates from a string """
    new_s = ""
    for char in s:
        if not new_s or new_s[-1] != char:
            new_s += char
    return new_s

# Reading the Finnish UR manual file
manual_file_path = 'data/finnish/finnish_ur_manual.txt'
manual_df = pd.read_csv(manual_file_path, sep=':', header=None, engine='python')
manual_df.columns = ['Analysis', 'Segmentation']

# Trimming leading spaces in the Segmentation column
manual_df['Segmentation'] = manual_df['Segmentation'].str.strip()

# Applying the function to the 'Segmentation' column in the manual DataFrame
manual_df['Segmentation'] = manual_df['Segmentation'].apply(remove_adjacent_duplicates)
manual_df['Analysis'] = manual_df['Analysis'].apply(remove_adjacent_duplicates)

# Writing the updated DataFrame back to the same file
manual_df.to_csv(manual_file_path, sep=':', index=False, header=False)

# Displaying the first few rows of the updated DataFrame
print(manual_df.head())
