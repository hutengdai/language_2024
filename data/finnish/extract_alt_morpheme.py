import pandas as pd

# Reading the file into a DataFrame
file_path = 'data/finnish/morpho-fin-cleaned.txt'
df = pd.read_csv(file_path, sep='\t')

def remove_adjacent_duplicates(s):
    """ Remove adjacent duplicates from a string """
    new_s = ""
    for char in s:
        if not new_s or new_s[-1] != char:
            new_s += char
    return new_s

def update_analysis_and_segmentation(row):
    # Splitting Segmentation into segments and applying adjacent duplicates removal
    segments = row['Segmentation'].split('-')
    updated_segments = [remove_adjacent_duplicates(segment) for segment in segments]
    updated_segmentation = '-'.join(updated_segments)

    analysis = row['Analysis'].split('-')

    # Original processing logic
    for i, anal in enumerate(analysis):
        if anal == "3SGPL" and updated_segments[i].startswith("ns"):
            analysis[i] = "3SGPL_nsA"
        if anal == "ILL" and updated_segments[i].startswith("s"):
            analysis[i] = "ILL_sIIn"
        if anal == "DVJA" and (updated_segments[i] == "jo" or updated_segments[i] == "jö"):
            analysis[i] = "DVJO"

    # Merge identical adjacent symbols in analysis
    merged_analysis = '-'.join([remove_adjacent_duplicates(a) for a in analysis])

    return updated_segmentation, merged_analysis

# Applying the function to update the DataFrame
updated_data = df.apply(update_analysis_and_segmentation, axis=1)
df['Segmentation'] = updated_data.apply(lambda x: x[0])
df['Analysis'] = updated_data.apply(lambda x: x[1])


# Saving the updated DataFrame to a new file
output_path = 'data/finnish/morpho-fin-updated.txt'
df.to_csv(output_path, sep='\t', index=False)

