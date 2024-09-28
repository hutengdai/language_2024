import pandas as pd
import chardet

def process_data(file_path, output_path):
    processed_data = []

    # Detect the encoding of the file
    with open(file_path, 'rb') as file:
        rawdata = file.read()
        result = chardet.detect(rawdata)
        file_encoding = result['encoding']
    
    print("Detected Encoding:", file_encoding)

    # Read the file with the detected encoding
    with open(file_path, encoding=file_encoding) as file:
        data = file.readlines()

    for line in data:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        sf, analysis = parts
        lemmas = []
        analyses = []
        segmentations = []
        seen = set()

        for morpheme in analysis.split():
            lemma_analysis = morpheme.split(":")
            if len(lemma_analysis) != 2:
                continue

            lemma, ana = lemma_analysis
            # Check for duplicates and stop processing if found
            if lemma in seen:
                break
            seen.add(lemma)

            lemmas.append(lemma)
            analyses.append(ana)
            segmentations.append(lemma)

        processed_data.append({
            "Lemma": sf,
            "SF": sf,
            "Analysis": "-".join(analyses),
            "Segmentation": "-".join(segmentations),
            "SF_Frequency": 1
        })

    df = pd.DataFrame(processed_data)
    df = df.groupby(['Lemma', 'SF', 'Analysis', 'Segmentation']).sum().reset_index()

    # Write the DataFrame to a new file in the detected encoding
    df.to_csv(output_path, index=False, sep='\t', encoding=file_encoding)

    return df



file_path = 'data/finnish/morpho-fin.txt'
output_path = 'data/finnish/morpho-fin-cleaned.txt'
df_sorted = process_data(file_path, output_path)
print(df_sorted)