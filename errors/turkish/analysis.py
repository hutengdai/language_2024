import pandas as pd
import ast
from collections import defaultdict

# Define the path to the error file
error_file_path = 'errors/turkish/turkish_0.4_maximal.txt'

# A function to read the error file and parse the Python dictionary data
def read_error_file(filepath):
    with open(filepath, 'r') as file:
        # Use ast.literal_eval to safely evaluate the string as a Python dict
        data = [ast.literal_eval(line.strip()) for line in file]
    return data

# Read the error file
error_data = read_error_file(error_file_path)


print(error_data)

