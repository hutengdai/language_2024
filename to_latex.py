import os

def convert_to_latex(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Add a tab before the first line and split each line into columns
    lines = ['\t' + line for line in lines]

    # Starting the LaTeX tabular environment
    num_columns = len(lines[0].split()) + 1
    latex_content = "\\begin{tabular}{" + "c" * num_columns + "}\n"

    first_line = True
    for line in lines:
        values = line.strip().split()
        if first_line:
            row = ' & ' + ' & '.join(values)
            first_line = False
        else:
            row = ' & '.join('\\good' if x == '1' else ('' if x == '0' else x) for x in values)
        latex_content += row + " \\\\\n"

    # Ending the LaTeX tabular environment
    latex_content += "\\end{tabular}"

    return latex_content


# Directory where your files are stored
directory = 'result/finnish/matrix/'

# Process each file
for filename in os.listdir(directory):
    if filename.startswith("matrix_filtering_nonlocal_flt-T_pad-F_conf-0.975_pen-10_split-0.9_") and filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        latex_table = convert_to_latex(file_path)
        
        # Print or save the LaTeX table as needed
        print(filename)
        print(latex_table)
