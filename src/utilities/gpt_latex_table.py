import pandas as pd

def dataframe_to_latex(df, caption, label):
    latex_str = df.to_latex(index=False, column_format='l' + 'r' * (len(df.columns) - 1), 
                            longtable=False, bold_rows=True, 
                            caption=caption, label=label, 
                            header=True, position='htbp',
                            float_format="{:0.2f}".format)
    
    # Customize the LaTeX table to fit user requirements
    lines = latex_str.split('\n')
    custom_lines = []

    # Add packages for booktabs if not present
    custom_lines.append('\\usepackage{booktabs}\n')

    for line in lines:
        if '\\begin{tabular}' in line:
            # Replace tabular with booktabs format
            custom_lines.append('\\begin{tabular}{@{}' + 'l' + 'r' * (len(df.columns) - 1) + '@{}}')
            custom_lines.append('\\toprule')
        elif '\\end{tabular}' in line:
            custom_lines.append('\\bottomrule')
            custom_lines.append(line)
        elif '\\hline' in line:
            # Remove all other hline instances
            pass
        else:
            custom_lines.append(line)

    return '\n'.join(custom_lines)


file_path = 'src/observational_data/Overview of know OB associations.xlsx'
data = pd.read_excel(file_path)

# Convert dataframe to LaTeX format with custom settings
latex_table = dataframe_to_latex(data, "Observational data on known associations", "table:known_associations")

# Output the LaTeX code (display a part of it due to length)
print('\n'.join(latex_table.split('\n')))  # Display the first 20 lines for brevity
print('\n...\n')  # Indicate that the output is truncated
