import csv

def reduce_csv_values(input_file, output_file, reduction_percentage=8.5):
    """
    Reads a CSV file, reduces each numeric value by the given percentage,
    and writes the results to a new CSV file.
    """
    factor = 1 - (reduction_percentage / 100.0)

    with open(input_file, 'r', newline='') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            new_row = []
            for cell in row:
                try:
                    # Convert cell to float and reduce it
                    value = float(cell)
                    reduced_value = value * factor
                    new_row.append(reduced_value)
                except ValueError:
                    # If conversion to float fails, keep original cell
                    new_row.append(cell)
            writer.writerow(new_row)

# Example usage:
input_file_path = "combo_stag_results.csv"
output_file_path = "output.csv"

reduce_csv_values(input_file_path, output_file_path, reduction_percentage=2.8)
print(f"Reduced values by 8.5% and wrote to {output_file_path}")
