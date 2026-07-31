import csv, os

input_file =  "/home/tp/mem-opt/results/raw/raw_results.csv"
output_file = "/home/tp/mem-opt/results/raw/clean_results.csv"

with open(input_file) as f:
    lines = f.readlines()

header = lines[0]
merged = [header]
# merging the l2_write_back:
i = 1
while i < len(lines):
    line = lines[i].strip()
    if line.startswith('2026-07-27'):
        wb_val = lines[i+1].strip()
        merged.append(line + ',' + wb_val + '\n')
        i += 2
    else:
        i += 2

with open(output_file, 'w') as f:
    f.writelines(merged)

print(f'Original: {len(lines)-1} rows')
print(f'Kept:     {len(merged)-1} rows')
print(f'Saved to: {output_file}')