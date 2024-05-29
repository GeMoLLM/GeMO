import re
import json

def parse_file_to_json(file_content):
    pattern = r'(Description|Functionality|Algorithm|Data Structure|Time Complexity|Space Complexity|Tags):\s*(.*)'
    parsed_data = {}

    for line in file_content.splitlines():
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            key = match.group(1).lower().replace(' ', '_')
            value = match.group(2).strip()
            parsed_data[key] = value

    # Fill in missing keys with empty strings
    keys = ['description', 'functionality', 'algorithm', 'data_structure', 'time_complexity', 'space_complexity', 'tags']
    for key in keys:
        if key not in parsed_data:
            parsed_data[key] = ''

    return parsed_data

input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    fileidno = fileid.split('-')[-1]
    pid = file_path.split('_')[-1].split('.')[0]

    for i in range(20):
        filename = f'description_{fileid}_{pid}_solution_{i}.txt'
        with open(filename, 'r') as file:
            file_content = file.read()
            json_data = parse_file_to_json(file_content)
            json.dump(json_data, open(f'parsed_description_{fileid}_{pid}_solution_{i}.json', 'w'), indent=4)
