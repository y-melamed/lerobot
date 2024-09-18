# import os
# import re

# def rename_files_in_directory(directory):
#     # Get all files in the directory
#     files = os.listdir(directory)
    
#     # Dictionary to store files by prefix
#     prefix_dict = {}

#     # Regex pattern to extract the prefix and episode number
#     pattern = r"(observation\.images\.[^.]+)_episode_(\d+)\.mp4"

#     for filename in files:
#         match = re.match(pattern, filename)
#         if match:
#             prefix = match.group(1)
#             episode_num = int(match.group(2))

#             # Group files by prefix in a dictionary
#             if prefix not in prefix_dict:
#                 prefix_dict[prefix] = []

#             prefix_dict[prefix].append((filename, episode_num))

#     # Now process each prefix group
#     for prefix, file_list in prefix_dict.items():
#         # Sort files by the original episode number
#         file_list.sort(key=lambda x: x[1])

#         # Rename files starting from 000000
#         for i, (original_file, _) in enumerate(file_list):
#             new_filename = f"{prefix}_episode_{i:06d}.mp4"
#             original_file_path = os.path.join(directory, original_file)
#             new_file_path = os.path.join(directory, new_filename)
            
#             # Rename the file
#             # os.rename(original_file_path, new_file_path)
#             print(f"Renamed {original_file} to {new_filename}")

# Example usage
# directory = r'/home/aloha/Desktop/lerobot/data/yanivmel1/rotem_recordings_splits/part3/videos'

import torch

# Load the .pth file
import torch
import re
import os

def update_paths(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'path' and isinstance(value, str):
                data[key] = transform_value(value)
            elif isinstance(value, dict):
                data[key] = update_paths(value)
    return data

def transform_value(old_value):
    pattern = re.compile(r'0000(\d{2})')
    match = pattern.search(old_value)
    if match:
        num_str = match.group(1)
        transformed_num = f'{int(num_str) - 20:06}'
        new_value = pattern.sub(f'0000{transformed_num}', old_value)
        return new_value
    return old_value

def process_files(directory):
    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):  # Make sure to process only PyTorch files
            full_path = os.path.join(directory, filename)
            data = torch.load(full_path)
            transformed_data = update_paths(data)
            torch.save(transformed_data, full_path)  # Overwrite the original file

# Directory containing the .pt files
directory_path = './data/yanivmel1/rotem_recordings_splits/part3 (copy)/episodes'
# process_files(directory_path)


# Load the .pth file
pth_file_path = './data/yanivmel1/rotem_recordings_splits/part3 (copy)/episodes/episode_000000.pth'
data = torch.load(pth_file_path)

# Print the contents
print(data)





# Print the contents
# print(data)

