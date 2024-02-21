import subprocess
import datetime
import os

# Define the command to run
command = "python3 main.py --dataset animalkingdom --model timesformerclipinitvideoguide --animal "

# Create a list to hold subprocesses
processes = []

# Where the sub csv files output to
directory = '../mount/data/dataset/AnimalKingdom/action_recognition/annotation/train_splits'

# Make the directory for all output files, name it to the date.
current_date = datetime.datetime.now()
date_string = current_date.strftime("%m-%d %H:%M")

output_dir = f"Output/{date_string}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# For each animal csv
files = os.listdir(directory)
    
# Iterate through each file
for file_name in files:
    # Extract the name of the animal from the file name
    name = file_name.split('.')[0]
    
    if '.ipynb' not in name:
        
        output_file_path = f"{output_dir}/{name}.txt"

        # Start the subprocess and redirect stdout and stderr to PIPE
        process = subprocess.Popen(command + name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((process, output_file_path))

        
# Wait for all subprocesses to finish and write their output to files
for process, output_file_path in processes:
    stdout, stderr = process.communicate()
    with open(output_file_path, "w") as file:
        file.write("Standard Output:\n")
        file.write(stdout)
        file.write("\nStandard Error:\n")
        file.write(stderr)

# All processes have finished
print("Split Main has finished!")