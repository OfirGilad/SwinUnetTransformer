import subprocess

# Specify the command to run your script
script_command = "python ./main.py"

# Specify the log file path
log_file_path = "output.log"

# Open the log file in append mode to preserve previous logs
with open(log_file_path, "a") as log_file:
    # Run the script asynchronously and redirect its output to the log file
    process = subprocess.Popen(script_command, stdout=log_file, stderr=log_file, shell=True)

# The main script continues here without waiting for the subprocess to finish
# You can perform other tasks or exit the script if needed

subprocess_pid = process.pid
print(f"Subprocess PID: {subprocess_pid}")
