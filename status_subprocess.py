import psutil

# Example: Check if a process with PID 1234 is alive
pid_to_check = 32672

def is_process_alive(pid):
    try:
        process = psutil.Process(pid)
        # Check if the process is still running
        return process.is_running()
    except psutil.NoSuchProcess:
        # NoSuchProcess exception is raised if the process with the given PID doesn't exist
        return False

if is_process_alive(pid_to_check):
    print(f"Process with PID {pid_to_check} is alive.")
else:
    print(f"Process with PID {pid_to_check} is not alive.")
