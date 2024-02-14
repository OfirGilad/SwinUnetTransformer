import subprocess

# Model subprocess
subprocess_pid = "32672"

# Kill the subprocess using its PID
print(f"Killing subprocess with PID: {subprocess_pid}")
subprocess.Popen(["kill", "-9", str(subprocess_pid)])
