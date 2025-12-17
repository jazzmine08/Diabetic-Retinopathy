# run_scripts.py
import subprocess
import threading
import shlex
import os
import time
from collections import deque

def _run_and_log(command, log_file):
    # open subprocess and stream stdout/stderr to log_file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- START {time.asctime()} ---\n")
        f.flush()
        process = subprocess.Popen(shlex.split(command),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True,
                                   bufsize=1)
        for line in process.stdout:
            f.write(line)
            f.flush()
        process.wait()
        f.write(f"--- FINISHED {time.asctime()} (returncode {process.returncode}) ---\n")
        f.flush()

def run_script_async(command, log_file):
    # spawn a thread to run the command (non-blocking for Flask)
    thread = threading.Thread(target=_run_and_log, args=(command, log_file), daemon=True)
    thread.start()
    return thread

def run_script_blocking(command, log_file):
    _run_and_log(command, log_file)

def last_log_tail(log_path, lines=100):
    if not os.path.exists(log_path):
        return f"(Log file {os.path.basename(log_path)} not found yet.)"
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            dq = deque(f, maxlen=lines)
            return ''.join(dq)
    except Exception as e:
        return f"(Error membaca log: {e})"
