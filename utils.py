# utils.py
import os, json
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)
