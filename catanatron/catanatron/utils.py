import os


def format_secs(secs):
    return f"{secs:.3f} secs"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
