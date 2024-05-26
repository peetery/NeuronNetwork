import os

def save_stat(stat, filename):
    with open(f"{filename}.txt", 'a') as f:
        f.write(f"{stat}\n")

def clear_stats(filename):
    if os.path.exists(f"{filename}.txt"):
        os.remove(f"{filename}.txt")