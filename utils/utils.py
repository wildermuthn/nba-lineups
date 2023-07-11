import os
import itertools


def get_latest_file(dir_path):
    # Get list of all files only from the given directory
    files = filter(os.path.isfile, os.listdir(dir_path))
    files = [os.path.join(dir_path, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0] if files else None


def all_combinations(elements, replacement):
    n = len(elements)
    for r in range(1, n):
        for indices in itertools.combinations(range(n), r):
            new_elements = [replacement] * n
            for index in indices:
                new_elements[index] = elements[index]
            yield new_elements