import os
import glob

def get_filenames(input_dir):

    if os.path.isdir(input_dir):
        return glob.glob(input_dir+"/*", recursive=True)
    else:
        return [input_dir]