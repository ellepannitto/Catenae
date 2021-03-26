import glob

def get_filenames(input_dir):
    return glob.glob(input_dir+"/*", recursive=True)