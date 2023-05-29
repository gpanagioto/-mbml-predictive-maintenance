import os

# Get the parent directory of the package directory (i.e. myproject directory)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join('..',root_dir,'data')
src_path = os.path.join('..',root_dir,'src')