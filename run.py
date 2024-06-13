import subprocess
import os

# Directory containing the scripts
models_dir = 'models'

# List of Python files to run
python_files = [
    'decisiontree.py',
    'randomforest.py',
    'gradientboosting.py',
    'supportvector.py',
    'logreg.py',
    'k-nearest.py',
]

for file in python_files:
    file_path = os.path.join(models_dir, file)
    subprocess.run(['python', file_path])
