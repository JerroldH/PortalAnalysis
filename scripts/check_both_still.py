from portal_analysis.training.settings import get_base_processed_directory
from pathlib import Path
import pandas as pd

base = get_base_processed_directory()
task_dir = base / 'both_still'

# Check label IDs
labels_path = task_dir / 'docs' / 'weak_supervision_final.csv'
df = pd.read_csv(labels_path).set_index('ID')
print('Label IDs (first 10):')
for i in df.index[:10]:
    print(' ', repr(i))

# Check file stems in right/distances and left/distances
print('\nFile stems in right/distances (first 10):')
right_dir = task_dir / 'right' / 'distances'
if right_dir.exists():
    files = list(right_dir.glob('*.csv'))
    for f in sorted(files)[:10]:
        print(' ', repr(f.stem))
else:
    print(' Directory does not exist:', right_dir)

print('\nFile stems in left/distances (first 10):')
left_dir = task_dir / 'left' / 'distances'
if left_dir.exists():
    files = list(left_dir.glob('*.csv'))
    for f in sorted(files)[:10]:
        print(' ', repr(f.stem))
else:
    print(' Directory does not exist:', left_dir)

# Check test-set file
test_path = task_dir / 'docs' / 'test-set-balanced.csv'
print('\nTest set file exists:', test_path.exists())
if test_path.exists():
    df_test = pd.read_csv(test_path)
    print('Columns:', list(df_test.columns))
    print('First 5 file_names:', list(df_test.iloc[:5, 0]))
