from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent / 'frozen_study'
    sys.path.insert(0, str(folder))
    runpy.run_path(str(folder / 'runner.py'), run_name='__main__')
