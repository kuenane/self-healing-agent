import os
import sys

# Make the repo root importable so `from src...` works under pytest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
