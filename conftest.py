import sys
import os

# Ensure the project root is on sys.path so tests can import env_common and
# other project-level modules without installing the package.
sys.path.insert(0, os.path.dirname(__file__))
