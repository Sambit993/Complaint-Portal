"""
Entry point shim — allows running `python app.py` from the project root.
Automatically redirects to the actual API in student-complaint-api/
"""
import os
import subprocess
import sys

api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student-complaint-api')

# Run the real app.py as a subprocess from the correct directory
result = subprocess.run([sys.executable, 'app.py'], cwd=api_dir)
sys.exit(result.returncode)
