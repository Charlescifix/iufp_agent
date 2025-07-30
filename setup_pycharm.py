#!/usr/bin/env python3
"""
PyCharm Interpreter Configuration Helper
Run this to get the correct interpreter path for PyCharm configuration.
"""

import sys
import os
from pathlib import Path

def main():
    print("=== PyCharm Interpreter Configuration ===\n")
    
    # Current Python executable
    python_exe = sys.executable
    print(f"Current Python interpreter: {python_exe}")
    
    # Project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # Virtual environment details
    venv_path = Path(python_exe).parent.parent
    print(f"Virtual environment: {venv_path}")
    
    # Check if we're in the right venv
    expected_venv = project_root / ".venv"
    if venv_path.resolve() == expected_venv.resolve():
        print("[OK] Correct virtual environment is active")
    else:
        print(f"[WARNING] Expected venv: {expected_venv}")
        print(f"[WARNING] Current venv: {venv_path}")
    
    # Package verification
    try:
        import fastapi
        import openai
        import psycopg2
        import sentence_transformers
        print("[OK] Core packages are importable")
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
    
    print("\n=== PyCharm Configuration Instructions ===")
    print("1. Open PyCharm Settings (Ctrl+Alt+S)")
    print("2. Go to Project > Python Interpreter")
    print("3. Click the gear icon > Add...")
    print("4. Select 'Existing environment'")
    print(f"5. Set interpreter path to: {python_exe}")
    print("6. Click OK and apply changes")
    
    print(f"\n=== Quick Copy (Interpreter Path) ===")
    print(f"{python_exe}")

if __name__ == "__main__":
    main()