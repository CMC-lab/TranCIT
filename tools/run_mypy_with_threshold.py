#!/usr/bin/env python3
"""
Run mypy with acceptable error threshold for research codebases.
"""

import subprocess
import sys

def run_mypy_check():
    """Run mypy and check if errors are within acceptable threshold."""
    
    ACCEPTABLE_ERRORS = 100
    
    try:
        result = subprocess.run(
            ["mypy", "trancit/"], 
            capture_output=True, 
            text=True
        )
            
        error_lines = [line for line in result.stdout.split('\n') if 'error:' in line]
        error_count = len(error_lines)
        
        print(f"MyPy check completed:")
        print(f"  Found {error_count} errors")
        print(f"  Threshold: {ACCEPTABLE_ERRORS} errors")
        
        if error_count <= ACCEPTABLE_ERRORS:
            print("✅ MyPy check PASSED - errors within acceptable threshold")
            return 0
        else:
            print(f"❌ MyPy check FAILED - {error_count - ACCEPTABLE_ERRORS} errors over threshold")
            print("\nFirst 10 errors:")
            for i, error in enumerate(error_lines[:10]):
                print(f"  {i+1}. {error}")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"Error running mypy: {e}")
        return 1
    except FileNotFoundError:
        print("Error: mypy not found. Run 'pip install mypy'")
        return 1

if __name__ == "__main__":
    sys.exit(run_mypy_check())
