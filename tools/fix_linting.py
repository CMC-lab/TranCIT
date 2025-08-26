#!/usr/bin/env python3
"""
Automated linting fix script for TranCIT repository.
This script applies multiple automated fixes to resolve linting issues.
"""

import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and report success/failure."""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} failed - command not found")
        print(f"Please install the required tool first")
        return False

def main():
    """Main linting fix process."""
    print("🚀 Starting automated linting fixes for TranCIT repository")
    
    os.chdir(Path(__file__).parent)
    
    targets = ["trancit/", "tests/", "examples/"]
    
    success_count = 0
    total_fixes = 0
    
    total_fixes += 1
    if run_command([
        "autoflake", 
        "--in-place", 
        "--remove-unused-variables",
        "--remove-all-unused-imports",
        "--recursive"
    ] + targets, "Removing unused imports and variables"):
        success_count += 1
    
    total_fixes += 1
    if run_command([
        "isort",
        "--profile", "black",
        "--line-length", "88"
    ] + targets, "Organizing imports"):
        success_count += 1
    
    total_fixes += 1
    if run_command([
        "black",
        "--line-length", "88",
        "--target-version", "py39"
    ] + targets, "Formatting code with Black"):
        success_count += 1
    
    print("\n🔧 Applying manual fixes for remaining issues")
    
    total_fixes += 1
    try:
        fix_comparison_issues()
        print("✅ Fixed comparison issues")
        success_count += 1
    except Exception as e:
        print(f"❌ Failed to fix comparison issues: {e}")
    
    total_fixes += 1
    if run_command([
        "flake8",
        "--max-line-length", "88",
        "--extend-ignore", "E203,W503,E712",
        "--exclude", "__pycache__,*.egg-info,build,dist"
    ] + targets, "Checking remaining linting issues"):
        success_count += 1
    
    print(f"\n📊 SUMMARY")
    print(f"✅ Successfully completed: {success_count}/{total_fixes} fixes")
    
    if success_count == total_fixes:
        print("🎉 All automated fixes completed successfully!")
        print("Your code should now pass most linting checks.")
    else:
        print("⚠️  Some fixes failed. Please check the errors above.")
        print("You may need to install missing tools or fix issues manually.")
    
    print("\n📋 NEXT STEPS:")
    print("1. Review the changes: git diff")
    print("2. Test your code: python -m pytest tests/")
    print("3. Commit the fixes: git add . && git commit -m 'Fix linting issues'")

def fix_comparison_issues():
    """Fix E712 comparison issues programmatically."""
    import re
    
    python_files = []
    for target in ["trancit/", "tests/", "examples/"]:
        for file_path in Path(target).rglob("*.py"):
            python_files.append(file_path)
    
    patterns = [
        (r'== True\b', 'is True'),
        (r'== False\b', 'is False'),
        (r'!= True\b', 'is not True'),
        (r'!= False\b', 'is not False'),
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  Fixed comparisons in {file_path}")
        
        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")

if __name__ == "__main__":
    main()
