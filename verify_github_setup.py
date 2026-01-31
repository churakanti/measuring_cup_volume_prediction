#!/usr/bin/env python3
"""
GitHub Setup Verification Script

This script checks that all necessary files for GitHub setup are in place
and that the .gitignore is properly excluding large files.

Run this before pushing to GitHub to ensure everything is ready.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print_success(f"{description} exists ({size:,} bytes)")
        return True
    else:
        print_error(f"{description} NOT FOUND at: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        files_count = len(list(Path(dirpath).rglob('*')))
        print_success(f"{description} exists ({files_count} files)")
        return True
    else:
        print_error(f"{description} NOT FOUND at: {dirpath}")
        return False

def check_gitignore_content(gitignore_path):
    """Verify .gitignore has essential exclusions"""
    if not os.path.exists(gitignore_path):
        print_error(".gitignore file not found")
        return False

    with open(gitignore_path, 'r', encoding='utf-8') as f:
        content = f.read()

    essential_exclusions = [
        'BMC_NewResized/',
        'BMC_OR/',
        '*.keras',
        '.venv/',
        '__pycache__/',
        '*.zip'
    ]

    all_present = True
    for exclusion in essential_exclusions:
        if exclusion in content:
            print_success(f".gitignore excludes: {exclusion}")
        else:
            print_warning(f".gitignore MISSING exclusion: {exclusion}")
            all_present = False

    return all_present

def check_requirements_clean(req_path):
    """Check if requirements.txt is cleaned (no robotics packages)"""
    if not os.path.exists(req_path):
        return False

    with open(req_path, 'r', encoding='utf-8') as f:
        content = f.read().lower()

    robotics_keywords = ['rospy', 'gazebo', 'turtlebot', 'moveit']
    found_robotics = []

    for keyword in robotics_keywords:
        if keyword in content:
            found_robotics.append(keyword)

    if found_robotics:
        print_warning(f"Found robotics packages in requirements.txt: {', '.join(found_robotics)}")
        print_info("Consider using the cleaned requirements.txt")
        return False
    else:
        print_success("requirements.txt is clean (no robotics packages)")
        return True

def estimate_repo_size(base_path):
    """Estimate repository size (excluding .gitignore patterns)"""
    total_size = 0
    file_count = 0

    # Read .gitignore patterns
    gitignore_path = os.path.join(base_path, '.gitignore')
    excluded_patterns = []

    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            excluded_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for root, dirs, files in os.walk(base_path):
        # Skip .git directory
        if '.git' in root:
            continue

        # Check if this directory should be excluded
        rel_path = os.path.relpath(root, base_path)
        should_skip = False

        for pattern in excluded_patterns:
            if pattern.rstrip('/') in rel_path:
                should_skip = True
                break

        if should_skip:
            continue

        for file in files:
            file_path = os.path.join(root, file)

            # Check if file extension should be excluded
            should_exclude = False
            for pattern in excluded_patterns:
                if pattern.startswith('*.') and file.endswith(pattern[1:]):
                    should_exclude = True
                    break

            if not should_exclude:
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    file_count += 1
                except:
                    pass

    return total_size, file_count

def main():
    print("\n" + "="*60)
    print("  GitHub Setup Verification Script")
    print("="*60 + "\n")

    base_path = os.getcwd()
    print_info(f"Checking directory: {base_path}\n")

    all_checks_passed = True

    # Check essential files
    print("📄 Checking essential files...\n")

    essential_files = [
        ("README.md", "Main README"),
        (".gitignore", "Git ignore file"),
        ("LICENSE", "License file"),
        ("requirements.txt", "Requirements file"),
        ("GITHUB_SETUP_GUIDE.md", "GitHub setup guide"),
        ("SETUP_SUMMARY.md", "Setup summary"),
    ]

    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    print()

    # Check docs directory
    print("📁 Checking documentation directory...\n")

    docs_files = [
        ("docs/SETUP.md", "Setup documentation"),
        ("docs/DATASET.md", "Dataset documentation"),
        ("docs/MODELS.md", "Models documentation"),
    ]

    for filepath, description in docs_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    print()

    # Check .gitignore content
    print("🚫 Checking .gitignore exclusions...\n")
    if not check_gitignore_content('.gitignore'):
        all_checks_passed = False

    print()

    # Check requirements.txt
    print("📦 Checking requirements.txt...\n")
    check_requirements_clean('requirements.txt')

    print()

    # Estimate repository size
    print("💾 Estimating repository size...\n")
    total_size, file_count = estimate_repo_size(base_path)

    size_mb = total_size / (1024 * 1024)
    print_info(f"Estimated repo size: {size_mb:.2f} MB ({file_count} files)")

    if size_mb > 100:
        print_warning("Repository size > 100 MB. Check if large files are being tracked.")
        print_info("Large files should be excluded by .gitignore")
    else:
        print_success("Repository size is reasonable for GitHub")

    print()

    # Check for large files that might not be excluded
    print("🔍 Checking for large files...\n")
    large_files = []

    for root, dirs, files in os.walk(base_path):
        # Skip excluded directories
        if any(excluded in root for excluded in ['.git', '.venv', 'BMC_NewResized', 'BMC_OR', '__pycache__']):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > 10 * 1024 * 1024:  # Files > 10 MB
                    rel_path = os.path.relpath(file_path, base_path)
                    large_files.append((rel_path, size))
            except:
                pass

    if large_files:
        print_warning(f"Found {len(large_files)} large files (>10 MB):")
        for filepath, size in large_files[:5]:  # Show first 5
            size_mb = size / (1024 * 1024)
            print(f"  - {filepath} ({size_mb:.2f} MB)")
        if len(large_files) > 5:
            print(f"  ... and {len(large_files) - 5} more")
        print_info("Make sure these are excluded by .gitignore")
    else:
        print_success("No large files found (>10 MB)")

    print()

    # Final summary
    print("="*60)
    if all_checks_passed and size_mb < 100:
        print_success("✓ All checks passed! You're ready to push to GitHub.")
        print_info("Next step: Follow GITHUB_SETUP_GUIDE.md to push your code")
    else:
        print_warning("⚠ Some issues detected. Please review above.")
        print_info("Fix any red ✗ items before pushing to GitHub")
    print("="*60 + "\n")

    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())
