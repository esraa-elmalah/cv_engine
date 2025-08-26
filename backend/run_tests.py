#!/usr/bin/env python3
"""
Test runner script for CV Engine.
Provides easy commands to run different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CV Engine Test Runner")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "api", "coverage", "quick"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    # Test type specific commands
    if args.test_type == "all":
        print("🧪 Running all tests...")
        success = run_command(base_cmd + ["tests/"], "All tests")
        
    elif args.test_type == "unit":
        print("🧪 Running unit tests...")
        success = run_command(base_cmd + ["tests/unit/"], "Unit tests")
        
    elif args.test_type == "integration":
        print("🧪 Running integration tests...")
        success = run_command(base_cmd + ["tests/integration/"], "Integration tests")
        
    elif args.test_type == "api":
        print("🧪 Running API tests...")
        success = run_command(base_cmd + ["tests/api/"], "API tests")
        
    elif args.test_type == "coverage":
        print("🧪 Running tests with coverage...")
        coverage_cmd = base_cmd + [
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "tests/"
        ]
        success = run_command(coverage_cmd, "Tests with coverage")
        
    elif args.test_type == "quick":
        print("🧪 Running quick tests (unit only)...")
        quick_cmd = base_cmd + [
            "--maxfail=5",
            "--tb=short",
            "tests/unit/"
        ]
        success = run_command(quick_cmd, "Quick tests")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
