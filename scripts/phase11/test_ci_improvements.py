#!/usr/bin/env python3
"""CI Improvements Tests"""
import sys
from pathlib import Path

print("CI Improvements Integration Tests")
print("=" * 50)

# Check workflow exists
workflows = Path(".github/workflows")
if workflows.exists():
    print("✓ Workflows directory exists")
else:
    print("✗ Workflows directory missing")
    sys.exit(1)

# Check pytest.ini
if Path("pytest.ini").exists():
    print("✓ pytest.ini exists")
else:
    print("✗ pytest.ini missing")
    sys.exit(1)

print("\nAll tests passed! ✓")
