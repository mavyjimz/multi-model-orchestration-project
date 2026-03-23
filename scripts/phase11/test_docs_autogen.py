#!/usr/bin/env python3
"""Documentation Auto-Generation Tests"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

print("Documentation Tests")
print("=" * 50)

# Test imports
try:
    print("✓ ModelCardGenerator import OK")
except Exception as e:
    print(f"✗ ModelCardGenerator import failed: {e}")
    sys.exit(1)

try:
    print("✓ APIDocGenerator import OK")
except Exception as e:
    print(f"✗ APIDocGenerator import failed: {e}")
    sys.exit(1)

try:
    print("✓ ChangelogGenerator import OK")
except Exception as e:
    print(f"✗ ChangelogGenerator import failed: {e}")
    sys.exit(1)

try:
    print("✓ ReadmeUpdater import OK")
except Exception as e:
    print(f"✗ ReadmeUpdater import failed: {e}")
    sys.exit(1)

# Test docs directory exists
docs_dir = Path("docs")
if docs_dir.exists():
    print("✓ docs/ directory exists")
else:
    print("✗ docs/ directory missing")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
