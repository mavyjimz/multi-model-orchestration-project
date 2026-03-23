#!/usr/bin/env python3
"""Master Documentation Generator CLI"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

print("=" * 50)
print("Documentation Generator")
print("=" * 50)

try:
    from docs.model_card_generator import ModelCardGenerator
    print("\n[1/4] Generating model card...")
    gen = ModelCardGenerator()
    card = gen.generate_card("intent-classifier-sgd", "v1.0.2")
    print(f"  OK: {card['model_name']}")
except Exception as e:
    print(f"  Skip: {e}")

try:
    from docs.api_doc_generator import APIDocGenerator
    print("\n[2/4] Generating API docs...")
    gen = APIDocGenerator()
    schema = gen.generate_openapi()
    print(f"  OK: {schema['info']['title']}")
except Exception as e:
    print(f"  Skip: {e}")

try:
    from docs.changelog_generator import ChangelogGenerator
    print("\n[3/4] Generating changelog...")
    gen = ChangelogGenerator()
    gen.generate()
    print("  OK: CHANGELOG.md")
except Exception as e:
    print(f"  Skip: {e}")

try:
    from docs.readme_updater import ReadmeUpdater
    print("\n[4/4] Updating README...")
    gen = ReadmeUpdater()
    gen.update()
    print("  OK: README.md")
except Exception as e:
    print(f"  Skip: {e}")

print("\n" + "=" * 50)
print("Documentation generation complete!")
print("=" * 50)
