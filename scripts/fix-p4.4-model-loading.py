#!/usr/bin/env python3
"""
fix-p4.4-model-loading.py
Comprehensive fix for model loading in p4.4-inference-api.py
"""

import re
from pathlib import Path

api_path = Path('scripts/p4.4-inference-api.py')

with open(api_path, 'r') as f:
    content = f.read()

# Fix 1: Extract model from dictionary IMMEDIATELY after loading (before any attribute access)
old_pattern = r"""with open\(self\.model_path, 'rb'\) as f:
            self\.model = pickle\.load\(f\)
        logger\.info\(f"✓ Model loaded: \{self\.model_path\}"\)
        logger\.info\(f"  Model type: \{type\(self\.model\)\}"\)"""

new_code = """with open(self.model_path, 'rb') as f:
            loaded = pickle.load(f)
        
        # Extract model from dictionary wrapper if present
        if isinstance(loaded, dict):
            self.model = loaded.get('model', loaded)
            logger.info("  Model extracted from dictionary wrapper")
        else:
            self.model = loaded
        
        logger.info(f"✓ Model loaded: {self.model_path}")
        logger.info(f"  Model type: {type(self.model)}")"""

content = re.sub(
    r"""with open\(self\.model_path, 'rb'\) as f:
            self\.model = pickle\.load\(f\)
        logger\.info\(f"✓ Model loaded: \{self\.model_path\}"\)
        logger\.info\(f"  Model type: \{type\(self\.model\)\}"\)""",
    new_code,
    content
)

# Fix 2: Safe access to classes_ attribute
content = content.replace(
    "logger.info(f\"  Model classes: {self.model.classes_}\")",
    "logger.info(f\"  Model classes: {getattr(self.model, 'classes_', 'N/A')}\")"
)

# Fix 3: Safe access in load method where classes_ is used
content = re.sub(
    r"self\.classes = self\.model\.classes_\.tolist\(\)",
    """# Extract classes safely
        if isinstance(self.model, dict):
            self.model = self.model.get('model', self.model)
        self.classes = self.model.classes_.tolist() if hasattr(self.model, 'classes_') else []""",
    content
)

with open(api_path, 'w') as f:
    f.write(content)

print("✓ Fixed p4.4-inference-api.py model loading")
print("✓ Model extraction now happens BEFORE attribute access")
