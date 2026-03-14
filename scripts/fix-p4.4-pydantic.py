#!/usr/bin/env python3
"""
Fix Pydantic validation errors by converting int classes to strings
"""

import re

# Read the file
with open('scripts/p4.4-inference-api.py', 'r') as f:
    content = f.read()

# Fix 1: Update ModelMetadataResponse to convert classes to strings
# Find the get_model_metadata function and fix the classes conversion
old_classes = "classes=classes"
new_classes = "classes=[str(c) for c in classes]"
content = content.replace(old_classes, new_classes)

# Fix 2: In predict() method, convert prediction to string
# Find where prediction is assigned and convert to string
old_predict = 'prediction = self.classes[prediction_idx]'
new_predict = 'prediction = str(self.classes[prediction_idx])'
content = content.replace(old_predict, new_predict)

# Fix 3: In predict_batch() method, convert prediction to string
old_predict_batch = 'prediction = self.classes[pred_idx]'
new_predict_batch = 'prediction = str(self.classes[pred_idx])'
content = content.replace(old_predict_batch, new_predict_batch)

# Fix 4: Convert class_probs keys to strings in predict()
old_class_probs = "class_name = self.classes[i]"
new_class_probs = "class_name = str(self.classes[i])"
content = content.replace(old_class_probs, new_class_probs)

# Write back
with open('scripts/p4.4-inference-api.py', 'w') as f:
    f.write(content)

print("✓ Fixed Pydantic validation errors")
print("✓ Integer classes converted to strings")
print("✓ Restart API server to apply changes")
