#!/usr/bin/env python3
import re

def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Check for 'with open' statement that might be causing the error
        # This targets the specific pattern around line 229
        if 'with open(self.model_path, "rb") as f:' in line:
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # If next line is not indented enough (less than 16 spaces usually for inside with)
                # We assume standard 4-space indentation per level
                # The 'with' is likely at 12 spaces, so content should be at 16
                if not next_line.startswith('                ') and next_line.strip():
                    # Force indentation
                    stripped = next_line.lstrip()
                    new_lines.append('                ' + stripped)
                    i += 1
                    continue
        i += 1
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Attempted fix on {file_path}")

if __name__ == '__main__':
    fix_indentation('scripts/p4.4-inference-api.py')
