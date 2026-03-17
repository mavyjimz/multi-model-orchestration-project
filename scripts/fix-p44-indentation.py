#!/usr/bin/env python3
"""
Fix indentation error in p4.4-inference-api.py
"""

def fix_indentation(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for the problematic pattern: with open... followed by non-indented code
        if 'with open(self.model_path' in line and i + 1 < len(lines):
            fixed_lines.append(line)
            i += 1
            
            # Next line should be 'loaded = pickle.load(f)' with proper indent
            if i < len(lines) and 'loaded = pickle.load(f)' in lines[i]:
                # Ensure proper indentation (12 spaces for inside with block)
                fixed_lines.append('            loaded = pickle.load(f)\n')
                i += 1
                
                # Add the wrapper extraction code with proper indentation
                wrapper_code = [
                    '\n',
                    '            # Extract model from dictionary wrapper\n',
                    '            if isinstance(loaded, dict) and \'model\' in loaded:\n',
                    '                self.model = loaded[\'model\']\n',
                    '                self.class_mapper = loaded.get(\'class_mapper\')\n',
                    '            else:\n',
                    '                self.model = loaded\n',
                    '                self.class_mapper = None\n',
                    '\n',
                    '            # Safe access to classes attribute\n',
                    '            if hasattr(self.model, \'classes_\'):\n',
                    '                self.classes = self.model.classes_.tolist()\n',
                    '            elif hasattr(self.model, \'class_mapper\') and self.model.class_mapper:\n',
                    '                self.classes = list(self.model.class_mapper.idx_to_label.values())\n',
                    '            else:\n',
                    '                self.classes = []\n',
                ]
                fixed_lines.extend(wrapper_code)
                continue
        
        fixed_lines.append(line)
        i += 1
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == '__main__':
    fix_indentation('scripts/p4.4-inference-api.py')
