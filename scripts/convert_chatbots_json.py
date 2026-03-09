#!/usr/bin/env python3
"""Convert chatbots Intent.json to CSV format."""
import json
import csv
from pathlib import Path

def convert_json_to_csv(json_path: str, output_path: str):
    """Convert JSON intents to CSV (user_input,intent)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for intent_obj in data['intents']:
        intent_name = intent_obj['intent']
        for text in intent_obj.get('text', []):
            rows.append({'user_input': text, 'intent': intent_name})
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['user_input', 'intent'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Converted {len(rows)} samples to {output_path}")
    return len(rows)

if __name__ == "__main__":
    json_file = "input-data/raw/chatbots-temp/Intent.json"
    output_file = "input-data/raw/chatbots-temp/chatbots_intents.csv"
    convert_json_to_csv(json_file, output_file)
