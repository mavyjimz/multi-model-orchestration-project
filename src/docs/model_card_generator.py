#!/usr/bin/env python3
"""Model Card Generator"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ModelCardGenerator:
    def __init__(self):
        self.output_dir = Path("docs/model_cards")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_card(self, model_name: str, model_version: str) -> Dict[str, Any]:
        card = {
            "model_name": model_name,
            "model_version": model_version,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_type": "SGDClassifier",
            "performance": {
                "accuracy": 0.7169,
                "latency_p95_ms": 21.75
            },
            "training_data": {
                "samples": 3341,
                "features": 5000
            },
            "limitations": [
                "Limited to 41 intent classes",
                "Performance may vary on new domains"
            ]
        }
        
        # Save as JSON
        json_path = self.output_dir / f"{model_name}_v{model_version}.json"
        with open(json_path, 'w') as f:
            json.dump(card, f, indent=2)
        
        # Save as Markdown
        md_path = self.output_dir / f"{model_name}_v{model_version}.md"
        with open(md_path, 'w') as f:
            f.write(f"# Model Card: {model_name}\n\n")
            f.write(f"**Version**: {model_version}\n\n")
            f.write(f"## Performance\n\n")
            f.write(f"- Accuracy: {card['performance']['accuracy']}\n")
            f.write(f"- P95 Latency: {card['performance']['latency_p95_ms']}ms\n")
        
        return card

if __name__ == "__main__":
    gen = ModelCardGenerator()
    card = gen.generate_card("intent-classifier-sgd", "v1.0.2")
    print(f"Model card generated: {card['model_name']} v{card['model_version']}")
