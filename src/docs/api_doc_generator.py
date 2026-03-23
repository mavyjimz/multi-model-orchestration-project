#!/usr/bin/env python3
"""API Documentation Generator"""

import json
from pathlib import Path
from typing import Any


class APIDocGenerator:
    def __init__(self):
        self.output_dir = Path("docs/api")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_openapi(self) -> dict[str, Any]:
        schema = {
            "openapi": "3.0.0",
            "info": {"title": "MLOps API", "version": "1.0.2"},
            "paths": {
                "/health": {"get": {"summary": "Health check"}},
                "/models": {"get": {"summary": "List models"}},
                "/predict": {"post": {"summary": "Make prediction"}},
            },
        }

        json_path = self.output_dir / "openapi.json"
        with open(json_path, "w") as f:
            json.dump(schema, f, indent=2)

        return schema


if __name__ == "__main__":
    gen = APIDocGenerator()
    schema = gen.generate_openapi()
    print(f"OpenAPI schema generated: {schema['info']['title']}")
