#!/usr/bin/env python3
# Offline Mode Fallback
import json
import sys

def offline_predict(query):
    # Simple rule-based fallback when ML model is unavailable
    query = query.lower()
    if "hello" in query or "hi" in query:
        return {"intent": "greeting", "confidence": 0.5, "mode": "offline"}
    elif "help" in query:
        return {"intent": "support", "confidence": 0.5, "mode": "offline"}
    else:
        return {"intent": "unknown", "confidence": 0.0, "mode": "offline"}

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "hello"
    print(json.dumps(offline_predict(query)))
