from sentence_transformers import SentenceTransformer
import redis 
import unicodedata
import hashlib  
import json
import time
import math
import random
import threading
import numpy as np

import requests

# Assuming redis connection and current model version are available globally
r = redis.Redis() # Make sure to have redis-server running
current_model_version = "v1.0"

# Model object that can use a mock or a real Ollama model
class Model:
    def __init__(self, use_ollama=False, model_name="gemma:2b"):
        self.type = "generative"
        self.use_ollama = use_ollama
        self.model_name = model_name
        
    def infer(self, input_data):
        if self.use_ollama:
            try:
                # Call local Ollama API
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": input_data,
                        "stream": False
                    }
                )
                response.raise_for_status()
                output = response.json().get("response", "")
                
                # Note: Standard Ollama /api/generate doesn't return token probabilities by default.
                # For this test, we simulate generative log probs (e.g. log(prob) between -0.1 and -0.5)
                # In production with TGI or vLLM, you would parse actual logprobs here.
                mock_log_probs = [-0.1, -0.2, -0.05]
                return output, mock_log_probs
            except Exception as e:
                print(f"Ollama Inference Error: {e}")
                return "Error generating response.", [-10.0]
        else:
            # Mock returning output and raw scores / log probs
            time.sleep(1) # Simulate inference delay
            return f"Mock response for: {input_data[:10]}...", [-0.1, -0.2, -0.05]

# Change use_ollama to True to use Ollama
model = Model(use_ollama=False, model_name="gemma:2b")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# ─────────────────────────────────────────
# BACKGROUND REFRESH
# ─────────────────────────────────────────
def background_refresh(key, input_data, config):
    output, raw_scores = model.infer(input_data)

    if model.type == "classification":
        confidence = float(np.max(softmax(raw_scores)))
    else:
        if config.get("confidence_method") == "geometric_mean":
            confidence = math.exp(np.mean(raw_scores))
        else:
            confidence = min(raw_scores)

    if confidence >= config.get("cache_threshold", 0.9):
        serve_threshold = config.get("serve_threshold", 0.5)
        lambda_val = config.get("lambda", 0.01)
        
        t = -math.log(serve_threshold / confidence) / lambda_val
        ttl = int(t * 86400)

        entry = {
            "output": output,
            "confidence": confidence,
            "timestamp": time.time(),
            "hit_count": 0,
            "model_ver": current_model_version,
            "original_TTL": ttl
        }
        r.set(key, json.dumps(entry), ex=ttl)


# ─────────────────────────────────────────
# CACHE LOOKUP / INFERENCE
# ─────────────────────────────────────────
def handle_request(input_data, config):
    # Normalize and Hash
    normalized = unicodedata.normalize('NFC', input_data).encode('utf-8')
    key = hashlib.sha256(normalized).hexdigest()

    # BLOCK 1 — CACHE LOOKUP
    entry_bytes = r.get(key)
    entry = json.loads(entry_bytes) if entry_bytes else None

    if entry:
        # Version check
        if entry.get("model_ver") == current_model_version:
            # Stampede protection
            remaining_ttl = r.ttl(key)
            original_ttl = entry.get("original_TTL", 1)

            if remaining_ttl > 0 and remaining_ttl < (original_ttl * 0.10):
                if random.random() < 0.05:
                    threading.Thread(
                        target=background_refresh, 
                        args=(key, input_data, config)
                    ).start()

            # TTL math guarantees confidence is still valid.
            # Serve directly.
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            # (Optional: write back updated hit_count asynchronously if desired)
            
            return entry.get("output")

    # BLOCK 2 — RUN INFERENCE
    output, raw_scores = model.infer(input_data)

    # Confidence extraction
    if model.type == "classification":
        confidence = float(np.max(softmax(raw_scores)))
    elif model.type == "generative":
        if config.get("confidence_method") == "geometric_mean":
            confidence = math.exp(np.mean(raw_scores))
        elif config.get("confidence_method") == "min_token":
            confidence = min(raw_scores)
        else:
            confidence = min(raw_scores) # Default fallback

    # BLOCK 3 — CACHING DECISION
    if confidence >= config.get("cache_threshold", 0.9):
        serve_threshold = config.get("serve_threshold", 0.5)
        lambda_val = config.get("lambda", 0.01)
        
        t = -math.log(serve_threshold / confidence) / lambda_val
        ttl = int(t * 86400)

        new_entry = {
            "output": output,
            "confidence": confidence,
            "timestamp": time.time(),
            "hit_count": 0,
            "model_ver": current_model_version,
            "original_TTL": ttl
        }
        r.set(key, json.dumps(new_entry), ex=ttl)

    # Below cache_threshold — serve without storing.
    # Next identical request will re-run inference.
    return output
