import time
from main import handle_request, r

# Configuration used in algorithm.md
config = {
    "confidence_method": "geometric_mean", # or "min_token"
    "cache_threshold": 0.8,   # High threshold to cache
    "serve_threshold": 0.5,   # Low threshold to serve
    "lambda": 0.01            # Decay parameter
}

def run_test():
    prompt = "Explain quantum computing in one sentence."
    
    print("--- 1. First Request (Cache Miss) ---")
    start = time.time()
    response = handle_request(prompt, config)
    print(f"Time: {time.time() - start:.2f}s | Response: {response}")
    
    print("\n--- 2. Second Request (Cache Hit) ---")
    start = time.time()
    response2 = handle_request(prompt, config)
    print(f"Time: {time.time() - start:.2f}s | Response: {response2}")
    
    print("\n--- 3. Let's wait for TTL to drop to simulate background refresh ---")
    # You can configure the lambda/thresholds in main.py to make the TTL very short for testing.
    # We will trigger the background refresh by waiting.
    # In a real test, you'd lower the generated TTL to a few seconds to observe it.

class MockRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        if key in self.store:
            val, exp = self.store[key]
            if time.time() > exp:
                del self.store[key]
                return None
            return val
        return None

    def set(self, key, value, ex=None):
        # ex is time-to-live in seconds
        self.store[key] = (value, time.time() + ex if ex else float('inf'))

    def ttl(self, key):
        if key in self.store:
            val, exp = self.store[key]
            rem = exp - time.time()
            if rem > 0:
                return int(rem)
            else:
                del self.store[key]
                return -2
        return -2

    def flushdb(self):
        self.store = {}

if __name__ == "__main__":
    import main
    
    # 💡 Monkey-patch main.r with our MockRedis so we don't need a real Redis server!
    main.r = MockRedis()
    
    print("Running test with MockRedis (no real Redis server needed)")
    main.r.flushdb()
    run_test()

