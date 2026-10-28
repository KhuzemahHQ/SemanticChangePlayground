import os
import sys
import time
import urllib.request
import zipfile

URL = "http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip"
TARGET_DIR = os.path.abspath("data/histwords/eng-fiction")
ZIP_PATH = os.path.abspath("data/cache/eng-fiction-all_sgns.zip")

def download_and_extract():
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Check if already extracted
    expected_file = os.path.join(TARGET_DIR, "sgns", "1990-w.npy")
    if os.path.exists(expected_file):
        print(f"HistWords dataset already extracted at {TARGET_DIR}")
        return True

    # Check if zip exists and is complete
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    current_size = os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 0

    if current_size < total_size:
        print(f"Downloading HistWords ({total_size / (1024*1024):.1f} MB)... Starting from {current_size / (1024*1024):.1f} MB")
        headers = {}
        mode = "wb"
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            mode = "ab"

        req = urllib.request.Request(URL, headers=headers)
        start_time = time.time()
        last_log = 0

        with urllib.request.urlopen(req, timeout=30) as resp, open(ZIP_PATH, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024) # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
                current_size += len(chunk)
                now = time.time()
                if now - last_log >= 10:
                    pct = (current_size / total_size) * 100 if total_size else 0
                    speed = (current_size / (1024*1024)) / (now - start_time + 0.001)
                    print(f"Progress: {current_size / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                    last_log = now

        print(f"Download complete: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/(1024*1024):.1f} MB)")

    print(f"Extracting to {TARGET_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(TARGET_DIR)
    print("Extraction complete!")
    return True

if __name__ == "__main__":
    try:
        download_and_extract()
    except Exception as e:
        print(f"Download/extraction error: {e}", file=sys.stderr)
        sys.exit(1)
