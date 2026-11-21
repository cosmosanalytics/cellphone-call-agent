"""Offline data build (run on a machine with the models): corpus + audio, then batch transcripts.

    python scripts/build_data.py corpus
    python scripts/build_data.py batch [workers]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.argv[1] == "corpus":
    from carrier import corpus
    corpus.build(progress=lambda i, n: print(f"{i}/{n}", flush=True) if i % 10 == 0 else None)
elif sys.argv[1] == "batch":
    from carrier import batch
    print(batch.run_all(workers=int(sys.argv[2]) if len(sys.argv) > 2 else 4))
