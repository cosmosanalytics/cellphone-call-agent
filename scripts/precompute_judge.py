"""Precompute LLM-judge scores (tone, outcome) for the held-out replays, so the Streamlit Cloud app
can show them without running an LLM.  LLM_BASE_URL=http://localhost:8081/v1 python scripts/precompute_judge.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carrier import corpus, empirical, evaluation  # noqa: E402
from carrier.world import REALITY, RuleSet  # noqa: E402

out, cache = {}, {}
held = [c for c in corpus.load() if c["split"] == "heldout"]
for c in held:
    for rs in (RuleSet(), RuleSet(dict(REALITY.overrides))):
        rec = evaluation.replay_call(c, rs)
        key = rec["call_id"] + ("" if not rs.overrides else "+learned")
        txt = evaluation.transcript_text(rec)
        out[key] = cache[txt] if txt in cache else cache.setdefault(txt, evaluation.llm_judge(rec))
        print(key, out[key], flush=True)
evaluation.PRECOMPUTED_LLM.write_text(json.dumps(out, indent=1))
