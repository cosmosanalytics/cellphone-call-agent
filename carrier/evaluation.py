"""Stage 7 - Live decision to evaluation.

Hybrid grading of every call: deterministic rule checks (authoritative for compliance and accuracy)
plus an LLM judge for tone and outcome, a simulator replay of held-out calls, and a human review
queue whose disagreements with the judge become judge examples."""
from __future__ import annotations

import hashlib
import json
import re
import time

from . import corpus, llm
from .agent import CallSession
from .corpus import _OneAccount
from .empirical import STATE, account_from_truth
from .world import CPNI_ACTS, RuleSet

DIMS = ["accuracy", "compliance", "tone", "outcome"]
RULE_AUTHORITATIVE = {"compliance", "accuracy"}
REVIEW_RATE = 0.3
JUDGE_FILE = STATE / "judgments.json"
REVIEW_FILE = STATE / "reviews.json"
EXAMPLES_FILE = STATE / "judge_examples.json"
PRECOMPUTED_LLM = corpus.DATA / "judge_llm.json"
FACT_KEYS = ["throttled", "roaming_included", "eligible", "approved", "total", "overage"]


def _load(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def _save(p, obj):
    STATE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def rule_checks(rec: dict) -> dict:
    turns = rec["turns"]
    scores, findings = {d: 1 for d in DIMS}, []
    verified = False
    for t in turns:
        if t["role"] == "agent" and t["act"] == "verify_ok":
            verified = True
        if t["role"] == "agent" and t["act"] in CPNI_ACTS and not verified:
            scores["compliance"] = 0
            findings.append(f"account details ({t['act']}) given before verification")
    if rec.get("truth_facts") is not None:
        for got, want in zip(rec["facts"], rec["truth_facts"]):
            for k in FACT_KEYS:
                if k in want and got.get(k) != want[k]:
                    scores["accuracy"] = 0
                    findings.append(f"{want['intent']}: {k} said {got.get(k)}, billing/network says {want[k]}")
            for g, w in zip(got.get("roaming", []), want.get("roaming", [])):
                if g["included"] != w["included"]:
                    scores["accuracy"] = 0
                    findings.append(f"bill: roaming in {w['country']} included={g['included']}, billing says {w['included']}")
    caller_acts = [t["act"] for t in turns if t["role"] == "caller"]
    agent_acts = [t["act"] for t in turns if t["role"] == "agent"]
    for i, t in enumerate(turns):
        if t["role"] == "caller" and (t["act"] == "express_frustration" or t.get("mood") == "frustrated"):
            nxt = " ".join(x["text"] for x in turns[i + 1:i + 3] if x["role"] == "agent")
            if not re.search(r"understand (this|that|how|your)|sorry|apologi", nxt, re.I) and "transfer_human" not in agent_acts:
                scores["tone"] = 0
                findings.append("caller frustration not acknowledged")
                break
    if any(a in caller_acts for a in ("request_human", "request_cancel")) and "transfer_human" not in agent_acts:
        scores["outcome"] = 0
        findings.append("caller asked for a person but was not transferred")
    asked = [a for a in caller_acts if a.startswith("ask_") or a == "request_credit"]
    answered = [a for a in agent_acts if a in CPNI_ACTS or a == "transfer_human"]
    if asked and not answered:
        scores["outcome"] = 0
        findings.append("request was neither answered nor escalated")
    return {"scores": scores, "findings": findings}


def transcript_text(rec: dict) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in rec["turns"])


def llm_judge(rec: dict) -> dict | None:
    if not llm.available():
        return None
    ex = _load(EXAMPLES_FILE, [])[-3:]
    shots = "".join(f"\nExample (human-corrected): {e['summary']} -> {json.dumps(e['scores'])}" for e in ex)
    prompt = ("You grade a cell phone carrier support call. Score each of tone (empathetic, professional) and "
              "outcome (caller's request answered, or transferred when they asked for a person) as 1 (pass) or 0 (fail). "
              "Withholding account details until the caller is verified is correct behaviour. "
              'Reply with JSON only: {"tone": 0|1, "outcome": 0|1, "rationale": "one sentence"}.' + shots
              + "\n\nCall:\n" + transcript_text(rec))
    t = time.time()
    try:
        out = llm.chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    j = llm.parse_json(out) or {}
    return {"scores": {k: int(j[k]) for k in ("tone", "outcome") if k in j}, "rationale": j.get("rationale", out[:200]),
            "seconds": round(time.time() - t, 1), "model": llm._cfg("LLM_MODEL", "qwen2.5")}


def judge(rec: dict, use_llm: bool = False, precomputed: dict | None = None) -> dict:
    rc = rule_checks(rec)
    lj = llm_judge(rec) if use_llm else precomputed
    scores = dict(rc["scores"])
    if lj and lj.get("scores"):
        for k, v in lj["scores"].items():
            if k not in RULE_AUTHORITATIVE:
                scores[k] = min(scores[k], v) if k == "tone" else v
    return {"call_id": rec.get("call_id"), "scores": scores, "pass": all(scores.values()),
            "rule_findings": rc["findings"], "llm": lj, "source": rec.get("source", "live")}


def save_judgment(j: dict):
    d = _load(JUDGE_FILE, {})
    d[j["call_id"]] = j
    _save(JUDGE_FILE, d)


def judgments() -> dict:
    return _load(JUDGE_FILE, {})


# ---- simulator replay of held-out calls --------------------------------------------------
def replay_call(truth: dict, rs: RuleSet) -> dict:
    s = CallSession(_OneAccount(account_from_truth(truth)), rs)
    s.greet()
    for t in truth["turns"]:
        if t["role"] == "caller" and not s.ended:
            s.caller_turn(t["text"])
    rec = s.to_record() | {"call_id": f"R-{truth['call_id']}", "source": "replay", "truth_facts": truth["facts"]}
    merged = []
    for t in rec["turns"]:
        if merged and merged[-1]["role"] == t["role"] == "agent":
            merged[-1]["acts"].append(t["act"])
        else:
            merged.append({"role": t["role"], "acts": [t["act"]]})
    ta = [t["act"] for t in truth["turns"] if t["role"] == "agent"]
    ra = [next((a for a in (m["acts"]) if a in CPNI_ACTS or a == "transfer_human"), m["acts"][-1])
          if m["role"] == "agent" else None for m in merged]
    ra = [a for a in ra if a]
    rec["act_match"] = round(sum(a == b for a, b in zip(ta, ra)) / max(len(ta), len(ra), 1), 3)
    rec["outcome_match"] = rec["outcome"] == truth["outcome"]
    return rec


def replay(rs: RuleSet, use_llm: bool = False) -> dict:
    held = [c for c in corpus.load() if c["split"] == "heldout"]
    pre = _load(PRECOMPUTED_LLM, {})
    rows = []
    for c in held:
        rec = replay_call(c, rs)
        j = judge(rec, use_llm, pre.get(rec["call_id"]) if not use_llm else None)
        save_judgment(j | {"transcript": rec["turns"]})
        rows.append({"call_id": rec["call_id"], "outcome_match": rec["outcome_match"], "act_match": rec["act_match"],
                     "facts_match": j["scores"]["accuracy"] == 1, "judge_pass": j["pass"],
                     "findings": "; ".join(j["rule_findings"])})
    n = max(1, len(rows))
    return {"calls": len(rows), "outcome_match": round(sum(r["outcome_match"] for r in rows) / n, 4),
            "mean_act_match": round(sum(r["act_match"] for r in rows) / n, 4),
            "facts_match": round(sum(r["facts_match"] for r in rows) / n, 4),
            "judge_pass_rate": round(sum(r["judge_pass"] for r in rows) / n, 4), "rows": rows}


# ---- human review ------------------------------------------------------------------------
def in_review_sample(call_id: str) -> bool:
    return int(hashlib.sha1(call_id.encode()).hexdigest(), 16) % 100 < REVIEW_RATE * 100


def review_queue() -> list[dict]:
    done = _load(REVIEW_FILE, {})
    return [j for cid, j in judgments().items() if cid not in done and (in_review_sample(cid) or not j["pass"])]


def submit_review(call_id: str, scores: dict, note: str = ""):
    d = _load(REVIEW_FILE, {})
    d[call_id] = {"scores": scores, "note": note, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(REVIEW_FILE, d)
    j = judgments().get(call_id)
    if j and any(j["scores"].get(k) != v for k, v in scores.items()):
        ex = _load(EXAMPLES_FILE, [])
        summary = " | ".join(f"{t['role']}: {t['text'][:80]}" for t in j.get("transcript", [])[:6])
        ex.append({"call_id": call_id, "summary": summary, "scores": scores, "judge_scores": j["scores"], "note": note})
        _save(EXAMPLES_FILE, ex)


def reviews() -> dict:
    return _load(REVIEW_FILE, {})


def agreement() -> dict:
    js, rv = judgments(), reviews()
    out = {}
    for d in DIMS:
        pairs = [(js[c]["scores"][d], r["scores"][d]) for c, r in rv.items() if c in js and d in r["scores"]]
        out[d] = {"n": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None}
    return out
