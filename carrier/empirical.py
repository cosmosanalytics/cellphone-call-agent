"""Stage 3 - Batch empirical learning.

From de-identified batch transcripts: a call-flow model (act transitions, outcomes per intent) and
policy contradictions - places where what billing / the network actually did (as said on past calls)
disagrees with the published rules. Contradictions become proposed rule corrections that a human
accepts or rejects; accepted ones are used on every live call. Retraining is suggested once enough
new calls have arrived. Evaluated on held-out calls (rules only vs rules + learned corrections)."""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from datetime import date

from . import corpus
from .batch import load_transcripts
from .world import (PLANS, PUBLISHED, Account, RuleSet, RulesEngine, Trip, key_label)

STATE = corpus.DATA / "state"
STATE_FILE = STATE / "empirical.json"
LIVE_FILE = STATE / "live_calls.jsonl"
MIN_NEW_CALLS = 20
MIN_SUPPORT = 3
MIN_RATE = 0.8


def _read_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"reviews": {}}


def _write_state(s: dict):
    STATE.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=1))


def live_calls() -> list[dict]:
    if not LIVE_FILE.exists():
        return []
    return [json.loads(x) for x in LIVE_FILE.read_text().splitlines() if x.strip()]


def save_live_call(rec: dict):
    STATE.mkdir(parents=True, exist_ok=True)
    with LIVE_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- flow model ---------------------------------------------------------------
class FlowModel:
    def __init__(self, transitions: dict | None = None, outcomes: dict | None = None, n: int = 0):
        self.transitions = transitions or {}
        self.outcomes = outcomes or {}
        self.n = n

    @classmethod
    def fit(cls, calls: list[list[dict]], outcomes: list[tuple[str | None, bool]]) -> "FlowModel":
        tr: dict = defaultdict(Counter)
        for turns in calls:
            seq = ["START"] + [f"{t['role']}:{t['act']}" for t in turns] + ["END"]
            for a, b in zip(seq, seq[1:]):
                tr[a][b] += 1
        oc: dict = defaultdict(Counter)
        for intent, esc in outcomes:
            oc[intent or "none"]["escalated" if esc else "resolved"] += 1
        return cls({k: dict(v) for k, v in tr.items()}, {k: dict(v) for k, v in oc.items()}, len(calls))

    def probs(self, state: str) -> dict:
        c = self.transitions.get(state, {})
        n = sum(c.values()) or 1
        return {k: v / n for k, v in sorted(c.items(), key=lambda x: -x[1])}

    def to_json(self):
        return {"transitions": self.transitions, "outcomes": self.outcomes, "n": self.n}


# ---- contradictions ---------------------------------------------------------------
def _obs(feats: list[dict]) -> tuple[list, list, list]:
    roam, data, upg = [], [], []
    for cid, f in feats:
        p = f.get("plan_id")
        if not p:
            continue
        for r in f.get("roaming", []):
            roam.append((cid, p, r["zone"], r["included"]))
        if f.get("used") is not None and f.get("throttled") is not None:
            data.append((cid, p, f["used"], f["throttled"]))
        if f.get("upgrade_eligible") is not None and f.get("payments_made") is not None:
            upg.append((cid, p, f["payments_made"], f["upgrade_eligible"]))
    return roam, data, upg


def find_contradictions(feats: list[tuple[str, dict]], rs: RuleSet = PUBLISHED) -> list[dict]:
    eng = RulesEngine(rs)
    roam, data, upg = _obs(feats)
    out = []
    groups = defaultdict(list)
    for cid, p, z, inc in roam:
        groups[(p, z)].append((cid, inc))
    for (p, z), obs in sorted(groups.items()):
        rule, _ = eng.roaming_included(p, z)
        bad = [c for c, inc in obs if inc != rule]
        if len(bad) >= MIN_SUPPORT and len(bad) / len(obs) >= MIN_RATE:
            out.append({"key": ["roaming_included", p, z], "current": rule, "proposed": not rule,
                        "support": len(obs), "contradicting": len(bad), "rate": round(len(bad) / len(obs), 3),
                        "examples": bad[:8], "evidence": "roaming answers on past calls (quotes and bills)"})
    by_plan = defaultdict(list)
    for cid, p, used, thr in data:
        by_plan[p].append((cid, used, thr))
    for p, obs in sorted(by_plan.items()):
        rule, _ = eng.throttle_gb(p)
        pred = lambda u, t=rule: t is not None and u >= t  # noqa: E731
        bad = [(c, u, t) for c, u, t in obs if pred(u) != t]
        if len(bad) < MIN_SUPPORT:
            continue
        lo = max([u for _, u, t in obs if not t], default=-1)
        hi = min([u for _, u, t in obs if t], default=None)
        if hi is None:
            continue
        cand = next((x for x in range(5, 1000, 5) if lo < x <= hi), hi)
        ok = sum((u >= cand) == t for _, u, t in obs)
        if ok / len(obs) >= MIN_RATE:
            out.append({"key": ["throttle_gb", p], "current": rule, "proposed": cand,
                        "support": len(obs), "contradicting": len(bad), "rate": round(len(bad) / len(obs), 3),
                        "examples": [c for c, _, _ in bad][:8],
                        "evidence": f"data answers: highest full-speed usage {lo} GB, lowest slowed usage {hi} GB"})
    by_plan = defaultdict(list)
    for cid, p, n, el in upg:
        by_plan[p].append((cid, n, el))
    for p, obs in sorted(by_plan.items()):
        rule = PLANS[p].upgrade_after if ("upgrade_after", p) not in rs.overrides else rs.overrides[("upgrade_after", p)]
        bad = [c for c, n, el in obs if (n >= rule) != el]
        if len(bad) >= MIN_SUPPORT:
            lo = max([n for _, n, el in obs if not el], default=-1)
            hi = min([n for _, n, el in obs if el], default=lo + 1)
            out.append({"key": ["upgrade_after", p], "current": rule, "proposed": max(lo + 1, min(hi, 24)),
                        "support": len(obs), "contradicting": len(bad), "rate": round(len(bad) / len(obs), 3),
                        "examples": bad[:8], "evidence": "upgrade answers on past calls"})
    for c in out:
        c["label"] = key_label(tuple(c["key"]))
        c["id"] = "|".join(map(str, c["key"]))
    return out


# ---- train / status / review ---------------------------------------------------------
def train() -> dict:
    tx = [t for t in load_transcripts() if t.get("split") == "train"]
    live = live_calls()
    flow = FlowModel.fit([t["turns"] for t in tx] + [r["turns"] for r in live],
                         [(t["features"].get("intent"), t["features"].get("escalated")) for t in tx]
                         + [((r["facts"] or [{}])[0].get("intent"), r["outcome"] == "escalated") for r in live])
    contra = find_contradictions([(t["call_id"], t["features"]) for t in tx])
    s = _read_state()
    s.update({"flow": flow.to_json(), "contradictions": contra, "trained_calls": len(tx), "trained_live": len(live),
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_state(s)
    return s


def status() -> dict:
    s = _read_state()
    if "flow" not in s:
        s = train()
    new = len(live_calls()) - s.get("trained_live", 0)
    return s | {"new_calls": new, "retrain_due": new >= MIN_NEW_CALLS, "min_new_calls": MIN_NEW_CALLS}


def review(cid: str, decision: str, reviewer: str = "reviewer"):
    s = _read_state()
    s.setdefault("reviews", {})[cid] = {"decision": decision, "by": reviewer, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_state(s)


def learned_ruleset() -> RuleSet:
    s = status()
    rs = RuleSet()
    for c in s.get("contradictions", []):
        if s.get("reviews", {}).get(c["id"], {}).get("decision") == "accept":
            k = tuple(c["key"])
            rs.overrides[k] = c["proposed"]
            rs.provenance[k] = f"learned from {c['contradicting']}/{c['support']} past calls, accepted by reviewer"
    return rs


# ---- held-out evaluation -----------------------------------------------------------------
def account_from_truth(t: dict) -> Account:
    s = t["snapshot"]
    return Account(t["account_id"], t["names"][0], t["names"][1], t["phone"], t["pin"], t["plan_id"], s["device"],
                   s["payments_made"], s["data_used_gb"], [Trip(c, d) for c, d in s["trips"]], s["past_due"], "active",
                   date.fromisoformat(s["last_courtesy_credit"]) if s["last_courtesy_credit"] else None)


def predict(eng: RulesEngine, acct: Account, fact: dict) -> list[tuple[str, object, object]]:
    """(question, predicted, actual) pairs for one ground-truth answer."""
    out = []
    i = fact["intent"]
    if i == "ask_data":
        out.append(("throttled", eng.data(acct).values["throttled"], fact["throttled"]))
    elif i == "ask_roaming":
        out.append(("roaming included", eng.roaming(acct, fact["country"], None).values["included"], fact["roaming_included"]))
    elif i == "ask_bill":
        for r in fact.get("roaming", []):
            out.append(("roaming included", eng.roaming(acct, r["country"], None).values["included"], r["included"]))
        out.append(("bill total", eng.bill(acct).values["total"], fact["total"]))
    elif i == "ask_upgrade":
        out.append(("upgrade eligible", eng.upgrade(acct).ok, fact["eligible"]))
    elif i == "request_credit":
        out.append(("credit approved", eng.credit(acct, "roaming" if acct.trips else "overage").ok, fact["approved"]))
    return out


def evaluate_heldout(rs: RuleSet | None = None) -> dict:
    held = [c for c in corpus.load() if c["split"] == "heldout"]
    res = {}
    for name, ruleset in (("rules_only", PUBLISHED), ("rules_plus_learned", rs or learned_ruleset())):
        eng = RulesEngine(ruleset)
        rows = []
        for c in held:
            acct = account_from_truth(c)
            for f in c["facts"]:
                for q, p, a in predict(eng, acct, f):
                    rows.append({"call_id": c["call_id"], "question": q, "predicted": p, "actual": a, "correct": p == a})
        res[name] = {"answer_acc": round(sum(r["correct"] for r in rows) / max(1, len(rows)), 4),
                     "n": len(rows), "errors": [r for r in rows if not r["correct"]]}
    res["calls"] = len(held)
    return res


# ---- simulator ---------------------------------------------------------------------------
class Simulator:
    """Samples caller behaviour from the flow model to stress-test the agent."""

    def __init__(self, flow: FlowModel, seed: int = 0):
        self.flow, self.rng = flow, random.Random(seed)

    def sample_acts(self, max_len: int = 16) -> list[str]:
        state, seq = "START", []
        for _ in range(max_len):
            p = self.flow.probs(state)
            if not p:
                break
            state = self.rng.choices(list(p), list(p.values()))[0]
            if state == "END":
                break
            seq.append(state)
        return seq


def simulator_check(n: int = 300) -> dict:
    """Compare simulated call flows with real held-out calls (act-bigram total variation distance)."""
    flow = FlowModel(**status()["flow"])
    sim = Simulator(flow)
    real = [t["turns"] for t in load_transcripts() if t.get("split") == "heldout"]

    def bigrams(seqs):
        c = Counter()
        for s in seqs:
            s = ["START"] + s + ["END"]
            c.update(zip(s, s[1:]))
        n = sum(c.values()) or 1
        return {k: v / n for k, v in c.items()}
    sim_b = bigrams([sim.sample_acts() for _ in range(n)])
    real_b = bigrams([[f"{t['role']}:{t['act']}" for t in r] for r in real])
    tvd = 0.5 * sum(abs(sim_b.get(k, 0) - real_b.get(k, 0)) for k in set(sim_b) | set(real_b))
    lens_sim = [len(sim.sample_acts()) for _ in range(n)]
    return {"bigram_tvd": round(tvd, 4), "mean_turns_sim": round(sum(lens_sim) / n, 2),
            "mean_turns_real": round(sum(len(r) for r in real) / max(1, len(real)), 2), "real_calls": len(real)}
