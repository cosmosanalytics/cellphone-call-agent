"""Stage 2 - Batch: voice to script.

Whole-call pass: VAD regions -> Whisper per region, prompted with carrier vocabulary and the running
call context -> speaker embeddings clustered over the whole call -> roles from world-model cues ->
term correction -> PII redaction (phone, PIN, names) -> dialogue acts -> de-identified facts.
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import nlu, speech
from .corpus import AUDIO, CORPUS, DATA, WAV_CACHE
from .textnorm import normalize_for_wer
from .world import BookOfBusiness

TRANSCRIPTS = DATA / "transcripts"
SUMMARY = DATA / "batch_summary.json"
_bob = BookOfBusiness()


def load_call_audio(cid: str) -> np.ndarray:
    import soundfile as sf
    p = WAV_CACHE / f"{cid}.wav"
    a, sr = sf.read(p if p.exists() else AUDIO / f"{cid}.ogg", dtype="float32")
    return speech.to16k(a, sr)


def _cluster2(E: np.ndarray) -> np.ndarray:
    """Average-linkage agglomerative clustering into two speakers (cosine distance)."""
    from sklearn.cluster import AgglomerativeClustering
    if len(E) < 2:
        return np.zeros(len(E), int)
    return AgglomerativeClustering(2, metric="cosine", linkage="average").fit_predict(E)


def process_audio(audio: np.ndarray, call_id: str, model: str | None = None, neural: bool | None = None) -> dict:
    model = model or speech.BATCH_MODEL
    utts, context = [], ""
    for a, b in speech.vad_regions(audio):
        chunk = audio[int(a * 16000):int(b * 16000)]
        text = speech.transcribe(chunk, model, prompt=nlu.stt_prompt() + " " + context[-200:])
        if text:
            utts.append({"start": a, "end": b, "text_raw": text, "emb": speech.speaker_embedding(chunk, neural)})
            context += " " + text
    labels = _cluster2(np.array([u.pop("emb") for u in utts])) if utts else np.zeros(0, int)
    cue = {0: 0.0, 1: 0.0}
    for u, lab in zip(utts, labels):
        cue[int(lab)] += nlu.role_cue_score(u["text_raw"])
    if utts:
        cue[int(labels[0])] += 1.5        # flow prior: the agent answers the phone
    agent_cluster = 0 if cue[0] >= cue[1] else 1
    for u, lab in zip(utts, labels):
        u["role"] = "agent" if lab == agent_cluster else "caller"
        u["role_textonly"] = "agent" if nlu.role_cue_score(u["text_raw"]) > 0 else "caller"
    turns: list[dict] = []
    for u in utts:
        if turns and turns[-1]["role"] == u["role"] and u["start"] - turns[-1]["end"] < 1.2:
            turns[-1]["end"] = u["end"]
            turns[-1]["text_raw"] += " " + u["text_raw"]
        else:
            turns.append(dict(u))
    fixes, redactions = [], []
    for t in turns:
        t["text"], f = nlu.correct_terms(t["text_raw"])
        fixes += f
        t["act"] = nlu.classify_act(t["role"], t["text"])
    phone = next((nlu.find_phone(t["text"]) for t in turns if t["role"] == "caller" and nlu.find_phone(t["text"])), None)
    acct = _bob.by_phone(phone) if phone else None
    names = [acct.first, acct.last] if acct else _names_from_agent(turns)
    for t in turns:
        t["text_redacted"], red = nlu.redact(t["text"], names, identity_turn=t["act"] == "give_identity")
        redactions += red
    return {"call_id": call_id, "model": model, "turns": turns, "term_fixes": fixes,
            "redactions": [{"type": r["type"]} for r in redactions], "features": extract_features(turns)}


def _names_from_agent(turns: list[dict]) -> list[str]:
    """When the phone number was misheard, the agent's verification line still names the caller."""
    return sorted({m.group(1) for t in turns if t["role"] == "agent"
                   for m in re.finditer(r"[Tt]hank you,? ([A-Z][a-z]+)[.,!]? [Yy]ou['’]?re verified", t["text"])})


def extract_features(turns: list[dict]) -> dict:
    """De-identified facts for the empirical layer, read from what the agent actually said."""
    agent = [t for t in turns if t["role"] == "agent"]
    acts = [t["act"] for t in turns]
    f: dict = {"plan_id": nlu.find_plan(" ".join(t["text"] for t in agent)),
               "intent": next((a for a in acts if a.startswith("ask_") or a == "request_credit"), None),
               "verified": "verify_ok" in acts or any(re.search(r"you'?re verified", t["text"], re.I) for t in agent),
               "escalated": "transfer_human" in acts, "roaming": []}
    for t in agent:
        x = t["text"]
        if re.search(r"used \d+ ?GB", x) and not re.search(r"of your \d+ ?GB", x) and f["plan_id"] != "BAS":
            f["used"] = nlu.find_gb(x)
            f["throttled"] = bool(re.search(r"speeds are reduced|past your priority data", x, re.I)) if \
                re.search(r"speeds are reduced|past your priority data|full speed", x, re.I) else None
        for m in re.finditer(r"(?i:roaming in) ([A-Z][a-z]+(?: [A-Z][a-z]+)?) (is included|isn[’']?t included|is not included)", x):
            c = nlu.find_country(m.group(1))
            if c:
                f["roaming"].append({"country": c, "zone": nlu.COUNTRIES[c], "included": m.group(2) == "is included", "src": "quote"})
        for m in re.finditer(r"plus (?i:roaming in) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", x):
            c = nlu.find_country(m.group(1))
            if c:
                f["roaming"].append({"country": c, "zone": nlu.COUNTRIES[c], "included": False, "src": "bill"})
        for m in re.finditer(r"(?i:usage in) ([A-Z][a-z]+(?: [A-Z][a-z]+)?) was included", x):
            c = nlu.find_country(m.group(1))
            if c:
                f["roaming"].append({"country": c, "zone": nlu.COUNTRIES[c], "included": True, "src": "bill"})
        if re.search(r"eligible to upgrade today", x):
            f["upgrade_eligible"] = True
        elif re.search(r"not eligible to upgrade|can'?t upgrade", x):
            f["upgrade_eligible"] = False
        m = re.search(r"has (\d+) of 24 installments", x)
        if m:
            f["payments_made"] = int(m.group(1))
        if re.search(r"i'?ve applied .*courtesy credit", x, re.I):
            f["credit_approved"] = True
        elif re.search(r"can'?t credit|above what i can credit", x, re.I):
            f["credit_approved"] = False
    return f


# ---- evaluation ---------------------------------------------------------------
def evaluate(res: dict, truth: dict) -> dict:
    ref = normalize_for_wer(" ".join(t["text"] for t in truth["turns"]))
    hyp_raw = normalize_for_wer(" ".join(t["text_raw"] for t in res["turns"]))
    hyp = normalize_for_wer(" ".join(t["text"] for t in res["turns"]))
    from jiwer import wer
    agree = agree_txt = total = 0.0
    for h in res["turns"]:
        for g in truth["turns"]:
            ov = max(0.0, min(h["end"], g["end"]) - max(h["start"], g["start"]))
            total += ov
            agree += ov * (h["role"] == g["role"])
            agree_txt += ov * (h.get("role_textonly", h["role"]) == g["role"])
    ref_n = f" {ref} "
    terms = [x for x in nlu.LEXICON if f" {normalize_for_wer(x)} " in ref_n]
    hit = sum(f" {normalize_for_wer(x)} " in f" {hyp} " for x in terms)
    hit_raw = sum(f" {normalize_for_wer(x)} " in f" {hyp_raw} " for x in terms)
    red = normalize_for_wer(" ".join(t["text_redacted"] for t in res["turns"])).split()
    pii = [truth["phone"].replace("-", ""), truth["pin"], truth["names"][0].lower()]
    leaked = sum(p in red for p in pii)
    a_truth, a_hyp = [t["act"] for t in truth["turns"]], [t["act"] for t in res["turns"]]
    act_acc = float(np.mean([a == b for a, b in zip(a_truth, a_hyp)])) if len(a_truth) == len(a_hyp) else None
    return {"wer": round(wer(ref, hyp), 4), "wer_raw": round(wer(ref, hyp_raw), 4),
            "role_acc": round(agree / total, 4) if total else 0.0,
            "role_acc_textonly": round(agree_txt / total, 4) if total else 0.0,
            "terms_total": len(terms), "terms_hit": int(hit), "terms_hit_raw": int(hit_raw),
            "pii_total": len(pii), "pii_leaked": int(leaked),
            "turns_truth": len(a_truth), "turns_hyp": len(a_hyp),
            "act_acc": None if act_acc is None else round(act_acc, 4)}


def run_one(cid: str, model: str | None = None, neural: bool | None = None, save: bool = True) -> dict:
    truth = json.loads((CORPUS / f"{cid}.json").read_text())
    res = process_audio(load_call_audio(cid), cid, model, neural)
    res["split"], res["source"] = truth["split"], "batch"
    res["eval"] = evaluate(res, truth)
    if save:
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS / f"{cid}.json").write_text(json.dumps(res, indent=1))
    return res


def refresh(res: dict) -> dict:
    """Re-derive acts, redaction, facts and metrics from a saved transcript's text (no re-transcription)."""
    turns = res["turns"]
    for t in turns:
        t["act"] = nlu.classify_act(t["role"], t["text"])
    phone = next((nlu.find_phone(t["text"]) for t in turns if t["role"] == "caller" and nlu.find_phone(t["text"])), None)
    acct = _bob.by_phone(phone) if phone else None
    reds = []
    for t in turns:
        t["text_redacted"], red = nlu.redact(t["text"], [acct.first, acct.last] if acct else _names_from_agent(turns),
                                             identity_turn=t["act"] == "give_identity")
        reds += red
    res["redactions"] = [{"type": r["type"]} for r in reds]
    res["features"] = extract_features(turns)
    res["eval"] = evaluate(res, json.loads((CORPUS / f"{res['call_id']}.json").read_text()))
    return res


def refresh_all() -> dict:
    evals = []
    for p in sorted(TRANSCRIPTS.glob("C*.json")):
        r = refresh(json.loads(p.read_text()))
        p.write_text(json.dumps(r, indent=1))
        evals.append(r["eval"])
    s = summarize(evals)
    SUMMARY.write_text(json.dumps(s, indent=1))
    return s


def _worker(cid):
    return run_one(cid)["eval"]


def run_all(workers: int = 4) -> dict:
    import os
    from multiprocessing import Pool
    os.environ.setdefault("STT_THREADS", str(max(1, (os.cpu_count() or 4) // workers)))
    ids = sorted(p.stem for p in CORPUS.glob("C*.json"))
    with Pool(workers) as pool:
        evals = []
        for i, e in enumerate(pool.imap(_worker, ids)):
            evals.append(e)
            print(f"{i + 1}/{len(ids)} wer={e['wer']} roles={e['role_acc']}", flush=True)
    s = summarize(evals)
    SUMMARY.write_text(json.dumps(s, indent=1))
    return s


def summarize(evals: list[dict]) -> dict:
    if not evals:
        return {}
    acts = [e["act_acc"] for e in evals if e["act_acc"] is not None]
    tt = sum(e["terms_total"] for e in evals) or 1
    pt = sum(e["pii_total"] for e in evals) or 1
    mean = lambda k: round(float(np.mean([e[k] for e in evals])), 4)  # noqa: E731
    return {"calls": len(evals), "wer": mean("wer"), "wer_raw": mean("wer_raw"), "role_acc": mean("role_acc"),
            "role_acc_textonly": mean("role_acc_textonly"),
            "term_recall": round(sum(e["terms_hit"] for e in evals) / tt, 4),
            "term_recall_raw": round(sum(e["terms_hit_raw"] for e in evals) / tt, 4),
            "redaction_recall": round(1 - sum(e["pii_leaked"] for e in evals) / pt, 4),
            "turn_count_match": round(float(np.mean([e["turns_truth"] == e["turns_hyp"] for e in evals])), 4),
            "act_acc": round(float(np.mean(acts)), 4) if acts else None}


def load_transcripts() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(TRANSCRIPTS.glob("C*.json"))]
