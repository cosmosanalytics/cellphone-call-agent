"""Northstar Wireless AI call agent - all seven touchpoints in one Streamlit app."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import pandas as pd
import streamlit as st

from carrier import batch, corpus, empirical, evaluation, llm, nlu, speech
from carrier.agent import CallSession
from carrier.world import (BILL_TRANSITIONS, CARRIER, COUNTRIES, CREDIT_TRANSITIONS, DEVICES, PLANS, ROAMING_DAY_RATE,
                           ZONE_NAMES, BookOfBusiness, RulesEngine, fmt_money)

st.set_page_config(page_title=f"{CARRIER} AI call agent", layout="wide")
# sha256 of the app password; override with APP_PASSWORD_SHA256 in Streamlit secrets
DEFAULT_PW_SHA256 = "6d8749c4fe00b757c1bc1c376a99f31d1bd38b422d44f5b6aa2d6a6c5e975cba"


def _pw_hash() -> str:
    try:
        return st.secrets.get("APP_PASSWORD_SHA256", DEFAULT_PW_SHA256)
    except Exception:
        return DEFAULT_PW_SHA256


def gate():
    if st.session_state.get("authed"):
        return
    st.title(f"{CARRIER} AI call agent")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(hashlib.sha256(pw.encode()).hexdigest(), _pw_hash()):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password")
    st.stop()


gate()


@st.cache_resource
def bob() -> BookOfBusiness:
    return BookOfBusiness()


@st.cache_data
def corpus_calls():
    return {c["call_id"]: c for c in corpus.load()}


@st.cache_data
def transcripts():
    return {t["call_id"]: t for t in batch.load_transcripts()}


def ruleset(learned: bool):
    return empirical.learned_ruleset() if learned else None


PAGES = ["Overview", "1 World model", "2 Batch: voice to script", "3 Empirical learning",
         "4-6 Live call", "7 Evaluation"]
page = st.sidebar.radio("Touchpoint", PAGES)
st.sidebar.caption("Rules first, learn from calls second, then use both on every live call.")
st.sidebar.caption(f"LLM: {'connected' if llm.available() else 'not configured (rules NLU + precomputed judge)'}")


def overview():
    st.title(f"{CARRIER}: AI touchpoints of a call agent")
    s = json.loads(batch.SUMMARY.read_text()) if batch.SUMMARY.exists() else {}
    emp = empirical.status()
    rs = empirical.learned_ruleset()
    c = st.columns(4)
    c[0].metric("Past calls (batch)", s.get("calls", 0))
    c[1].metric("Word error rate", f"{s.get('wer', 0):.1%}")
    c[2].metric("Rule corrections found", len(emp.get("contradictions", [])))
    c[3].metric("Corrections in use", len(rs.overrides))
    st.markdown("""
| # | Touchpoint | Here |
|---|---|---|
| 1 | World model / first principles | Plans, devices, accounts, bills, roaming zones, credit and order state machines, rules engine, expert scenarios |
| 2 | Batch voice to script | 120 recorded past calls: Whisper, speaker roles, term correction, PII redaction, dialogue acts, facts |
| 3 | Batch empirical learning | Call-flow model + rules the calls contradict, human accept/reject, held-out test, simulator check |
| 4 | Live voice to script | Microphone (or typed) caller turn -> Whisper with carrier vocabulary and prosody |
| 5 | Live script to decision | Verify (phone + PIN) before any account detail, answer from rules tools, or hand off to a person |
| 6 | Live decision to voice | Piper voice, slower and calmer when the caller sounds frustrated |
| 7 | Live decision to evaluation | Rule checks + LLM judge, held-out replay, human review queue and agreement |
""")


def world():
    st.title("1 World model")
    t1, t2, t3, t4 = st.tabs(["Entities", "Accounts", "Rules console", "Expert scenarios"])
    with t1:
        st.subheader("Plans")
        st.dataframe(pd.DataFrame([{"id": p.plan_id, "name": p.name, "price": p.price,
                                    "data": f"{p.data_gb} GB" if p.data_gb else "unlimited",
                                    "overage / GB": p.overage_per_gb or None, "slowed after (GB)": p.throttle_gb,
                                    "hotspot GB": p.hotspot_gb, "roaming included zones": sorted(p.roaming_zones),
                                    "upgrade after (payments)": p.upgrade_after} for p in PLANS.values()]), hide_index=True)
        c1, c2 = st.columns(2)
        c1.subheader("Roaming zones")
        c1.dataframe(pd.DataFrame([{"country": k, "zone": v, "zone name": ZONE_NAMES[v], "day pass": ROAMING_DAY_RATE[v]}
                                   for k, v in COUNTRIES.items()]), hide_index=True)
        c2.subheader("Devices (24 installments)")
        c2.dataframe(pd.DataFrame([{"model": d.model, "price": d.price, "monthly": d.installment} for d in DEVICES.values()]),
                     hide_index=True)
        c2.subheader("State machines")
        c2.json({"bill": {k: sorted(v) for k, v in BILL_TRANSITIONS.items()},
                 "credit request": {k: sorted(v) for k, v in CREDIT_TRANSITIONS.items()}}, expanded=False)
    with t2:
        st.dataframe(pd.DataFrame([{"account": a.account_id, "name": a.name, "phone": a.phone, "PIN": a.pin, "plan": a.plan_id,
                                    "device": a.device, "payments": a.payments_made, "data GB": a.data_used_gb,
                                    "trips": ", ".join(f"{t.country} {t.days}d" for t in a.trips), "past due": a.past_due}
                                   for a in bob().accounts.values()]), hide_index=True)
    with t3:
        a = bob().accounts[st.selectbox("Account", list(bob().accounts), format_func=lambda k: f"{k} {bob().accounts[k].name} ({bob().accounts[k].plan_id})")]
        learned = st.toggle("Use accepted learned corrections", True)
        eng = RulesEngine(ruleset(learned))
        tool = st.radio("Tool", ["bill", "data", "roaming", "upgrade", "credit"], horizontal=True)
        if tool == "roaming":
            r = eng.roaming(a, st.selectbox("Country", list(COUNTRIES)), st.number_input("Days", 1, 30, 5))
        elif tool == "credit":
            r = eng.credit(a, st.selectbox("Charge", ["roaming", "overage", "taxes"]))
        else:
            r = getattr(eng, tool)(a)
        st.json({"ok": r.ok, "reason": r.reason, "values": r.values, "rules used": r.rules})
    with t4:
        from carrier.scenarios import run_scenarios
        learned = st.toggle("Run on learned rules", False, key="sc_l")
        rows = run_scenarios(ruleset(learned))
        st.metric("Passing", f"{sum(r['pass'] for r in rows)}/{len(rows)}")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def batch_page():
    st.title("2 Batch: voice to script")
    s = json.loads(batch.SUMMARY.read_text()) if batch.SUMMARY.exists() else {}
    c = st.columns(6)
    c[0].metric("Calls", s.get("calls"))
    c[1].metric("WER (after term fix)", f"{s.get('wer', 0):.1%}", f"raw {s.get('wer_raw', 0):.1%}", delta_color="off")
    c[2].metric("Speaker roles", f"{s.get('role_acc', 0):.1%}", f"text only {s.get('role_acc_textonly', 0):.1%}", delta_color="off")
    c[3].metric("Carrier terms", f"{s.get('term_recall', 0):.1%}")
    c[4].metric("PII redacted", f"{s.get('redaction_recall', 0):.1%}")
    c[5].metric("Dialogue acts", f"{(s.get('act_acc') or 0):.1%}")
    st.caption("Precomputed offline with Whisper small.en and WeSpeaker embeddings. Re-run one call below on this server "
               "with a lighter model.")
    tx, cc = transcripts(), corpus_calls()
    cid = st.selectbox("Call", list(cc), format_func=lambda k: f"{k} - {cc[k]['intent']} ({cc[k]['split']})")
    if (corpus.AUDIO / f"{cid}.ogg").exists():
        st.audio(str(corpus.AUDIO / f"{cid}.ogg"))
    res = tx.get(cid)
    if st.button("Re-transcribe this call now (Whisper base.en, lightweight speaker features)"):
        with st.spinner("Transcribing..."):
            t = time.time()
            res = batch.run_one(cid, model="base.en", neural=False, save=False)
            st.success(f"Done in {time.time() - t:.0f}s")
    l, r = st.columns(2)
    l.subheader("Ground truth")
    for t in cc[cid]["turns"]:
        l.markdown(f"**{t['role']}** `{t['act']}` {t['text']}")
    r.subheader("Batch transcript (redacted)")
    if res:
        for t in res["turns"]:
            r.markdown(f"**{t['role']}** `{t['act']}` {t['text_redacted']}")
        r.json({"eval": res["eval"], "facts": res["features"], "term fixes": res["term_fixes"]}, expanded=False)


def empirical_page():
    st.title("3 Empirical learning")
    s = empirical.status()
    c = st.columns(3)
    c[0].metric("Trained on calls", s.get("trained_calls"))
    c[1].metric("New live calls since", s["new_calls"], f"retrain at {s['min_new_calls']}", delta_color="off")
    if c[2].button("Retrain now" + (" (due)" if s["retrain_due"] else "")):
        s = empirical.train() | {"new_calls": 0}
        st.rerun()
    st.subheader("Rules the past calls contradict")
    for k in s.get("contradictions", []):
        rev = s.get("reviews", {}).get(k["id"], {}).get("decision")
        with st.container(border=True):
            st.markdown(f"**{k['label']}**: rules say `{k['current']}`, calls say `{k['proposed']}` "
                        f"({k['contradicting']}/{k['support']} calls). {k['evidence']}. Status: **{rev or 'pending'}**")
            st.caption("Example calls: " + ", ".join(map(str, k["examples"])))
            b1, b2 = st.columns(8)[:2]
            if b1.button("Accept", key="a" + k["id"]):
                empirical.review(k["id"], "accept")
                st.rerun()
            if b2.button("Reject", key="r" + k["id"]):
                empirical.review(k["id"], "reject")
                st.rerun()
    st.subheader("Held-out test")
    ev = empirical.evaluate_heldout()
    c = st.columns(3)
    c[0].metric("Answer accuracy, rules only", f"{ev['rules_only']['answer_acc']:.1%}")
    c[1].metric("Rules + accepted corrections", f"{ev['rules_plus_learned']['answer_acc']:.1%}")
    c[2].metric("Held-out calls / answers", f"{ev['calls']} / {ev['rules_only']['n']}")
    with st.expander("Rules-only errors"):
        st.dataframe(pd.DataFrame(ev["rules_only"]["errors"]), hide_index=True)
    st.subheader("Call-flow model and simulator")
    sim = empirical.simulator_check()
    st.json(sim, expanded=False)
    flow = empirical.FlowModel(**s["flow"])
    state = st.selectbox("After", sorted(flow.transitions))
    st.dataframe(pd.DataFrame([{"next": k, "p": round(v, 3)} for k, v in flow.probs(state).items()]), hide_index=True)


# ---- live call -------------------------------------------------------------------------
def _tts(text, mood):
    a, sr = speech.synth(text, length_scale=1.12 if mood == "frustrated" else 1.0, noise_scale=0.5 if mood == "frustrated" else 0.667)
    return speech.wav_bytes(a, sr)


def live_page():
    st.title("4-6 Live call")
    ss = st.session_state
    c = st.columns([1, 1, 1, 2])
    learned = c[0].toggle("Use learned corrections", True)
    voice_on = c[1].toggle("Agent voice (Piper)", True)
    use_llm = c[2].toggle("LLM NLU fallback + judge", llm.available(), disabled=not llm.available())
    if c[3].button("Start new call", type="primary") or "call" not in ss:
        ss.call = CallSession(bob(), ruleset(learned), use_llm=use_llm)
        g = ss.call.greet()
        ss.log, ss.last_audio, ss.audio_key, ss.judged = [], None, ss.get("audio_key", 0) + 1, None
        ss.last_audio = _tts(g["reply"], "neutral") if voice_on else None
    call: CallSession = ss.call
    left, right = st.columns([3, 2])
    with left:
        for t in call.turns:
            with st.chat_message("assistant" if t["role"] == "agent" else "user"):
                st.markdown(t["text"] + (f"  \n`{t['act']}`" if t["role"] == "caller" else f"  \n`{t['act']}`"))
        if ss.last_audio:
            st.audio(ss.last_audio, format="audio/wav", autoplay=True)
        if not call.ended:
            rec = st.audio_input("Speak as the caller", key=f"mic{ss.audio_key}-{len(call.turns)}")
            typed = st.chat_input("...or type as the caller")
            text, pros, lat = None, None, {}
            if rec is not None:
                t = time.time()
                audio = speech.read_audio(rec.getvalue())
                text = speech.transcribe(audio, prompt=nlu.stt_prompt())
                lat["stt_ms"] = int((time.time() - t) * 1000)
                pros = speech.prosody(audio, len(text.split()))
            elif typed:
                text = typed
            if text:
                t = time.time()
                r = call.caller_turn(text, pros)
                lat["decide_ms"] = int((time.time() - t) * 1000)
                t = time.time()
                ss.last_audio = _tts(r["reply"], r["mood"]) if voice_on else None
                lat["tts_ms"] = int((time.time() - t) * 1000)
                ss.log.append({"heard": text, "prosody": pros, "latency": lat} | r)
                st.rerun()
        if call.ended or st.button("Hang up and grade"):
            if ss.judged is None:
                rec = call.to_record() | {"call_id": f"L-{int(call.started)}", "source": "live"}
                empirical.save_live_call(rec)
                ss.judged = evaluation.judge(rec, use_llm=use_llm)
                evaluation.save_judgment(ss.judged | {"transcript": rec["turns"]})
            j = ss.judged
            st.success("Call graded: " + ("PASS" if j["pass"] else "FAIL") + " " + json.dumps(j["scores"]))
            if j["rule_findings"]:
                st.warning("; ".join(j["rule_findings"]))
            if j.get("llm"):
                st.caption(f"LLM judge (advisory): {j['llm'].get('scores')} - {j['llm'].get('rationale')}")
    with right:
        st.subheader("Decision")
        st.markdown(f"Verified: **{call.acct is not None}** | Mood: **{call.mood}** | Failed checks: {call.failed}")
        if ss.log:
            last = ss.log[-1]
            st.markdown(f"Heard: _{last['heard']}_  \nCaller act: `{last['caller_act']}` -> agent: `{', '.join(last['acts'])}`")
            st.json({"latency": last["latency"], "prosody": last["prosody"], "tools": last["tools"],
                     "rules used": last["rules"], "term fixes": last["term_fixes"]}, expanded=True)
        with st.expander("Test accounts (phone and PIN)"):
            demo = list(bob().accounts.values())[:6]
            st.dataframe(pd.DataFrame([{"phone": a.phone, "PIN": a.pin, "plan": a.plan.name, "data GB": a.data_used_gb,
                                        "trips": ", ".join(f"{t.country} {t.days}d" for t in a.trips),
                                        "past due": a.past_due} for a in demo]), hide_index=True)
        st.caption("Try: 'My data is really slow', 'I'm going to France for 5 days, will I be charged?', "
                   "'Can I upgrade my phone?', 'Can you credit the overage?'")


def eval_page():
    st.title("7 Evaluation")
    t1, t2, t3 = st.tabs(["Replay held-out calls", "Human review", "All judgments"])
    with t1:
        learned = st.toggle("Agent uses learned corrections", True, key="ev_l")
        r = evaluation.replay(ruleset(learned))
        c = st.columns(4)
        c[0].metric("Outcome match", f"{r['outcome_match']:.0%}")
        c[1].metric("Agent act match", f"{r['mean_act_match']:.0%}")
        c[2].metric("Facts match billing/network", f"{r['facts_match']:.0%}")
        c[3].metric("Judge pass rate", f"{r['judge_pass_rate']:.0%}")
        st.dataframe(pd.DataFrame(r["rows"]), hide_index=True)
        st.caption("Scores come from the rule checks. The LLM judge (Qwen2.5 3B via llama.cpp, precomputed offline for "
                   f"replays) is advisory: it disagreed on {sum(bool(x['llm_disagrees']) for x in r['rows'])} calls, which go to human review.")
    with t2:
        ag = evaluation.agreement()
        st.dataframe(pd.DataFrame([{"dimension": k} | v for k, v in ag.items()]), hide_index=True)
        q = evaluation.review_queue()
        st.caption(f"{len(q)} calls in queue ({int(evaluation.REVIEW_RATE * 100)}% sample + every failed call). "
                   "Disagreements with the judge become judge examples.")
        if q:
            j = q[0]
            st.markdown(f"**{j['call_id']}** judge: {j['scores']} {j['rule_findings']}")
            if j.get("llm"):
                st.caption(f"LLM judge: {j['llm'].get('scores')} - {j['llm'].get('rationale')}")
            for t in j.get("transcript", []):
                st.markdown(f"**{t['role']}**: {t['text']}")
            with st.form("rev"):
                sc = {d: int(st.radio(d, [1, 0], index=0 if j["scores"][d] else 1, horizontal=True, key="rv" + d))
                      for d in evaluation.DIMS}
                note = st.text_input("Note")
                if st.form_submit_button("Submit review"):
                    evaluation.submit_review(j["call_id"], sc, note)
                    st.rerun()
    with t3:
        js = evaluation.judgments()
        st.dataframe(pd.DataFrame([{"call": k, "source": v["source"], "pass": v["pass"], **v["scores"],
                                    "findings": "; ".join(v["rule_findings"])} for k, v in js.items()]), hide_index=True)


{"Overview": overview, "1 World model": world, "2 Batch: voice to script": batch_page,
 "3 Empirical learning": empirical_page, "4-6 Live call": live_page, "7 Evaluation": eval_page}[page]()
