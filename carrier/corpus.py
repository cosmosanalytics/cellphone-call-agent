"""Synthetic call corpus: scripted callers talk to the call agent running on the *hidden* reality
rules (what billing and the network really do), then both sides are rendered with Piper voices.
Ground truth (turns, roles, acts, facts) is stored next to each call."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .agent import CallSession
from .nlu import AGENT_PRIORITY, classify_agent
from .world import COUNTRIES, DEVICES, REALITY, TODAY, BookOfBusiness, Trip

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
AUDIO = DATA / "audio"
WAV_CACHE = Path.home() / ".cache" / "carrier-wav"


class _OneAccount:
    def __init__(self, acct):
        self.acct = acct

    def verify(self, phone, pin):
        return self.acct if phone == self.acct.phone and pin == self.acct.pin else None


def _country(rng, zone_w=(0.25, 0.5, 0.25)):
    zone = rng.choices([1, 2, 3], zone_w)[0]
    return rng.choice([c for c, z in COUNTRIES.items() if z == zone])


def _scenario(bob, rng, i):
    intent = rng.choices(["ask_data", "ask_roaming", "ask_bill", "ask_upgrade", "request_credit"],
                         [0.26, 0.26, 0.2, 0.14, 0.14])[0]
    base = rng.choice(list(bob.accounts.values()))
    ch: dict = {"trips": []}
    if intent == "ask_data":
        ch["data_used_gb"] = {"BAS": rng.randint(1, 10), "UNS": rng.randint(10, 120), "UNP": rng.randint(5, 150)}[base.plan_id]
    elif intent == "ask_bill":
        if rng.random() < 0.65:
            ch["trips"] = [Trip(_country(rng), rng.randint(2, 9))]
        if base.plan_id == "BAS" and rng.random() < 0.5:
            ch["data_used_gb"] = rng.randint(6, 10)
    elif intent == "ask_upgrade":
        ch["payments_made"] = rng.randint(0, 24)
        ch["past_due"] = rng.random() < 0.15
    elif intent == "request_credit":
        if base.plan_id == "BAS" and rng.random() < 0.5:
            ch["data_used_gb"] = rng.randint(6, 12)
        else:
            ch["trips"] = [Trip(_country(rng, (0.2, 0.3, 0.5)), rng.randint(2, 8))]
        ch["last_courtesy_credit"] = TODAY.replace(year=TODAY.year - 1, month=rng.randint(1, 12), day=15) if rng.random() < 0.25 else None
        if ch["last_courtesy_credit"] and (TODAY - ch["last_courtesy_credit"]).days >= 365:
            ch["last_courtesy_credit"] = None
    return intent, bob.snapshot(base, **ch)


def _opening(intent, acct, rng, slots):
    if intent == "ask_data":
        return rng.choice(["Hi, my internet has been really slow lately. Is something wrong with my data?",
                           "I want to check how much data I've used this month.",
                           "Why is my data so slow? Pages take forever to load.",
                           "Hi there. How much data do I have left?"])
    if intent == "ask_roaming":
        c, d = slots["country"], slots["days"]
        return rng.choice([f"I'm traveling to {c} for {d} days next week. Will I be charged for using my phone there?",
                           f"I'm going to {c} next month for {d} days. Do I need a day pass?",
                           f"Will my phone work in {c}? I'll be there for {d} days.",
                           "Hi, I'm traveling abroad soon and want to know what roaming will cost."])
    if intent == "ask_bill":
        opts = ["Why is my bill so high this month?", "Can you explain my bill? It's higher than usual."]
        if acct.trips:
            opts.append(f"I just got back from {acct.trips[0].country} and I want to know if I was charged roaming on my bill.")
        return rng.choice(opts)
    if intent == "ask_upgrade":
        return rng.choice([f"I'd like to upgrade to the new {rng.choice([d for d in DEVICES if d != acct.device])}. Am I eligible?",
                           "Can I upgrade my phone yet?",
                           f"I want a new phone. Can I trade in my {acct.device}?"])
    kind = "roaming" if acct.trips else "overage"
    return rng.choice([f"I was charged for roaming on my last bill. Can you take that off?" if kind == "roaming" else
                       "Can you waive the data overage charge? I didn't realize I went over.",
                       "I'd like a credit for the extra charges on my bill."])


def generate(i: int, bob: BookOfBusiness, rng: random.Random) -> dict:
    intent, acct = _scenario(bob, rng, i)
    mood = rng.choices(["neutral", "frustrated"], [0.7, 0.3])[0]
    slots = {"country": _country(rng), "days": rng.randint(2, 12)}
    s = CallSession(_OneAccount(acct), REALITY)
    s.greet()
    say_caller = s.caller_turn

    opening = _opening(intent, acct, rng, slots)
    if mood == "frustrated" and rng.random() < 0.5:
        opening = rng.choice(["This is the second time I'm calling. ", "Honestly, I'm pretty annoyed. "]) + opening
    r = say_caller(opening)
    pin_wrong = rng.random() < 0.12
    fail_twice = pin_wrong and rng.random() < 0.3

    def give_id(wrong=False):
        pin = acct.pin if not wrong else f"{(int(acct.pin) + 1111) % 10000:04d}"
        if rng.random() < 0.5:
            return say_caller(rng.choice([f"Sure, it's {acct.phone} and my PIN is {pin}.",
                                          f"The number is {acct.phone}, PIN {pin}."]))
        say_caller(rng.choice([f"Sure, it's {acct.phone}.", f"It's {acct.phone}."]))
        return say_caller(rng.choice([f"It's {pin}.", f"My PIN is {pin}."]))

    if "request_verification" in r["acts"]:
        r = give_id(wrong=pin_wrong)
        if "verify_fail" in r["acts"]:
            r = give_id(wrong=fail_twice)
    for _ in range(3):   # answer any clarifying question
        if "ask_country" in r["acts"]:
            r = say_caller(f"I'm going to {slots['country']} for {slots['days']} days.")
        elif "clarify" in r["acts"]:
            r = say_caller(rng.choice(["The roaming charge.", "The data overage."]))
        else:
            break
    if not s.ended:
        f = s.facts[-1] if s.facts else {}
        negative = (f.get("throttled") or f.get("roaming_included") is False or f.get("eligible") is False
                    or f.get("approved") is False or f.get("overage") or
                    any(not x["included"] for x in f.get("roaming", [])))
        if negative and (mood == "frustrated" or rng.random() < 0.2):
            say_caller(rng.choice(["That's ridiculous. Nobody told me about that.",
                                   "Are you kidding me? I pay for this every month.", "This is really frustrating."]))
            if mood == "frustrated" and rng.random() < 0.5:
                say_caller(rng.choice(["I want to speak to a supervisor.", "Can I talk to a real person, please?"]))
        if not s.ended:
            say_caller(rng.choice(["No, that's all. Thank you.", "No, that's it. Thanks for your help.",
                                   "Nope, that's everything. Bye."]))
    # merge consecutive agent segments into one spoken turn; primary act by priority
    turns = []
    for t in s.turns:
        if turns and turns[-1]["role"] == t["role"] == "agent":
            turns[-1]["text"] += " " + t["text"]
            turns[-1]["acts"].append(t["act"])
        else:
            turns.append({"role": t["role"], "text": t["text"], "acts": [t["act"]]})
    for t in turns:
        t["act"] = classify_agent(t["text"]) if t["role"] == "agent" else t["acts"][0]
        assert t["role"] == "caller" or t["act"] in t["acts"] or t["act"] in AGENT_PRIORITY
    rec = s.to_record()
    return {"call_id": f"C{i:04d}", "intent": intent, "mood": mood, "account_id": acct.account_id,
            "phone": acct.phone, "pin": acct.pin, "names": [acct.first, acct.last], "plan_id": acct.plan_id,
            "snapshot": {"data_used_gb": acct.data_used_gb, "payments_made": acct.payments_made,
                         "past_due": acct.past_due, "trips": [[t.country, t.days] for t in acct.trips],
                         "last_courtesy_credit": acct.last_courtesy_credit.isoformat() if acct.last_courtesy_credit else None,
                         "device": acct.device},
            "turns": turns, "facts": rec["facts"], "outcome": rec["outcome"], "verified": rec["verified"]}


def render(call: dict, agent_voice: str, caller_voice: str, rng: random.Random) -> np.ndarray:
    from . import speech
    pieces, t = [np.zeros(int(0.3 * 16000), np.float32)], 0.3
    for turn in call["turns"]:
        v = agent_voice if turn["role"] == "agent" else caller_voice
        audio = speech.synth16k(turn["text"], v, length_scale=rng.uniform(0.95, 1.1))
        turn["start"], turn["end"] = round(t, 2), round(t + len(audio) / 16000, 2)
        gap = rng.uniform(0.35, 0.9)
        pieces += [audio, np.zeros(int(gap * 16000), np.float32)]
        t = turn["end"] + gap
    out = np.concatenate(pieces)
    out += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 0.003, len(out)).astype(np.float32)
    return np.clip(out, -1, 1)


def build(n: int = 120, holdout: int = 30, audio: bool = True, progress=None) -> list[dict]:
    import soundfile as sf
    from . import speech
    for d in (CORPUS, AUDIO, WAV_CACHE):
        d.mkdir(parents=True, exist_ok=True)
    bob, rng = BookOfBusiness(), random.Random(11)
    callers = [v for v in speech.local_voices() if v != speech.AGENT_VOICE]
    out = []
    for i in range(n):
        call = generate(i, bob, rng)
        call["split"] = "heldout" if i >= n - holdout else "train"
        if audio:
            call["agent_voice"], call["caller_voice"] = speech.AGENT_VOICE, rng.choice(callers)
            a = render(call, call["agent_voice"], call["caller_voice"], rng)
            sf.write(WAV_CACHE / f"{call['call_id']}.wav", a, 16000)
            sf.write(AUDIO / f"{call['call_id']}.ogg", a, 16000, format="OGG", subtype="VORBIS")
            call["duration"] = round(len(a) / 16000, 2)
        (CORPUS / f"{call['call_id']}.json").write_text(json.dumps(call, indent=1))
        out.append(call)
        if progress:
            progress(i + 1, n)
    return out


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CORPUS.glob("C*.json"))]
