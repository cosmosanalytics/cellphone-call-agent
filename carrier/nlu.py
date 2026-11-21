"""Carrier vocabulary, term correction, entity extraction, PII redaction and dialogue acts."""
from __future__ import annotations

import re

from .textnorm import digit_runs
from .world import CARRIER, COUNTRIES, DEVICES, PLANS

# ---- vocabulary -------------------------------------------------------------
LEXICON = sorted({p.name.lower() for p in PLANS.values()} | {d.model.lower() for d in DEVICES.values()}
                 | {c.lower() for c in COUNTRIES}
                 | {"roaming", "day pass", "hotspot", "data overage", "overage", "installment", "upgrade",
                    "payoff", "courtesy credit", "billing cycle", "full speed", "priority data", "past due",
                    "account pin", "northstar wireless", "autopay", "taxes and fees"})


def stt_prompt() -> str:
    return (f"{CARRIER} customer care call. Plans: Basic 5 GB, Unlimited Starter, Unlimited Plus. "
            "Phones: iPhone 16, Galaxy S25, Pixel 9, Moto G. Roaming day pass, hotspot, data overage, "
            "installment, upgrade, courtesy credit, account PIN.")


# ASR variants -> canonical term
ALIASES = [
    (r"\bnorth\s?star(\s+wireless)?\b", "Northstar Wireless"),
    (r"\bi[\s-]?phone(s?)\b", r"iPhone\1"),
    (r"\bgalaxy\s+s\s*-?\s*(?:25|twenty[\s-]five)\b", "Galaxy S25"),
    (r"\bpixel\s+(?:nine|9)\b", "Pixel 9"),
    (r"\bmoto(?:rola)?\s+(?:g|gee)\b", "Moto G"),
    (r"\bunlimited\s+start(?:er|a|ers)\b", "Unlimited Starter"),
    (r"\bunlimited\s+plus\b", "Unlimited Plus"),
    (r"\bbasic\s+(?:5|five)\s*(?:gb|gigs?|gigabytes?)\b", "Basic 5 GB"),
    (r"\b(\d+)\s*(?:gigabytes?|gigs?|g\.?b\.?)(?=\W|$)", r"\1 GB"),
    (r"\b(?:hot\s+spot|hot-spot)\b", "hotspot"),
    (r"\bday\s*-?\s*pass(es)?\b", r"day pass\1"),
    (r"\bover\s*-?\s*age\b", "overage"),
    (r"\bpin\s+number\b", "PIN"),
    (r"\b(?:roman|rowing|roam in)\s+(charges?|fees?|day)\b", r"roaming \1"),
    (r"\bin\s+stall\s*ments?\b", "installment"),
    (r"\bu\.?k\.?\b(?!\w)", "United Kingdom"),
]


def correct_terms(text: str) -> tuple[str, list[dict]]:
    fixes = []
    out = text
    for pat, rep in ALIASES:
        def sub(m, rep=rep):
            new = m.expand(rep)
            if new != m.group(0):
                fixes.append({"from": m.group(0), "to": new})
            return new
        out = re.sub(pat, sub, out, flags=re.I)
    return out, fixes


# ---- entities -----------------------------------------------------------------
def find_phone(text: str) -> str | None:
    for r in digit_runs(text):
        if len(r) == 11 and r[0] == "1":
            r = r[1:]
        if len(r) == 10:
            return f"{r[:3]}-{r[3:6]}-{r[6:]}"
    return None


def find_pin(text: str) -> str | None:
    runs = [r for r in digit_runs(text) if len(r) not in (10, 11)]
    m = re.search(r"\b(?:pin|code|passcode)\b(.*)", text, re.I)
    if m:
        after = [r for r in digit_runs(m.group(1)) if len(r) == 4]
        if after:
            return after[0]
    fours = [r for r in runs if len(r) == 4]
    return fours[0] if fours else None


def find_country(text: str) -> str | None:
    t = text.lower()
    hits = [(t.find(c.lower()), c) for c in COUNTRIES if re.search(rf"\b{c.lower()}\b", t)]
    if re.search(r"\b(england|london|britain)\b", t):
        hits.append((t.find("england"), "United Kingdom"))
    if re.search(r"\b(paris)\b", t):
        hits.append((t.find("paris"), "France"))
    if re.search(r"\b(rome|milan)\b", t):
        hits.append((0, "Italy"))
    if re.search(r"\b(tokyo)\b", t):
        hits.append((0, "Japan"))
    if re.search(r"\b(toronto|vancouver)\b", t):
        hits.append((0, "Canada"))
    return min(hits)[1] if hits else None


_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split())}


def find_days(text: str) -> int | None:
    t = text.lower()
    m = re.search(r"\b(\d+|" + "|".join(_NUM) + r"|a)\s+(days?|weeks?|nights?)\b", t)
    if m:
        n = 1 if m.group(1) == "a" else int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
        return n * 7 if m.group(2).startswith("week") else n
    if "weekend" in t:
        return 2
    return None


def find_device(text: str) -> str | None:
    t = text.lower()
    for d in DEVICES.values():
        if any(re.search(rf"\b{re.escape(s)}\b", t) for s in d.spoken):
            return d.model
    return None


def find_plan(text: str) -> str | None:
    t = text.lower()
    for p in PLANS.values():
        if p.name.lower() in t:
            return p.plan_id
    return None


def find_charge_type(text: str) -> str | None:
    t = text.lower()
    if re.search(r"roam|day pass|international|abroad|travel|trip", t):
        return "roaming"
    if re.search(r"overage|data (charge|fee)|extra data|went over|over my data", t):
        return "overage"
    if re.search(r"tax|fee", t):
        return "taxes"
    if re.search(r"installment|device|phone payment", t):
        return "device"
    return None


def find_gb(text: str) -> int | None:
    m = re.search(r"\b(\d+)\s*GB\b", text)
    return int(m.group(1)) if m else None


def find_money(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\$(\d[\d,]*(?:\.\d\d)?)", text)]


# ---- redaction ------------------------------------------------------------------
def redact(text: str, names: list[str] | None = None, identity_turn: bool = False) -> tuple[str, list[dict]]:
    found: list[dict] = []

    def sub(pattern, tag, s, flags=re.I):
        def rep(m):
            found.append({"type": tag, "text": m.group(0)})
            return f"[{tag}]"
        return re.sub(pattern, rep, s, flags=flags)

    digit = r"(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|oh)"
    t = sub(rf"\(?\b{digit}(?:[\s,.()-]*{digit}){{9,10}}\b", "PHONE", text)
    t = sub(rf"\b(pin|passcode|code)(\s+is)?[\s,:]*{digit}(?:[\s,.-]*{digit}){{3}}\b", "PIN", t)
    if identity_turn:
        t = sub(rf"\b{digit}(?:[\s,.-]*{digit}){{3}}\b", "PIN", t)
    t = sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "EMAIL", t)
    for n in sorted(set(filter(None, names or [])), key=len, reverse=True):
        t = sub(rf"\b{re.escape(n)}\b", "NAME", t)
    return t, found


# ---- dialogue acts -----------------------------------------------------------------
CALLER_ACTS = [
    ("request_human", r"\b(real person|a human|supervisor|manager|representative|speak (to|with) (someone|a person))\b"),
    ("request_cancel", r"\bcancel|port (my number )?out|switch(ing)? (to another )?carriers?|leave northstar\b"),
    ("request_credit", r"\bcredit|refund|waive|take (that|it|them) off|reverse (the|that)|remove (the|that) charge"),
    ("ask_roaming", r"\b(will|would|do) i (be charged|pay|need|have to pay)|going to|travel(l?ing)?|trip to|heading to|"
                    r"planning|next (week|month)|day pass|work abroad|use my phone in|phone work in|roaming cost"),
    ("ask_upgrade", r"\bupgrade|new phone|trade[ -]?in|switch phones|get (the|a) (new )?(iphone|galaxy|pixel|moto)"),
    ("ask_bill", r"\bbill|charged|charges?\b|statement|invoice|higher than|why .*(pay|owe)"),
    ("ask_data", r"\bdata|slow|speeds?|internet|\bGB\b|hotspot|throttl|loading"),
]


def has_identity(text: str) -> bool:
    return bool(find_phone(text)) or (bool(find_pin(text)) and len(text.split()) <= 14)


def classify_caller(text: str) -> str:
    t = text.lower()
    for act, pat in CALLER_ACTS[:3]:
        if re.search(pat, t, re.I):
            return act
    if has_identity(text) and not re.search(r"\b(bill|data|travel|upgrade|roam)", t):
        return "give_identity"
    for act, pat in CALLER_ACTS[3:]:
        if re.search(pat, text, re.I):
            return act
    if re.search(r"ridiculous|kidding|frustrat|unacceptable|terrible|fed up|annoy|outrageous|joke|ripped off", t):
        return "express_frustration"
    if re.search(r"\b(that'?s all|bye|goodbye|that'?s it|nothing else|no thanks?|have a good)\b", t):
        return "thank_bye"
    if re.search(r"\b(ok|okay|yes|yeah|sure|great|got it|alright|all right|thanks|thank you|perfect|sounds good|makes sense)\b", t):
        return "confirm"
    if has_identity(text):
        return "give_identity"
    return "other"


AGENT_ACTS = [   # ordered by priority: a turn's primary act is its most consequential one
    ("answer_credit", r"courtesy credit|credit (the|that|your)|can'?t credit|cannot credit|i'?ve applied"),
    ("answer_upgrade", r"upgrade|payoff|installments? paid|pay off"),
    ("answer_bill", r"your (current )?bill (is|total)|bill total|on your bill"),
    ("answer_roaming", r"day pass|roaming in \w+|included on your plan at no|no roaming"),
    ("answer_data", r"used \d+ ?GB|full speed|speeds are reduced|priority data|of your \d+ ?GB|left this cycle"),
    ("transfer_human", r"transferring you|transfer you|connect you (to|with)"),
    ("verify_fail", r"doesn'?t match|does not match|couldn'?t verify|wasn'?t able to verify"),
    ("verify_ok", r"you'?re verified|thanks? for verifying|i'?ve verified"),
    ("request_verification", r"phone number (on|for) the account|(four|4)[- ]digit (account )?pin|your account pin"),
    ("ask_country", r"which country|where are you (traveling|travelling|going|headed)"),
    ("clarify", r"which charge|didn'?t catch"),
    ("empathize", r"understand|sorry|apolog"),
    ("closing", r"have a (great|good|nice|wonderful) (day|trip)"),
    ("greeting", r"thanks? (you )?for calling .*how can i help"),
    ("anything_else", r"anything else|what can i help"),
]

AGENT_PRIORITY = [a for a, _ in AGENT_ACTS]


def classify_agent(text: str) -> str:
    for act, pat in AGENT_ACTS:
        if re.search(pat, text, re.I):
            return act
    return "clarify"


def classify_act(role: str, text: str) -> str:
    return classify_agent(text) if role == "agent" else classify_caller(text)


_AGENT_CUES = r"thanks? for calling|how can i help|let me (check|pull|look)|i see|your account|you'?re verified|anything else|" \
              r"have a (great|good) day|i can help|i'?m sorry|i understand|your bill|you'?ve used|your plan"
_CALLER_CUES = r"\bmy (bill|phone|data|number|pin|plan)\b|i want|i'?m (going|traveling|calling)|can i\b|\bwhy (is|was|am)\b|i need|i was"


def role_cue_score(text: str) -> float:
    t = text.lower()
    return float(len(re.findall(_AGENT_CUES, t)) - len(re.findall(_CALLER_CUES, t)))


def detect_mood(text: str, prosody: dict | None = None) -> str:
    t = text.lower()
    score = len(re.findall(r"ridiculous|kidding|frustrat|unacceptable|terrible|fed up|annoy|outrageous|ripped off|again\?|!", t))
    if prosody and prosody.get("loud") and prosody.get("fast"):
        score += 1
    if score >= 1:
        return "frustrated"
    if re.search(r"\b(confus|don'?t understand|what does that mean|not sure|huh)\b", t):
        return "confused"
    return "neutral"
