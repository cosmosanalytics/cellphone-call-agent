"""Text normalization shared by TTS (spoken forms) and evaluation (WER)."""
from __future__ import annotations

import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def money_words(m: re.Match) -> str:
    amount = float(m.group(1).replace(",", ""))
    dollars, cents = int(amount), int(round((amount - int(amount)) * 100))
    s = f"{num_words(dollars)} dollar{'s' if dollars != 1 else ''}"
    if cents:
        s += f" and {num_words(cents)} cents"
    return s


def spoken(text: str) -> str:
    """Make text easy for TTS: read phone numbers and PINs digit by digit, expand money and units."""
    text = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b",
                  lambda m: ", ".join(" ".join(g) for g in m.groups()), text)
    text = re.sub(r"\$(\d[\d,]*(?:\.\d\d)?)", money_words, text)
    text = re.sub(r"\b(\d{4})\b(?!\s*(GB|days?))", lambda m: " ".join(m.group(1)), text)
    text = re.sub(r"\bS25\b", "S twenty five", text)
    text = re.sub(r"\b(\d+) GB\b", lambda m: f"{m.group(1)} gigabytes", text)
    return text.replace("PIN", "pin")


DIGIT_WORDS = {w: str(i) for i, w in enumerate(ONES[:10])} | {"oh": "0", "o": "0"}
TEEN_WORDS = {w: i for i, w in enumerate(ONES) if i >= 10}
TENS_WORDS = {w: i * 10 for i, w in enumerate(TENS) if w != "_"}


def digit_runs(text: str) -> list[str]:
    """Contiguous digit strings, reading spoken digits ('three one two', 'forty seven', 'oh')."""
    toks = re.findall(r"\d+|[a-z]+", text.lower().replace("-", " "))
    runs, cur = [], ""
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit():
            cur += t
        elif t in DIGIT_WORDS and (t not in ("o",) or cur):
            cur += DIGIT_WORDS[t]
        elif t in TEEN_WORDS:
            cur += str(TEEN_WORDS[t])
        elif t in TENS_WORDS:
            v = TENS_WORDS[t]
            if i + 1 < len(toks) and toks[i + 1] in DIGIT_WORDS and toks[i + 1] not in ("oh", "o", "zero"):
                v += int(DIGIT_WORDS[toks[i + 1]])
                i += 1
            cur += str(v)
        elif t == "double" and i + 1 < len(toks) and toks[i + 1] in DIGIT_WORDS:
            cur += DIGIT_WORDS[toks[i + 1]] * 2
            i += 1
        else:
            if cur:
                runs.append(cur)
            cur = ""
        i += 1
    if cur:
        runs.append(cur)
    return runs


def normalize_for_wer(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = re.sub(r"\$(\d[\d,]*)(\.00)?", r"\1 dollars", t)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\.(\d\d)\b", r" \1", t)
    t = re.sub(r"\bgigabytes?\b|\bgigs?\b", "gb", t)
    t = re.sub(r"(\d)gb\b", r"\1 gb", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = t.replace("'s", "s").replace("'", "")
    out: list[str] = []
    for tok in t.split():
        if tok in DIGIT_WORDS and tok not in ("o", "oh") or tok in TEEN_WORDS:
            tok = DIGIT_WORDS.get(tok) or str(TEEN_WORDS[tok])
        elif tok in TENS_WORDS:
            tok = str(TENS_WORDS[tok])
        if out and tok.isdigit() and out[-1].isdigit():
            prev = out[-1]
            if len(prev) >= 2 and prev.endswith("0") and len(tok) == 1 and int(prev) < 100 and int(prev) % 10 == 0:
                out[-1] = str(int(prev) + int(tok))    # "forty" "seven" -> 47
            else:
                out[-1] = prev + tok                   # digit strings: "3 1 2" -> 312
        else:
            out.append(tok)
    return " ".join(out)
