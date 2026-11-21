import random

from carrier import corpus, nlu
from carrier.agent import CallSession
from carrier.empirical import find_contradictions
from carrier.evaluation import rule_checks
from carrier.scenarios import run_scenarios
from carrier.textnorm import normalize_for_wer
from carrier.world import REALITY, BookOfBusiness

BOB = BookOfBusiness()
A = list(BOB.accounts.values())


def test_scenarios_pass_on_published_rules():
    assert all(r["pass"] for r in run_scenarios())


def test_no_account_details_before_verification():
    s = CallSession(BOB)
    s.greet()
    r = s.caller_turn("Why is my bill so high?")
    assert r["acts"] == ["request_verification"] and not r["tools"]


def test_verify_then_answer_pending_intent():
    s = CallSession(BOB)
    s.greet()
    s.caller_turn("My data is really slow")
    r = s.caller_turn(f"It's {A[0].phone} and my PIN is {A[0].pin}")
    assert r["acts"][:2] == ["verify_ok", "answer_data"]


def test_two_failed_verifications_transfer():
    s = CallSession(BOB)
    s.greet()
    s.caller_turn("Can I upgrade my phone?")
    s.caller_turn(f"{A[0].phone}, PIN 0000")
    r = s.caller_turn(f"{A[0].phone}, PIN 0001")
    assert "transfer_human" in r["acts"] and s.outcome == "escalated"


def test_human_request_transfers():
    s = CallSession(BOB)
    s.greet()
    assert "transfer_human" in s.caller_turn("I want to speak to a supervisor")["acts"]


def test_spoken_phone_and_pin():
    assert nlu.find_phone("it's three one two, five five five, oh one four seven") == "312-555-0147"
    assert nlu.find_pin("my pin is four seven two one") == "4721"


def test_redaction():
    t, found = nlu.redact("The number is 312-555-0147, PIN 4721. Thanks, Maria.", ["Maria"])
    assert "0147" not in t and "4721" not in t and "Maria" not in t


def test_wer_normalization():
    assert normalize_for_wer("I used 72 gigabytes, three one two") == normalize_for_wer("I used seventy two GB, 312")


def test_rule_check_flags_cpni_leak():
    rec = {"turns": [{"role": "caller", "act": "ask_bill", "text": "bill?"},
                     {"role": "agent", "act": "answer_bill", "text": "Your current bill is $90."}], "facts": []}
    assert rule_checks(rec)["scores"]["compliance"] == 0


def test_contradictions_found_from_ground_truth_calls():
    bob, rng = BookOfBusiness(), random.Random(11)
    feats = []
    for i in range(120):
        c = corpus.generate(i, bob, rng)
        f = {"plan_id": c["plan_id"], "roaming": []}
        for x in c["facts"]:
            if "throttled" in x and c["plan_id"] != "BAS":
                f |= {"used": x["used"], "throttled": x["throttled"]}
            if "roaming_included" in x:
                f["roaming"].append({"zone": x["zone"], "included": x["roaming_included"]})
            f["roaming"] += [{"zone": r["zone"], "included": r["included"]} for r in x.get("roaming", [])]
        feats.append((c["call_id"], f))
    found = {tuple(c["key"]): c["proposed"] for c in find_contradictions(feats)}
    assert found == dict(REALITY.overrides)
