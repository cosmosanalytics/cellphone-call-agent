"""Expert scenarios: hand-written cases a carrier policy expert signs off on (stage 1 tests)."""
from __future__ import annotations

from .world import Account, RulesEngine, TODAY, Trip


def _acct(**kw) -> Account:
    base = dict(account_id="T1", first="Test", last="Caller", phone="312-555-0100", pin="1234", plan_id="UNS",
                device="Pixel 9", payments_made=12, data_used_gb=10, trips=[], past_due=False)
    return Account(**(base | kw))


SCENARIOS = [
    ("S01", "Basic 5 GB, 3 GB used: 2 GB left, no overage", lambda e: e.data(_acct(plan_id="BAS", data_used_gb=3)).values["remaining"] == 2),
    ("S02", "Basic 5 GB, 8 GB used: $30 overage", lambda e: e.data(_acct(plan_id="BAS", data_used_gb=8)).values["overage"] == 30),
    ("S03", "Unlimited Starter, 72 GB used: still full speed", lambda e: not e.data(_acct(data_used_gb=72)).values["throttled"]),
    ("S04", "Unlimited Starter, 120 GB used: slowed", lambda e: e.data(_acct(data_used_gb=120)).values["throttled"]),
    ("S05", "Unlimited Plus never slowed", lambda e: not e.data(_acct(plan_id="UNP", data_used_gb=300)).values["throttled"]),
    ("S06", "Canada included on Unlimited Starter", lambda e: e.roaming(_acct(), "Canada", 5).values["included"]),
    ("S07", "Canada on Basic: $5/day, 4 days = $20", lambda e: e.roaming(_acct(plan_id="BAS"), "Canada", 4).values["total"] == 20),
    ("S08", "France on Unlimited Plus: $10/day day pass", lambda e: e.roaming(_acct(plan_id="UNP"), "France", 3).values["total"] == 30),
    ("S09", "Japan on Unlimited Plus: $15/day", lambda e: e.roaming(_acct(plan_id="UNP"), "Japan", 2).values["total"] == 30),
    ("S10", "Upgrade with 12/24 payments on Unlimited Starter: eligible", lambda e: e.upgrade(_acct(payments_made=12)).ok),
    ("S11", "Upgrade with 6/24 payments on Unlimited Plus: eligible", lambda e: e.upgrade(_acct(plan_id="UNP", payments_made=6)).ok),
    ("S12", "Upgrade with past-due balance: blocked", lambda e: e.upgrade(_acct(past_due=True)).reason == "PAST_DUE"),
    ("S13", "Courtesy credit for $30 roaming: approved", lambda e: e.credit(_acct(plan_id="BAS", trips=[Trip("Mexico", 6)]), "roaming").ok),
    ("S14", "Courtesy credit over $50: needs a specialist", lambda e: e.credit(_acct(trips=[Trip("Japan", 5)]), "roaming").reason == "OVER_LIMIT"),
    ("S15", "Second courtesy credit within 12 months: denied",
     lambda e: e.credit(_acct(plan_id="BAS", trips=[Trip("Mexico", 2)], last_courtesy_credit=TODAY.replace(month=3)), "roaming").reason == "RECENT_CREDIT"),
    ("S16", "Taxes are never creditable", lambda e: e.credit(_acct(), "taxes").reason == "NOT_CREDITABLE"),
    ("S17", "Bill: plan + installment + taxes", lambda e: e.bill(_acct(payments_made=3)).values["total"] == round(55 + 27 + 6.75, 2)),
]


def run_scenarios(ruleset=None) -> list[dict]:
    e = RulesEngine(ruleset)
    return [{"id": i, "scenario": d, "pass": bool(f(e))} for i, d, f in SCENARIOS]
