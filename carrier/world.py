"""Stage 1 - World model for a cell phone carrier call center (Northstar Wireless).

Entity dictionary (plans, devices, accounts/lines, bills, credit requests, roaming
zones), state machines, a deterministic rules engine and a seeded synthetic book
of business. All data is synthetic.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

TODAY = date(2026, 11, 20)
CYCLE_END = date(2026, 12, 1)
CARRIER = "Northstar Wireless"

# ---- entity dictionary -----------------------------------------------------
ZONE_NAMES = {1: "North America", 2: "Europe", 3: "Asia-Pacific and South America"}
COUNTRIES = {
    "Canada": 1, "Mexico": 1,
    "France": 2, "Italy": 2, "Spain": 2, "Germany": 2, "United Kingdom": 2,
    "Japan": 3, "Australia": 3, "Brazil": 3, "India": 3,
}
ROAMING_DAY_RATE = {1: 5.0, 2: 10.0, 3: 15.0}


@dataclass(frozen=True)
class Plan:
    plan_id: str
    name: str
    price: float
    data_gb: int | None             # None = unlimited
    overage_per_gb: float           # for capped plans
    throttle_gb: int | None         # unlimited plans: speeds reduced after this much priority data
    hotspot_gb: int
    roaming_zones: frozenset[int]   # zones included at no charge
    upgrade_after: int              # device installment payments required before an upgrade


PLANS: dict[str, Plan] = {p.plan_id: p for p in [
    Plan("BAS", "Basic 5 GB", 35, 5, 10.0, None, 0, frozenset(), 12),
    Plan("UNS", "Unlimited Starter", 55, None, 0.0, 100, 5, frozenset({1}), 12),
    Plan("UNP", "Unlimited Plus", 75, None, 0.0, None, 30, frozenset({1}), 6),
]}


@dataclass(frozen=True)
class Device:
    model: str
    price: float
    spoken: tuple[str, ...]

    @property
    def installment(self) -> float:
        return round(self.price / INSTALLMENTS, 2)


INSTALLMENTS = 24
DEVICES: dict[str, Device] = {d.model: d for d in [
    Device("iPhone 16", 840, ("iphone 16", "iphone sixteen", "iphone")),
    Device("Galaxy S25", 816, ("galaxy s25", "galaxy s 25", "samsung", "galaxy")),
    Device("Pixel 9", 648, ("pixel 9", "pixel nine", "pixel")),
    Device("Moto G", 192, ("moto g", "motorola")),
]}


@dataclass
class Trip:
    country: str
    days: int

    @property
    def zone(self) -> int:
        return COUNTRIES[self.country]


@dataclass
class Account:
    account_id: str
    first: str
    last: str
    phone: str          # the line on the account, NNN-NNN-NNNN
    pin: str            # 4-digit account PIN
    plan_id: str
    device: str
    payments_made: int
    data_used_gb: int
    trips: list[Trip] = field(default_factory=list)   # this billing cycle
    past_due: bool = False
    status: str = "active"                          # active | suspended
    last_courtesy_credit: date | None = None

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def plan(self) -> Plan:
        return PLANS[self.plan_id]


# ---- state machines ----------------------------------------------------------
BILL_STATES = ["issued", "due", "paid", "past_due", "suspended"]
BILL_TRANSITIONS = {
    "issued": {"due"}, "due": {"paid", "past_due"},
    "past_due": {"paid", "suspended"}, "suspended": {"paid"}, "paid": set(),
}
CREDIT_STATES = ["requested", "approved", "denied", "escalated"]
CREDIT_TRANSITIONS = {"requested": {"approved", "denied", "escalated"},
                      "approved": set(), "denied": set(), "escalated": {"approved", "denied"}}
ORDER_STATES = ["eligible_check", "ordered", "shipped", "activated", "returned"]
ORDER_TRANSITIONS = {"eligible_check": {"ordered"}, "ordered": {"shipped"},
                     "shipped": {"activated", "returned"}, "activated": set(), "returned": set()}

REASONS = {
    "NOT_ENOUGH_PAYMENTS": "the phone needs more installment payments before it's upgrade eligible",
    "PAST_DUE": "the account has a past-due balance",
    "RECENT_CREDIT": "a courtesy credit was already applied in the last twelve months",
    "OVER_LIMIT": "the amount is above what I can credit on this call",
    "NOT_CREDITABLE": "only roaming and data overage charges are eligible for a courtesy credit",
    "NO_CHARGE": "there is no charge of that type on the current bill",
}
CREDIT_LIMIT = 50.0
TAXES_FEES = 6.75

# acts that disclose customer proprietary network information (CPNI)
CPNI_ACTS = {"answer_bill", "answer_data", "answer_roaming", "answer_upgrade", "answer_credit"}
INTENTS = ["ask_bill", "ask_data", "ask_roaming", "ask_upgrade", "request_credit"]

# ---- rules engine ----------------------------------------------------------------


@dataclass
class RuleSet:
    """Overrides on top of the published rate card / policy book."""
    overrides: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def get(self, key, default):
        return self.overrides.get(key, default)

    def source(self, key) -> str:
        return self.provenance.get(key, "published policy")


PUBLISHED = RuleSet()


@dataclass
class Result:
    ok: bool
    values: dict
    reason: str | None = None
    rules: list[str] = field(default_factory=list)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


class RulesEngine:
    def __init__(self, ruleset: RuleSet | None = None):
        self.rs = ruleset or PUBLISHED

    def _rule(self, key, default):
        v = self.rs.get(key, default)
        return v, f"{key_label(key)} = {v} ({self.rs.source(key)})"

    # roaming ---------------------------------------------------------------
    def roaming_included(self, plan_id: str, zone: int) -> tuple[bool, str]:
        return self._rule(("roaming_included", plan_id, zone), zone in PLANS[plan_id].roaming_zones)

    def roaming(self, acct: Account, country: str, days: int | None) -> Result:
        zone = COUNTRIES[country]
        inc, why = self.roaming_included(acct.plan_id, zone)
        rate = 0.0 if inc else ROAMING_DAY_RATE[zone]
        vals = {"country": country, "zone": zone, "included": inc, "daily_rate": rate,
                "days": days, "total": round(rate * days, 2) if days else None}
        return Result(True, vals, None, [why])

    # data ------------------------------------------------------------------
    def throttle_gb(self, plan_id: str) -> tuple[int | None, str]:
        return self._rule(("throttle_gb", plan_id), PLANS[plan_id].throttle_gb)

    def data(self, acct: Account) -> Result:
        p = acct.plan
        rules = []
        if p.data_gb is not None:
            over = max(0, acct.data_used_gb - p.data_gb)
            vals = {"used": acct.data_used_gb, "allowance": p.data_gb, "unlimited": False,
                    "remaining": max(0, p.data_gb - acct.data_used_gb), "overage_gb": over,
                    "overage": round(math.ceil(over) * p.overage_per_gb, 2), "throttled": False}
            rules.append(f"{p.name} includes {p.data_gb} GB, then {fmt_money(p.overage_per_gb)} per GB (published policy)")
        else:
            thr, why = self.throttle_gb(p.plan_id)
            rules.append(why)
            vals = {"used": acct.data_used_gb, "allowance": None, "unlimited": True, "remaining": None,
                    "overage_gb": 0, "overage": 0.0, "throttled": thr is not None and acct.data_used_gb >= thr}
        vals["cycle_end"] = CYCLE_END.isoformat()
        return Result(True, vals, None, rules)

    # bill ------------------------------------------------------------------
    def bill(self, acct: Account) -> Result:
        p = acct.plan
        items = [{"type": "plan", "label": f"{p.name} plan", "amount": p.price}]
        if acct.payments_made < INSTALLMENTS:
            d = DEVICES[acct.device]
            items.append({"type": "device", "label": f"{d.model} installment {acct.payments_made + 1} of {INSTALLMENTS}",
                          "amount": d.installment})
        rules = []
        dr = self.data(acct)
        if dr.values["overage"]:
            items.append({"type": "overage", "label": f"data overage, {dr.values['overage_gb']} GB",
                          "amount": dr.values["overage"]})
            rules += dr.rules
        for t in acct.trips:
            r = self.roaming(acct, t.country, t.days)
            rules += r.rules
            items.append({"type": "roaming", "label": f"roaming in {t.country}, {t.days} days",
                          "amount": r.values["total"], "included": r.values["included"], "country": t.country})
        items.append({"type": "taxes", "label": "taxes and fees", "amount": TAXES_FEES})
        total = round(sum(i["amount"] for i in items), 2)
        extras = [i for i in items if i["type"] in ("overage", "roaming") and i["amount"] > 0]
        return Result(True, {"items": items, "total": total, "extras": extras,
                             "status": "past_due" if acct.past_due else "due"}, None, rules)

    # upgrade ---------------------------------------------------------------
    def upgrade(self, acct: Account) -> Result:
        need, why = self._rule(("upgrade_after", acct.plan_id), acct.plan.upgrade_after)
        d = DEVICES[acct.device]
        remaining = max(0, INSTALLMENTS - acct.payments_made)
        vals = {"device": d.model, "payments_made": acct.payments_made, "required": need,
                "payoff": round(remaining * d.installment, 2)}
        rules = [why, "no upgrade with a past-due balance (published policy)"]
        if acct.past_due:
            return Result(False, vals, "PAST_DUE", rules)
        if acct.payments_made < need:
            vals["payments_short"] = need - acct.payments_made
            return Result(False, vals, "NOT_ENOUGH_PAYMENTS", rules)
        return Result(True, vals, None, rules)

    # courtesy credit -------------------------------------------------------
    def credit(self, acct: Account, charge_type: str | None) -> Result:
        rules = [f"courtesy credit: roaming or overage only, up to {fmt_money(CREDIT_LIMIT)}, once per 12 months (published policy)"]
        bill = self.bill(acct).values
        if charge_type not in ("roaming", "overage"):
            return Result(False, {"charge_type": charge_type, "amount": 0.0}, "NOT_CREDITABLE", rules)
        amount = round(sum(i["amount"] for i in bill["extras"] if i["type"] == charge_type), 2)
        vals = {"charge_type": charge_type, "amount": amount}
        if amount <= 0:
            return Result(False, vals, "NO_CHARGE", rules)
        if acct.last_courtesy_credit and (TODAY - acct.last_courtesy_credit).days < 365:
            return Result(False, vals, "RECENT_CREDIT", rules)
        if amount > CREDIT_LIMIT:
            return Result(False, vals, "OVER_LIMIT", rules)
        return Result(True, vals, None, rules)


def key_label(key) -> str:
    if key[0] == "roaming_included":
        return f"roaming included on {PLANS[key[1]].name} in zone {key[2]} ({ZONE_NAMES[key[2]]})"
    if key[0] == "throttle_gb":
        return f"{PLANS[key[1]].name} speeds reduced after (GB)"
    if key[0] == "upgrade_after":
        return f"{PLANS[key[1]].name} upgrade after (payments)"
    return str(key)


# ---- dialogue policy (allowed acts per state) ------------------------------------
def allowed_agent_acts(verified: bool, failed: int) -> set[str]:
    base = {"greeting", "request_verification", "verify_ok", "verify_fail", "empathize", "transfer_human",
            "anything_else", "closing", "clarify", "ask_country"}
    if failed >= 2:
        return {"transfer_human", "empathize", "closing"}
    return base | (CPNI_ACTS if verified else set())


def rules_next_act(verified: bool, failed: int, caller_act: str) -> str:
    if caller_act in ("request_human", "request_cancel") or failed >= 2:
        return "transfer_human"
    if caller_act in INTENTS:
        return {"ask_bill": "answer_bill", "ask_data": "answer_data", "ask_roaming": "answer_roaming",
                "ask_upgrade": "answer_upgrade", "request_credit": "answer_credit"}[caller_act] if verified else "request_verification"
    if caller_act == "express_frustration":
        return "empathize"
    if caller_act == "thank_bye":
        return "closing"
    return "anything_else"


# ---- book of business --------------------------------------------------------------
FIRST = "Maria James Aisha Robert Linda Kevin Priya Daniel Sofia Marcus Emily Jorge Hannah Wei Olivia Andre Grace Samuel Nadia Tyler".split()
LAST = "Garcia Johnson Patel Smith Nguyen Brown Williams Kim Lopez Davis Chen Wilson Martinez Clark Lewis Walker Hall Young Allen Scott".split()
AREAS = ["312", "415", "646", "206", "512", "617", "305", "720"]

# Hidden reality: what billing and the network actually do. The agent never sees this;
# stage 3 has to discover it from past calls.
REALITY = RuleSet(
    overrides={("roaming_included", "UNP", 2): True, ("throttle_gb", "UNS"): 50},
    provenance={("roaming_included", "UNP", 2): "carrier promo bulletin (hidden)",
                ("throttle_gb", "UNS"): "network management policy (hidden)"},
)


class BookOfBusiness:
    def __init__(self, n: int = 60, seed: int = 7):
        rng = random.Random(seed)
        self.accounts: dict[str, Account] = {}
        used_phones = set()
        for i in range(n):
            while True:
                phone = f"{rng.choice(AREAS)}-555-01{rng.randint(0, 99):02d}"
                if phone not in used_phones:
                    used_phones.add(phone)
                    break
            plan = rng.choices(list(PLANS), [0.3, 0.4, 0.3])[0]
            used = {"BAS": rng.randint(1, 9), "UNS": rng.randint(5, 95), "UNP": rng.randint(5, 120)}[plan]
            trips = [Trip(rng.choice(list(COUNTRIES)), rng.randint(2, 10))] if rng.random() < 0.25 else []
            past_due = rng.random() < 0.12
            a = Account(f"A{100200 + i}", rng.choice(FIRST), rng.choice(LAST), phone, f"{rng.randint(0, 9999):04d}",
                        plan, rng.choice(list(DEVICES)), rng.randint(0, 24), used, trips, past_due,
                        "suspended" if past_due and rng.random() < 0.3 else "active",
                        TODAY - timedelta(days=rng.randint(30, 700)) if rng.random() < 0.35 else None)
            self.accounts[a.account_id] = a
        # a few accounts that show the hidden policies on live calls
        demo = list(self.accounts.values())
        demo[0].plan_id, demo[0].data_used_gb, demo[0].trips, demo[0].past_due = "UNS", 72, [], False
        demo[1].plan_id, demo[1].trips, demo[1].past_due, demo[1].last_courtesy_credit = "UNP", [Trip("France", 6)], False, None
        demo[2].plan_id, demo[2].data_used_gb, demo[2].past_due, demo[2].last_courtesy_credit = "BAS", 9, False, None
        demo[3].past_due, demo[3].payments_made = True, 20

    def by_phone(self, phone: str) -> Account | None:
        digits = "".join(c for c in phone if c.isdigit())
        return next((a for a in self.accounts.values() if a.phone.replace("-", "") == digits), None)

    def verify(self, phone: str | None, pin: str | None) -> Account | None:
        a = self.by_phone(phone or "")
        return a if a and pin and a.pin == pin else None

    def snapshot(self, acct: Account, **changes) -> Account:
        a = copy.deepcopy(acct)
        for k, v in changes.items():
            setattr(a, k, v)
        return a
