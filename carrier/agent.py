"""Stage 5 - Live script to decision: the call agent.

Each caller turn -> NLU act + entities -> a decision constrained by the world model
(CPNI: no account details before the caller is verified with phone number + PIN) ->
answer from a rules-engine tool, a clarifying question, or a human handoff.
The same response templates are used by the synthetic corpus, so batch and live speak alike.
"""
from __future__ import annotations

import time

from . import llm, nlu
from .world import (CARRIER, CPNI_ACTS, CYCLE_END, DEVICES, INSTALLMENTS, INTENTS, REASONS, ZONE_NAMES,
                    BookOfBusiness, RuleSet, RulesEngine, allowed_agent_acts, fmt_money)

AGENT_NAME = "Amy"


# ---- response templates (shared with the corpus generator) ------------------------
class Say:
    greeting = f"Thanks for calling {CARRIER}, this is {AGENT_NAME}. How can I help you today?"
    request_verification = ("I can help with that. First, can you give me the phone number on the account "
                            "and your four-digit account PIN?")
    ask_pin = "Thanks. And your four-digit account PIN?"
    verify_fail = ("I'm sorry, that doesn't match our records. Could you give me the phone number on the "
                   "account and your four-digit account PIN again?")
    transfer_failed = ("I'm sorry, I still couldn't verify the account. I'm transferring you to a specialist "
                       "who can help. Please hold.")
    transfer = "Of course. I'm transferring you to a specialist now. Please hold."
    transfer_cancel = "I'm transferring you to our account services team, who can help with that. Please hold."
    ask_country = "Which country are you traveling to, and for how many days?"
    ask_charge = "Which charge would you like me to look at, the roaming charge or the data overage?"
    empathize = "I completely understand your frustration, and I'm sorry for the trouble."
    anything_else = "Is there anything else I can help you with today?"
    what_help = "What can I help you with today?"
    closing = f"Thank you for calling {CARRIER}. Have a great day."
    clarify = "Sorry, I didn't catch that. Is this about billing, data, travel, a new phone, or a credit?"

    @staticmethod
    def verify_ok(acct) -> str:
        return f"Thank you, {acct.first}, you're verified. I see you're on {acct.plan.name}."

    @staticmethod
    def data(v: dict) -> str:
        end = "December 1st" if CYCLE_END.month == 12 and CYCLE_END.day == 1 else CYCLE_END.isoformat()
        if not v["unlimited"]:
            if v["overage_gb"]:
                return (f"You've used {v['used']} GB of your {v['allowance']} GB this cycle, so {v['overage_gb']} GB "
                        f"is billed as data overage at $10 per GB, {fmt_money(v['overage'])} so far.")
            return f"You've used {v['used']} GB of your {v['allowance']} GB this cycle, so you have {v['remaining']} GB left until {end}."
        if v["throttled"]:
            return (f"You've used {v['used']} GB this cycle. You're past your priority data, so speeds are reduced "
                    f"until the cycle resets on {end}.")
        return f"You've used {v['used']} GB this cycle, and you're still at full speed."

    @staticmethod
    def roaming(v: dict) -> str:
        if v["included"]:
            return (f"Good news, roaming in {v['country']} is included on your plan at no extra charge, "
                    "so you can use your phone there normally.")
        s = (f"Roaming in {v['country']} isn't included on your plan. The {ZONE_NAMES[v['zone']]} day pass is "
             f"{fmt_money(v['daily_rate'])} per day")
        if v.get("days"):
            s += f", so {v['days']} days would be {fmt_money(v['total'])}"
        return s + ". It only charges on days you use your phone there."

    @staticmethod
    def bill(v: dict, acct) -> str:
        items = {i["type"]: i for i in v["items"]}
        parts = [f"your {acct.plan.name} plan at {fmt_money(acct.plan.price)}"]
        if "device" in items:
            parts.append(f"the {acct.device} installment of {fmt_money(items['device']['amount'])}")
        parts.append(f"taxes and fees of {fmt_money(items['taxes']['amount'])}")
        s = f"Your current bill is {fmt_money(v['total'])}. That's " + ", ".join(parts)
        for i in v["items"]:
            if i["type"] == "overage":
                s += f", plus data overage of {fmt_money(i['amount'])} for {i['label'].split(', ')[1]}"
            elif i["type"] == "roaming" and i["amount"] > 0:
                s += f", plus roaming in {i['country']} for {i['label'].split(', ')[1]} at {fmt_money(i['amount'])}"
        s += "."
        for i in v["items"]:
            if i["type"] == "roaming" and not i["amount"]:
                s += f" Your usage in {i['country']} was included on your plan, so there's no roaming charge."
        return s

    @staticmethod
    def upgrade(r) -> str:
        v = r.values
        if r.ok:
            s = f"Good news, you're eligible to upgrade today. Your {v['device']} has {v['payments_made']} of {INSTALLMENTS} installments paid"
            return s + (f", and the remaining balance of {fmt_money(v['payoff'])} would be due at upgrade." if v["payoff"] else ".")
        if r.reason == "PAST_DUE":
            return "I'm sorry, you can't upgrade right now because the account has a past-due balance. Once that's paid, I can check again."
        return (f"I'm sorry, you're not eligible to upgrade yet. Your {v['device']} has {v['payments_made']} of "
                f"{INSTALLMENTS} installments paid and your plan needs {v['required']}. You can upgrade early if you "
                f"pay off the remaining balance of {fmt_money(v['payoff'])}.")

    @staticmethod
    def credit(r) -> str:
        v = r.values
        kind = "roaming" if v["charge_type"] == "roaming" else "data overage"
        if r.ok:
            return f"I've applied a one-time courtesy credit of {fmt_money(v['amount'])} for the {kind} charges. You'll see it on your next bill."
        if r.reason == "OVER_LIMIT":
            return (f"That charge is {fmt_money(v['amount'])}, which is above what I can credit on this call, so I'm "
                    "transferring you to a billing specialist who can review it. Please hold.")
        return f"I'm sorry, I can't credit that because {REASONS[r.reason]}."


class CallSession:
    def __init__(self, bob: BookOfBusiness | None = None, ruleset: RuleSet | None = None, use_llm: bool = False):
        self.bob = bob or BookOfBusiness()
        self.ruleset = ruleset or RuleSet()
        self.engine = RulesEngine(self.ruleset)
        self.use_llm = use_llm and llm.available()
        self.acct = None
        self.failed = 0
        self.phone: str | None = None
        self.pending: str | None = None
        self.slots: dict = {}
        self.mood = "neutral"
        self.turns: list[dict] = []
        self.decisions: list[dict] = []
        self.facts: list[dict] = []
        self.outcome: str | None = None
        self.ended = False
        self.started = time.time()
        self.frustrated_ack = False

    # -----------------------------------------------------------------
    def _agent(self, act: str, text: str):
        if act not in allowed_agent_acts(self.acct is not None, self.failed) and act in CPNI_ACTS:
            act, text = "request_verification", Say.request_verification   # hard CPNI guard
        self.turns.append({"role": "agent", "act": act, "text": text, "t": round(time.time() - self.started, 2)})
        return act, text

    def greet(self) -> dict:
        act, text = self._agent("greeting", Say.greeting)
        return {"reply": text, "acts": [act], "tools": [], "rules": [], "mood": self.mood}

    def caller_turn(self, raw_text: str, prosody: dict | None = None) -> dict:
        text, fixes = nlu.correct_terms(raw_text)
        act = nlu.classify_caller(text)
        nlu_src = "rules"
        if act == "other" and self.use_llm:
            act = llm.classify_act(text) or act
            nlu_src = "llm"
        self.mood = nlu.detect_mood(text, prosody) if self.mood != "frustrated" or act != "confirm" else self.mood
        self.turns.append({"role": "caller", "act": act, "text": text, "text_raw": raw_text,
                           "t": round(time.time() - self.started, 2), "mood": self.mood})
        for k, f in (("country", nlu.find_country), ("days", nlu.find_days),
                     ("device", nlu.find_device), ("charge_type", nlu.find_charge_type)):
            v = f(text)
            if v:
                self.slots[k] = v
        if act in INTENTS:
            self.pending = act
        out: list[tuple[str, str]] = []
        tools: list[dict] = []
        rules: list[str] = []

        def say(a, t):
            out.append(self._agent(a, t))

        if self.ended:
            say("closing", Say.closing)
        elif act in ("request_human", "request_cancel"):
            say("transfer_human", Say.transfer_cancel if act == "request_cancel" else Say.transfer)
            tools.append({"tool": "transfer", "args": {"queue": "account services" if act == "request_cancel" else "specialist"}})
            self.outcome, self.ended = "escalated", True
        elif self.acct is None and (act == "give_identity" or nlu.has_identity(text)):
            self.phone = nlu.find_phone(text) or self.phone
            pin = nlu.find_pin(text.replace(self.phone.replace("-", ""), "") if self.phone else text)
            if self.phone and pin:
                acct = self.bob.verify(self.phone, pin)
                tools.append({"tool": "verify_account", "args": {"phone": self.phone, "pin": "****"},
                              "result": "verified" if acct else "no match"})
                if acct:
                    self.acct = acct
                    say("verify_ok", Say.verify_ok(acct))
                    if self.pending:
                        self._answer(say, tools, rules)
                    else:
                        say("anything_else", Say.what_help)
                else:
                    self.failed += 1
                    self.phone = None
                    if self.failed >= 2:
                        say("transfer_human", Say.transfer_failed)
                        tools.append({"tool": "transfer", "args": {"queue": "specialist", "why": "verification failed twice"}})
                        self.outcome, self.ended = "escalated", True
                    else:
                        say("verify_fail", Say.verify_fail)
            elif self.phone:
                say("request_verification", Say.ask_pin)
            else:
                say("request_verification", Say.request_verification)
        elif act in INTENTS or (self.pending and act == "other" and self.slots):
            if act == "express_frustration" or self.mood == "frustrated" and not self.frustrated_ack:
                say("empathize", Say.empathize)
                self.frustrated_ack = True
            if self.acct is None:
                say("request_verification", Say.request_verification)
            else:
                self._answer(say, tools, rules)
        elif act == "express_frustration":
            say("empathize", Say.empathize)
            self.frustrated_ack = True
            if self.acct is None and self.pending:
                say("request_verification", Say.request_verification)
            else:
                say("anything_else", "If you'd like a specialist, just let me know. " + Say.anything_else)
        elif act == "thank_bye":
            say("closing", Say.closing)
            self.outcome = self.outcome or ("resolved" if self.facts else "no_request")
            self.ended = True
        elif act == "confirm":
            if self.pending and self.acct is not None and self.slots.get("country") and self.pending == "ask_roaming":
                self._answer(say, tools, rules)
            else:
                say("anything_else", Say.anything_else)
        else:
            say("clarify", Say.clarify)
        self.decisions.append({"turn": len(self.turns), "caller_act": act, "nlu": nlu_src, "acts": [a for a, _ in out],
                               "tools": tools, "rules": rules, "term_fixes": fixes, "mood": self.mood})
        return {"reply": " ".join(t for _, t in out), "acts": [a for a, _ in out], "tools": tools, "rules": rules,
                "mood": self.mood, "caller_act": act, "term_fixes": fixes, "ended": self.ended,
                "verified": self.acct is not None}

    # -----------------------------------------------------------------
    def _answer(self, say, tools, rules):
        a, intent, eng = self.acct, self.pending, self.engine
        fact = {"intent": intent, "plan_id": a.plan_id}
        if intent == "ask_data":
            r = eng.data(a)
            tools.append({"tool": "get_data_usage", "args": {"account": a.account_id}, "result": r.values})
            say("answer_data", Say.data(r.values))
            fact |= {"used": r.values["used"], "throttled": r.values["throttled"], "overage": r.values["overage"]}
        elif intent == "ask_roaming":
            if not self.slots.get("country"):
                say("ask_country", Say.ask_country)
                return
            r = eng.roaming(a, self.slots["country"], self.slots.get("days"))
            tools.append({"tool": "quote_roaming", "args": {"country": r.values["country"], "days": r.values["days"]},
                          "result": r.values})
            say("answer_roaming", Say.roaming(r.values))
            fact |= {"country": r.values["country"], "zone": r.values["zone"], "roaming_included": r.values["included"]}
        elif intent == "ask_bill":
            r = eng.bill(a)
            tools.append({"tool": "get_bill", "args": {"account": a.account_id}, "result": r.values})
            say("answer_bill", Say.bill(r.values, a))
            fact |= {"total": r.values["total"],
                     "roaming": [{"country": i["country"], "zone": nlu.COUNTRIES[i["country"]], "included": i["included"]}
                                 for i in r.values["items"] if i["type"] == "roaming"]}
        elif intent == "ask_upgrade":
            r = eng.upgrade(a)
            tools.append({"tool": "check_upgrade", "args": {"account": a.account_id}, "result": r.values | {"eligible": r.ok}})
            say("answer_upgrade", Say.upgrade(r))
            fact |= {"eligible": r.ok, "reason": r.reason}
        elif intent == "request_credit":
            ctype = self.slots.get("charge_type")
            if ctype not in ("roaming", "overage"):
                extras = {i["type"] for i in eng.bill(a).values["extras"]}
                if len(extras) == 1:
                    ctype = extras.pop()
                elif extras:
                    say("clarify", Say.ask_charge)
                    return
            r = eng.credit(a, ctype)
            tools.append({"tool": "apply_credit", "args": {"charge_type": ctype}, "result": r.values | {"approved": r.ok, "reason": r.reason}})
            if r.reason == "OVER_LIMIT":
                say("transfer_human", Say.credit(r))
                self.outcome, self.ended = "escalated", True
            else:
                say("answer_credit", Say.credit(r))
            fact |= {"approved": r.ok, "reason": r.reason, "amount": r.values["amount"]}
        rules += r.rules
        self.facts.append(fact)
        self.pending = None
        self.slots = {}
        if not self.ended:
            say("anything_else", Say.anything_else)

    def to_record(self) -> dict:
        return {"turns": self.turns, "decisions": self.decisions, "facts": self.facts,
                "outcome": self.outcome or ("resolved" if self.facts else "abandoned"),
                "verified": self.acct is not None, "account_id": self.acct.account_id if self.acct else None,
                "plan_id": self.acct.plan_id if self.acct else None,
                "rule_overrides": [[list(k), v, self.ruleset.source(k)] for k, v in self.ruleset.overrides.items()]}
