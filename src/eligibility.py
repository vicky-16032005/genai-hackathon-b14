"""Eligibility checker: match a citizen profile to schemes (PS-SC4 deliverable 3).

Deterministic rule-based filtering over the structured scheme metadata, so
results are exact and verifiable — no LLM hallucination. Every matched scheme
is returned with its official link and source as a citation.
"""
import json
from functools import lru_cache
from pathlib import Path

from . import config

# Map friendly UI life-situations to the corpus's target_group tags / categories.
INTEREST_MAP = {
    "Housing": {"tags": {"general"}, "cats": {"Housing"}},
    "Health insurance": {"tags": set(), "cats": {"Health Insurance"}},
    "Business / Self-employment": {"tags": {"entrepreneur"}, "cats": set()},
    "Street vendor": {"tags": {"street_vendor"}, "cats": set()},
    "Artisan / craftsperson": {"tags": {"artisan"}, "cats": set()},
    "Unorganised / gig worker": {"tags": {"unorganised_worker"}, "cats": set()},
    "Senior citizen": {"tags": {"senior_citizen"}, "cats": set()},
    "Girl child / daughter": {"tags": {"girl_child"}, "cats": set()},
    "Woman head of family": {"tags": {"woman_head"}, "cats": set()},
    "Insurance / Pension": {"tags": set(), "cats": {"Insurance / Social Security", "Pension / Social Security", "Pension / Social Welfare"}},
    "Food security": {"tags": set(), "cats": {"Food Security"}},
    "General / not sure": {"tags": {"general"}, "cats": set()},
}

ALL_INTERESTS = list(INTEREST_MAP.keys())


@lru_cache(maxsize=1)
def _load():
    return json.loads(Path(config.SCHEMES_JSON).read_text())["schemes"]


def check(age: int | None, annual_income: int | None, state: str = "All India",
          gender: str = "Any", interests: list[str] | None = None) -> list[dict]:
    interests = interests or []
    want_tags, want_cats = set(), set()
    for it in interests:
        m = INTEREST_MAP.get(it, {})
        want_tags |= m.get("tags", set())
        want_cats |= m.get("cats", set())

    results = []
    for s in _load():
        el = s.get("eligibility", {})
        tags = set(el.get("target_groups", []))
        reasons, fails = [], []

        # ---- hard filters ----
        # State gate: national ("All India") schemes show everywhere; a
        # state-specific scheme shows ONLY when the user picked that exact state.
        states = s.get("applicable_states", [])
        if "All India" not in states:
            if state == "All India" or state not in states:
                continue  # state-specific scheme; user's state doesn't match
            reasons.append(f"Available in {state}")

        sg = el.get("gender", "any")
        if sg in ("female", "male") and gender != "Any" and gender.lower() != sg:
            continue  # gender-specific scheme not for this applicant
        if sg == "female":
            reasons.append("For women")

        is_child_scheme = "girl_child" in tags
        if age is not None and not is_child_scheme:
            if el.get("min_age") is not None and age < el["min_age"]:
                fails.append(f"minimum age {el['min_age']}")
            if el.get("max_age") is not None and age > el["max_age"]:
                fails.append(f"maximum age {el['max_age']}")
        if fails:
            continue
        if el.get("min_age") is not None or el.get("max_age") is not None:
            lo, hi = el.get("min_age"), el.get("max_age")
            if not is_child_scheme:
                reasons.append("Age " + (f"{lo}-{hi}" if hi else f"{lo}+") + " — you qualify")

        inc = el.get("max_annual_income")
        if inc is not None and annual_income is not None:
            if annual_income > inc:
                continue  # income above the ceiling
            reasons.append(f"Income within ceiling (<= Rs {inc:,}/yr)")

        # ---- relevance score (soft) ----
        score = 1
        if tags & want_tags:
            score += 3 * len(tags & want_tags)
            reasons.append("Matches your selected need")
        if s.get("category") in want_cats:
            score += 3
            reasons.append("Matches your selected need")
        if is_child_scheme and "Girl child / daughter" not in interests:
            score -= 2  # de-prioritise unless explicitly requested

        results.append({
            "scheme_name": s["scheme_name"],
            "category": s.get("category", ""),
            "level": s.get("level", ""),
            "benefits": s.get("benefits", ""),
            "eligibility_text": s.get("eligibility_text", ""),
            "documents_needed": s.get("documents_needed", []),
            "application_link": s.get("application_link", ""),
            "source": s.get("source", ""),
            "why": list(dict.fromkeys(reasons)),
            "_score": score,
        })

    results.sort(key=lambda r: r["_score"], reverse=True)
    return results
