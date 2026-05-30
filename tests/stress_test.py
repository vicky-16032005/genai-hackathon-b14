"""Stress test for the eligibility engine — runs a grid of citizen profiles and
asserts that every returned scheme actually satisfies its own rules. Flags any
violation (income above ceiling, wrong gender, out-of-age-band, wrong state).

Run:  python -m tests.stress_test
"""
import json
import itertools
from pathlib import Path

from src import config, eligibility

SCHEMES = {s["scheme_name"]: s for s in
           json.loads(Path(config.SCHEMES_JSON).read_text())["schemes"]}

AGES = [5, 17, 18, 30, 40, 41, 50, 60, 70, 95]
INCOMES = [0, 50_000, 250_000, 600_000, 2_000_000, 20_000_000]
GENDERS = ["Any", "Female", "Male"]
STATES = ["All India", "Karnataka", "Maharashtra"]
INTEREST_SETS = [[], eligibility.ALL_INTERESTS, ["Housing"], ["Street vendor"]]

violations = []


def check_invariants(profile, results):
    age, income, gender, state, _interests = profile
    for r in results:
        s = SCHEMES[r["scheme_name"]]
        el = s["eligibility"]
        tags = set(el.get("target_groups", []))
        child = "girl_child" in tags
        name = r["scheme_name"]

        # income ceiling
        cap = el.get("max_annual_income")
        if cap is not None and income > cap:
            violations.append(f"INCOME: {name} (cap {cap}) shown to income {income}")
        # gender
        sg = el.get("gender", "any")
        if sg in ("female", "male") and gender != "Any" and gender.lower() != sg:
            violations.append(f"GENDER: {name} ({sg}) shown to {gender}")
        # age band (child schemes apply age to the child, not the applicant)
        if not child:
            if el.get("min_age") is not None and age < el["min_age"]:
                violations.append(f"AGE<min: {name} (min {el['min_age']}) shown to age {age}")
            if el.get("max_age") is not None and age > el["max_age"]:
                violations.append(f"AGE>max: {name} (max {el['max_age']}) shown to age {age}")
        # state: a state-only scheme must match the user's state
        states = s.get("applicable_states", [])
        if "All India" not in states:
            if state == "All India" or state not in states:
                violations.append(f"STATE: {name} ({states}) shown to state '{state}'")


n = 0
for profile in itertools.product(AGES, INCOMES, GENDERS, STATES, INTEREST_SETS):
    age, income, gender, state, interests = profile
    res = eligibility.check(age=age, annual_income=income, state=state,
                            gender=gender, interests=interests)
    check_invariants(profile, res)
    n += 1

print(f"Ran {n} profiles across the eligibility engine.")
if violations:
    print(f"\n❌ {len(violations)} INVARIANT VIOLATIONS (showing unique):")
    for v in sorted(set(violations))[:40]:
        print("  -", v)
else:
    print("\n✅ No violations — every returned scheme satisfies income/gender/age/state rules.")
