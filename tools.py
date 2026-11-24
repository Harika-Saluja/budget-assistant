"""tools.py - the model/system boundary (Lecture 12).

The model may *propose* an action by naming a tool and giving arguments. This module defines the tool's
schema and validates every proposal before the app shows it to the person. Nothing here changes a budget:
the person must confirm in the UI, and the change is simulated.
"""
from __future__ import annotations

MAX_BUDGET = 5000.0
TOOL_NAME = "propose_budget_change"


def propose_budget_tool(categories) -> dict:
    """JSON schema in the shape the Anthropic Messages API expects for a tool definition."""
    return {
        "name": TOOL_NAME,
        "description": ("Propose changing the monthly budget for one category. This changes nothing by itself: "
                        "the app shows the proposal and the user must confirm."),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": sorted(categories)},
                "amount": {"type": "number", "minimum": 0, "description": "new monthly budget in pounds"},
            },
            "required": ["category", "amount"],
        },
    }


def _result(ok, checks, call=None, warning=""):
    return {"ok": ok, "checks": checks, "call": call, "warning": warning}


def validate_tool_call(name, args, budgets) -> dict:
    """Return {ok, checks, call, warning}. `checks` is a list of {label, ok, detail} for display."""
    checks = []

    def add(label, ok, detail=""):
        checks.append({"label": label, "ok": bool(ok), "detail": detail})

    add("The tool exists", name == TOOL_NAME, str(name))
    if name != TOOL_NAME:
        return _result(False, checks)

    args = args if isinstance(args, dict) else {}
    category, amount = args.get("category"), args.get("amount")
    shaped = (isinstance(category, str) and isinstance(amount, (int, float))
              and not isinstance(amount, bool))
    add("The arguments match the schema", shaped, repr(args)[:80])
    if not shaped:
        return _result(False, checks)

    category = category.strip().lower()
    known = category in budgets
    add("The category is one the app knows", known, category)
    in_range = 0 < amount <= MAX_BUDGET
    add("The amount is in a sensible range", in_range, f"£{amount:g}")
    ok = known and in_range
    add("It needs the person's confirmation before anything changes", True, "always")
    if not ok:
        return _result(False, checks)

    call = {"category": category, "amount": float(amount)}
    old = budgets[category]
    warning = ""
    if old and (amount >= 2 * old or amount <= old / 2):
        warning = f"That is a big change (£{old:.0f} to £{amount:.0f}). Is that what you meant?"
    return _result(True, checks, call, warning)
