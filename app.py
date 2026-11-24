"""app.py - Streamlit chat UI for the budgeting-assistant prototype (v2).

Run with:  streamlit run app.py
All data is synthetic unless you upload your own CSV. Not financial advice.
"""
import os
from datetime import date

import streamlit as st


def _load_secrets_into_env():
    """Streamlit Cloud keeps keys in st.secrets. Copy the ones we use into the environment, where core.py reads them.
    Nothing is written to disk and no key is stored in the repo."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BUDGET_MODEL"):
        if os.getenv(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:        # no secrets file, e.g. when running locally
            value = None
        if value:
            os.environ[name] = str(value).strip()


_load_secrets_into_env()

from core import (DEFAULT_BUDGETS, DESIGN_PROFILE, MEMORY_KINDS, QUICK_QUESTIONS, SPINE_MAP, TASK_SPEC, WELCOME,
                  MemoryStore, answer, classify_memory_text, compute_alerts, effective_transactions, is_stale,
                  llm_mode, make_transactions, new_memory, outcome_note, parse_budget_command, parse_remember,
                  parse_transactions_csv, propose_inferences, suggest_categories, transactions_to_csv)
from tools import validate_tool_call

st.set_page_config(page_title="Budget assistant (prototype)", page_icon="💷", layout="wide")

MODE_NOTE = {
    "live": "Live model reply.",
    "replay": "Replayed from a recorded live run (no API key needed).",
    "demo": "Demo reply: no API key and no recording, so this is rule-based.",
    "declined": "Declined: outside what this prototype can answer.",
    "nothing_found": "Nothing found in the data, so I didn't guess.",
}


def init_state():
    ss = st.session_state
    if "store" not in ss:
        ss.store = MemoryStore()
        ss.base_tx = make_transactions()
        ss.upload_id = None
        ss.budgets = dict(DEFAULT_BUDGETS)
        ss.messages = [{"role": "assistant", "text": WELCOME, "mode": "action"}]
        ss.pending = None          # {category, amount, warning} awaiting confirmation
        ss.clarify = None          # {amount, options} awaiting a category choice (a repair turn)
        ss.dismissed = set()       # topics the user said no to, or forgot
        ss.last_result = None
        ss.budget_history = []     # (category, old amount) for undo


init_state()
ss = st.session_state
store = ss.store


def tx_now():
    """Transactions as the assistant sees them: one-off notes remove their transaction."""
    return effective_transactions(ss.base_tx, store)


def say(role, text, **extra):
    ss.messages.append({"role": role, "text": text, **extra})


# ----------------------------------------------------------------------------- chat
def handle(prompt: str):
    say("user", prompt)
    note = parse_remember(prompt)
    cmd = parse_budget_command(prompt)
    if note:
        store.add(new_memory(classify_memory_text(note), note, "stated", f"You told me on {date.today():%d %b %Y}."))
        st.toast("Saved to memory")
        say("assistant", f"Noted. I'll remember: “{note}”. You can edit or forget it in the panel on the right.",
            mode="action")
    elif cmd and cmd["category"] in ss.budgets:
        verdict = validate_tool_call("propose_budget_change", cmd, ss.budgets)
        if verdict["ok"]:
            ss.pending = {**verdict["call"], "warning": verdict["warning"]}
            say("assistant", "I can propose that change. Nothing has changed yet: please confirm or cancel below.",
                mode="action")
        else:
            say("assistant", "I couldn't turn that into a valid budget change.", mode="action")
    elif cmd:
        ss.clarify = {"amount": cmd["amount"], "options": suggest_categories(cmd["raw"], ss.budgets)}
        say("assistant", f"I don't have a budget called “{cmd['raw']}”. Which of these did you mean?", mode="action")
    else:
        history = [(m["role"], m["text"]) for m in ss.messages[:-1]]
        with st.spinner("Working on it…"):
            res = answer(prompt, store, ss.base_tx, ss.budgets, history)
        ss.last_result = res
        if res["proposal"]:
            ss.pending = res["proposal"]
        say("assistant", res["text"], result=res)


def source_label(m):
    return "🗣️ You said" if m.source == "stated" else "🔍 My guess"


def render_result_details(res):
    used = res["memories_used"]
    if used:
        with st.expander(f"Memory used in this reply ({len(used)})"):
            for m in used:
                st.markdown(f"**{source_label(m)}** · {m.kind} · {m.text}")
                st.caption(f"{m.evidence} (shown as it was when used)")
    if res["evidence"]:
        with st.expander(f"Evidence I looked at ({len(res['evidence'])} payments)"):
            for t in res["evidence"]:
                st.write(f"{t['date']} · {t['merchant']} · £{t['amount']:.2f} · {t['category']}")
    applicable = [c for c in res["checks"] if c["ok"] is not None]
    if applicable:
        passed = sum(1 for c in applicable if c["ok"])
        with st.expander(f"Checks on this reply ({passed}/{len(applicable)} passed)"):
            for c in res["checks"]:
                icon = "➖" if c["ok"] is None else ("✅" if c["ok"] else "⚠️")
                st.write(f"{icon} {c['label']}: {c['detail']}")
    caption = MODE_NOTE.get(res["mode"], "")
    if res["latency"]:
        caption += f" Took {res['latency']:.1f}s."
    if caption:
        st.caption(caption)
    if res["stale_unused"]:
        st.caption(f"{res['stale_unused']} older note(s) were not used because they may be out of date "
                   "(see the Memory tab).")


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg.get("result"):
            render_result_details(msg["result"])


def render_pending():
    if not ss.pending:
        return
    cat, amount, warning = ss.pending["category"], ss.pending["amount"], ss.pending.get("warning", "")
    old = ss.budgets[cat]
    with st.chat_message("assistant"):
        st.write(f"**Proposed change (simulated):** {cat} budget £{old:.0f} → £{amount:.0f}.")
        if warning:
            st.warning(warning)
        a, b = st.columns(2)
        if a.button("Confirm", key="pend-ok"):
            ss.budgets[cat] = amount
            ss.budget_history.append((cat, old))
            ss.pending = None
            say("assistant", f"Done: your {cat} budget is now £{amount:.0f} (was £{old:.0f}). "
                             f"{outcome_note(cat, amount, tx_now())} This is a simulated budget; no money moved. "
                             "You can undo it from the sidebar.", mode="action")
            st.rerun()
        if b.button("Cancel", key="pend-no"):
            ss.pending = None
            say("assistant", f"Cancelled. Your {cat} budget is still £{old:.0f}.", mode="action")
            st.rerun()


def render_clarify():
    if not ss.clarify:
        return
    with st.chat_message("assistant"):
        st.write("Pick one, or ignore this:")
        cols = st.columns(len(ss.clarify["options"]) + 1)
        for i, cat in enumerate(ss.clarify["options"]):
            if cols[i].button(cat, key=f"clar-{cat}"):
                verdict = validate_tool_call("propose_budget_change",
                                             {"category": cat, "amount": ss.clarify["amount"]}, ss.budgets)
                ss.clarify = None
                if verdict["ok"]:
                    ss.pending = {**verdict["call"], "warning": verdict["warning"]}
                    say("assistant", "Thanks. Please confirm or cancel below.", mode="action")
                st.rerun()
        if cols[-1].button("Neither", key="clar-none"):
            ss.clarify = None
            say("assistant", "OK, nothing changed.", mode="action")
            st.rerun()


# ----------------------------------------------------------------------------- memory tab
def render_memory(m):
    stale = is_stale(m)
    with st.container(border=True):
        st.markdown(f"**{source_label(m)}** · {m.kind}")
        st.write(m.text)
        status = ""
        if stale:
            status = " ⚠️ May be out of date, so I'm not using it until you confirm."
        elif m.expires:
            status = f" Expires {m.expires}."
        st.caption(f"{m.evidence} Saved {m.created}.{status}")
        c1, c2, c3 = st.columns(3)
        if m.source == "inferred" and c1.button("Still true", key=f"ok-{m.id}"):
            store.confirm(m.id)
            st.rerun()
        with c2.popover("Edit"):
            new_text = st.text_input("Correct it", value=m.text, key=f"txt-{m.id}")
            if st.button("Save", key=f"save-{m.id}"):
                store.edit(m.id, new_text)
                st.toast("Memory corrected")
                st.rerun()
        if c3.button("Forget", key=f"del-{m.id}"):
            if m.tag:
                ss.dismissed.add(m.tag)
            store.forget(m.id)
            st.toast("Forgotten")
            st.rerun()


def memory_tab():
    st.subheader("What I remember about you")
    st.caption("🗣️ = you told me. 🔍 = my guess from your transactions; guesses expire after 30 days unless you "
               "confirm them. I only use up to four notes per reply.")
    if not store.items:
        st.info("Nothing remembered yet.")
    for m in list(store.items):
        render_memory(m)
    held = {m.tag for m in store.items if m.tag}
    proposals = [p for p in propose_inferences(tx_now()) if p["tag"] not in ss.dismissed and p["tag"] not in held]
    if proposals:
        st.subheader("Things I could remember")
        st.caption("I only store these if you say yes.")
    for p in proposals:
        with st.container(border=True):
            st.write(p["text"])
            st.caption(p["evidence"])
            y, n = st.columns(2)
            if y.button("Remember", key=f"yes-{p['tag']}"):
                store.add(new_memory(p["kind"], p["text"], "inferred", p["evidence"], tag=p["tag"]))
                st.toast("Saved as a guess")
                st.rerun()
            if n.button("No thanks", key=f"no-{p['tag']}"):
                ss.dismissed.add(p["tag"])
                st.rerun()
    with st.expander("Derived profile summary (cleared whenever memory changes)"):
        st.write(store.summary())
    with st.expander("What the four kinds of memory mean"):
        for kind, text in MEMORY_KINDS.items():
            st.write(f"**{kind}**: {text}")
    if store.items:
        sure = st.checkbox("I want to forget everything", key="sure-all")
        if st.button("Forget everything", disabled=not sure):
            ss.dismissed |= {m.tag for m in store.items if m.tag}
            store.forget_all()
            st.rerun()


# ----------------------------------------------------------------------------- alerts tab (signal detection, L4)
def alerts_tab():
    st.subheader("Budget alerts")
    st.caption("An alert fires when last month is over budget by more than the percentage below. "
               "Lower = more real overspending caught, but more false alarms. Higher = fewer false alarms, "
               "but real overspending can be missed.")
    sens = st.slider("Alert sensitivity: alert when over budget by more than (%)", min_value=0, max_value=100,
                     value=0, step=5, key="sens")
    alerts = compute_alerts(tx_now(), ss.budgets, sens)
    if not alerts:
        st.success("No alerts at this setting.")
    for a in alerts:
        with st.container(border=True):
            st.markdown(f"⚠️ **{a['category'].capitalize()}**: £{a['total']:.0f} in {a['month']} vs budget "
                        f"£{a['budget']:.0f} (over by £{a['over_by']:.0f})")
            t = a["top"]
            st.caption(f"Biggest payment: {t['merchant']} £{t['amount']:.0f} on {t['date']}.")
            if st.button("Mark that payment as a one-off", key=f"oneoff-{t['id']}"):
                store.add(new_memory("episodic",
                                     f"{t['merchant']} £{t['amount']:.0f} on {t['date']} was a one-off; "
                                     "leave it out of averages and alerts.",
                                     "stated", f"You marked this as a one-off on {date.today():%d %b %Y}.",
                                     ref=t["id"]))
                st.toast("Marked as a one-off")
                st.rerun()
    st.caption("How many alerts each setting would raise right now:")
    levels = [0, 25, 50, 100]
    st.table({"Sensitivity %": levels, "Alerts": [len(compute_alerts(tx_now(), ss.budgets, s)) for s in levels]})


# ----------------------------------------------------------------------------- design notes tab (L1, L2, L9)
def design_tab():
    st.subheader("Design notes")
    st.caption("How this prototype maps to the module's ideas.")
    st.markdown("**Four variables (Lecture 1)**")
    st.table(DESIGN_PROFILE)
    st.markdown("**The spine of layers (Lecture 2)**")
    st.table(SPINE_MAP)
    with st.expander("Task specification sent to the model (Lecture 9)"):
        for name, text in TASK_SPEC.items():
            st.markdown(f"**{name}**")
            st.text(text)


# ----------------------------------------------------------------------------- layout
left, right = st.columns([3, 2], gap="large")

with left:
    st.title("Budget assistant")
    st.caption("Prototype. Synthetic data. Not financial advice. It never moves money. Try the buttons below, "
               "“remember I'm saving for a laptop”, or “set eating out budget to 150”.")
    for msg in ss.messages:
        render_message(msg)
    render_pending()
    render_clarify()
    qcols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        if qcols[i % 2].button(q, key=f"quick-{i}"):
            handle(q)
            st.rerun()
    prompt = st.chat_input("Ask about your spending…")
    if prompt:
        handle(prompt)
        st.rerun()

with right:
    tab_mem, tab_alerts, tab_design = st.tabs(["Memory", "Alerts", "Design notes"])
    with tab_mem:
        memory_tab()
    with tab_alerts:
        alerts_tab()
    with tab_design:
        design_tab()

with st.sidebar:
    st.header("Under the hood")
    st.write(f"Mode: **{llm_mode()}**")
    with st.expander("Key check (no key values shown)"):
        st.caption("App version: v2.1")
        try:
            secret_names = [str(k) for k in st.secrets.keys()]
        except Exception as exc:
            secret_names = []
            st.caption(f"Could not read secrets: {type(exc).__name__}")
        st.caption(f"Secret names the app can see: {secret_names or 'none'}")
        st.caption(f"OPENAI_API_KEY in environment: {bool(os.getenv('OPENAI_API_KEY'))}")
        st.caption(f"ANTHROPIC_API_KEY in environment: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    res = ss.last_result
    st.caption("The seven stages of the last model call (Lecture 9):")
    if res and res["stages"]:
        st.table([{"Stage": s[0], "What happened": s[1]} for s in res["stages"]])
        if res["tool_checks"]:
            st.caption("Checks on the model's proposed action (Lecture 12):")
            st.table([{"Check": c["label"], "Passed": "yes" if c["ok"] else "NO"} for c in res["tool_checks"]])
        st.caption("What was sent (counts only):")
        st.json(res["record"])
    else:
        st.info("Ask a spending question first.")

    st.subheader("Simulated budgets")
    st.write(ss.budgets)
    if ss.budget_history and st.button("Undo last budget change"):
        cat, old = ss.budget_history.pop()
        ss.budgets[cat] = old
        say("assistant", f"Undone: your {cat} budget is back to £{old:.0f}.", mode="action")
        st.rerun()

    st.subheader("Data")
    st.caption("Synthetic by default. You can upload a CSV with columns date, merchant, category, amount. "
               "Don't upload real bank data to a public deployment.")
    st.download_button("Download the synthetic sample as CSV", transactions_to_csv(make_transactions()),
                       file_name="sample_transactions.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    if uploaded is not None and ss.upload_id != (uploaded.name, uploaded.size):
        rows, errors = parse_transactions_csv(uploaded.getvalue().decode("utf-8", errors="replace"))
        ss.upload_id = (uploaded.name, uploaded.size)
        if rows:
            ss.base_tx = rows
            st.success(f"Loaded {len(rows)} rows.")
        for e in errors[:5]:
            st.warning(e)
    if st.button("Use the synthetic data again"):
        ss.base_tx = make_transactions()
        st.rerun()
    with st.expander("Transactions the assistant is using"):
        st.dataframe(tx_now(), use_container_width=True)