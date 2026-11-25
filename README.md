# Budget assistant: a memory you can inspect, correct and forget

README prose word count: **about 1330** (code and commands not counted)

Coursework prototype for *Intelligent Interactive Systems* (University of Birmingham). A chat assistant answers questions about **synthetic** spending data and remembers things about the user. Its memory sits in a panel beside the chat, where each note can be confirmed, edited or forgotten. It is not financial advice and never moves money.

- Repository: https://github.com/Harika-Saluja/budget-assistant
- Live app: [PASTE STREAMLIT APP LINK]
- Submitted commit: [PASTE COMMIT HASH]

## Project status

| Feature | Status |
|---|---|
| Chat over synthetic transactions; optional CSV upload | Implemented |
| Memory panel: stated vs guessed, evidence, date, edit, forget, "still true"; four memory kinds | Implemented |
| Guesses need consent, expire after 30 days, and are not used when stale | Implemented |
| "One-off" notes change later answers and alerts; forgetting the note restores them | Implemented |
| Evidence rows and notes shown for each reply; "nothing found" is not guessed | Implemented |
| Context assembly with a record of what was sent (at most 4 notes) | Implemented |
| Six-part task specification; seven-stage panel of the last model call | Implemented |
| Reply checks (figures match data, length, guess flagged, no false claim of action) | Implemented |
| Out-of-scope questions declined in code; clarify, confirm, cancel, undo | Implemented |
| Alert sensitivity slider (false alarms vs misses) | Implemented |
| Budget change: validated, warned, confirmed, undone | Simulated (no real accounts) |
| Live model (OpenAI path used; Anthropic path with a budget tool also coded) | Implemented, but see "Known limits" |
| Embeddings or other derived copies of memory | Planned |
| Real bank connection | Planned, out of scope |

**Known limits.** In my demonstration runs every live OpenAI call returned HTTP 429 (Too Many Requests), so every recorded reply came from the rule-based fallback. The interface mechanisms were checked; the language model's own behaviour (for example sycophancy and refusal wording) was not. The Anthropic tool-proposal path was not run live. Note selection uses crude word overlap.

## Technology stack

- Python 3.10+. Streamlit for the interface. The core logic (`core.py`, `checks.py`, `tools.py`) uses only the standard library.
- Storage: local `memory.json` (git-ignored).
- Model: optional. The app calls the OpenAI API (set `OPENAI_API_KEY`, optional `BUDGET_MODEL`) or the Anthropic API (`ANTHROPIC_API_KEY`). The model only writes reply text; code handles commands, retrieval, memory, alerts and validation.
- Without a key the app replays recorded replies from `recordings/` where they exist, otherwise gives a labelled rule-based reply.
- Undecided: embedding-based retrieval, and which model to use for a future user study.

## Running, replaying and inspecting

```bash
pip install -r requirements.txt
streamlit run app.py                 # works with no key
python -m unittest -v                # 32 tests of the core logic; no network or key needed
```

**API key.** I used my own OpenAI key for the live model. It is not in this repository and cannot be shared. To use the live model yourself:

1. Create a key in your own OpenAI account.
2. Locally: set it in the shell, then run the app.
```bash
export OPENAI_API_KEY=...            # PowerShell: $env:OPENAI_API_KEY = "..."
streamlit run app.py
```
3. On Streamlit Community Cloud: open the app's Settings, then Secrets, and add one line `OPENAI_API_KEY = "..."`. `app.py` copies it into the environment. Reboot the app.
4. The sidebar shows the active mode. Delete the key afterwards.

Without a key, reviewers can use the deployed app or run locally: the app then uses replay or rule-based mode, and every reply is labelled as such. To record replies for others to replay (synthetic data only): set `BUDGET_RECORD=1`, ask the questions, and commit `recordings/live_responses.json`.

## Data

Transactions come from `make_transactions()` (seeded, so reproducible) and include one planted one-off payment, a £135 conference dinner. The sidebar can download them as CSV and upload a CSV with columns `date, merchant, category, amount`. No real bank data was used.

## Development record

Commit links: [PASTE LINKS FOR THE MAIN COMMITS]

- **Stage 1, core and memory.** Seeded synthetic data, a memory store with four kinds of note, stated vs guessed labels, consent, 30-day expiry, and context assembly capped at four notes. Why: the research question is about visible, correctable memory. Checked: unit tests.
- **Stage 2, grounding and boundaries.** Evidence rows, reply checks, refusal of out-of-scope questions in code, and a validated budget-change path with confirm, cancel and undo. Why: keep suggestions separate from actions. Checked: unit tests and manual scenarios.
- **Stage 3, interface.** Memory panel, alerts with a sensitivity slider and one-off marking, quick-question buttons, and a seven-stage panel. Why: make each mechanism inspectable. Checked: walkthroughs T1 to T7.
- **Stage 4, deployment and live model.** Public repository and Streamlit Cloud deployment. Reading the key from Streamlit Secrets fixed the "no key" mode that appeared after deployment. A temporary key-check panel was added then removed; the unused design-notes tab was removed (the report explains the design); and a misleading "no API key" caption was reworded because the real cause was HTTP 429. Checked: redeployed and re-ran the walkthroughs.

### Test log (manual walkthroughs on seeded data)

| Test | What I did | Result |
|---|---|---|
| T1 Memory labels, consent | Added notes, viewed a guess, Remember / No thanks | Pass |
| T2 Grounded reply | Asked about eating out last month; compared figures with the evidence rows | Pass (4/4 checks) |
| T3 One-off | Marked the £135 dinner as a one-off | Pass (£205 became £70; alert removed) |
| T4 Alert sensitivity | Moved the slider through 0, 25, 50 and 80% | Pass |
| T5 Guarded change | "set groceries budget to 900", confirm, undo; "holidays" | Pass (warned, confirmed, undone; asked which category) |
| T6 Limits | Asked about Bitcoin and holidays | Pass (declined; no figure invented) |
| T7 Transparency | Opened the seven-stage panel and the note list | Pass (four of 11 notes sent) |
| T8 Live model | Asked with the OpenAI key set | Not evaluated (HTTP 429, fallback reply) |

Screenshots of each test are in the report.

## What I did for the demonstration

1. Loaded the app on seeded data and checked the sidebar mode.
2. Typed "remember ..." messages to create stated notes, then edited one and forgot one.
3. Accepted and declined a guessed note, then used "Still true".
4. Asked "How much did I spend eating out last month?" and opened the notes, evidence and checks panels.
5. Opened the Alerts tab, moved the slider, then marked the conference dinner as a one-off and forgot that note to restore it.
6. Typed "set groceries budget to 900", read the warning, confirmed, then undid it; typed an unknown category to trigger clarification.
7. Asked about Bitcoin and holidays to see the decline and no-data replies.
8. Opened the seven-stage panel and recorded the HTTP 429 fallback honestly.

## AI assistance

I used three AI tools.

- **Claude:** helped design and write the prototype code (memory, retrieval, checks, validation, Streamlit interface), a unit-test suite, the demonstration steps, the report and this README. I ran the tests and the app myself, read and changed suggestions (for example removing the design tab and rewording captions), and corrected claims that did not match my screenshots.
- **ChatGPT:** [EDIT: say what you used it for, e.g. explaining concepts, debugging, rewording].
- **GitHub Copilot:** [EDIT: say what you used it for, e.g. code completion and small fixes].
- **OpenAI API (inside the prototype):** the model that writes reply text. It is part of the system, not an assistant used to produce this work.

How I checked the results: unit tests, manual walkthroughs with screenshots, comparing reply figures against the transaction rows, and reading AI-written text against what the app actually did. AI text that overstated results was corrected. Example: a draft claimed live-model operation, but every call returned HTTP 429, so the report says it was not evaluated.

## Additional files

- `recordings/live_responses.json`: recorded replies, if present.
- `sample_transactions.csv`: download from the app sidebar.
- Report PDF: submitted separately on Canvas.
