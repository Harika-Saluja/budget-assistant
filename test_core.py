"""Run with:  python -m unittest -v
No network or API key is needed: tests clear the key variables and use temporary files."""
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import checks
import core
import tools

TODAY = date(2026, 11, 19)
NO_KEYS = {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "BUDGET_RECORD": ""}


def tmp_path(name):
    return Path(tempfile.mkdtemp()) / name


def fresh_store():
    return core.MemoryStore(tmp_path("memory.json"))


def fresh_recordings():
    return core.Recordings(tmp_path("recordings.json"))


class MemoryTests(unittest.TestCase):
    def test_round_trip_persistence(self):
        s = fresh_store()
        s.add(core.new_memory("semantic", "I rent a flat.", "stated", "told", today=TODAY))
        self.assertEqual([m.text for m in core.MemoryStore(s.path).items], ["I rent a flat."])

    def test_forget_clears_item_and_derived_summary(self):
        s = fresh_store()
        m = core.new_memory("semantic", "Saving for a laptop.", "stated", "told", today=TODAY)
        s.add(m)
        self.assertIn("laptop", s.summary(TODAY))
        s.forget(m.id)
        self.assertNotIn("laptop", s.summary(TODAY))
        self.assertNotIn("laptop", s.path.read_text(encoding="utf-8"))

    def test_edit_becomes_stated_and_never_expires(self):
        s = fresh_store()
        m = core.new_memory("episodic", "Eating out jumped.", "inferred", "guess", today=TODAY)
        s.add(m)
        s.edit(m.id, "That was a one-off work trip.", today=TODAY)
        got = s.get(m.id)
        self.assertEqual((got.source, got.expires, got.text), ("stated", None, "That was a one-off work trip."))

    def test_inferred_goes_stale_and_is_left_out_until_confirmed(self):
        s = fresh_store()
        m = core.new_memory("semantic", "You eat out a lot.", "inferred", "guess", today=TODAY)
        s.add(m)
        later = TODAY + timedelta(days=core.INFERRED_TTL_DAYS + 1)
        self.assertTrue(core.is_stale(m, later))
        self.assertEqual(s.active(later), [])
        s.confirm(m.id, today=later)
        self.assertEqual(len(s.active(later)), 1)

    def test_memory_kinds(self):
        self.assertEqual(core.classify_memory_text("Never suggest cancelling the gym"), "procedural")
        self.assertEqual(core.classify_memory_text("Please always answer in pounds"), "procedural")
        self.assertEqual(core.classify_memory_text("I'm saving for a laptop"), "semantic")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tx = core.make_transactions(today=TODAY)
        self.budgets = dict(core.DEFAULT_BUDGETS)

    def test_deterministic_and_has_ids(self):
        self.assertEqual(self.tx, core.make_transactions(today=TODAY))
        self.assertTrue(all(t["id"] for t in self.tx))

    def test_csv_round_trip_and_validation(self):
        rows, errors = core.parse_transactions_csv(core.transactions_to_csv(self.tx))
        self.assertEqual((len(rows), errors), (len(self.tx), []))
        bad = "date,merchant,category,amount\n2026-10-01,Shop,groceries,12.5\nnot-a-date,Shop,groceries,5\n2026-10-02,,groceries,5\n"
        rows, errors = core.parse_transactions_csv(bad)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(errors), 2)
        rows, errors = core.parse_transactions_csv("date,merchant\n2026-10-01,Shop\n")
        self.assertEqual(rows, [])
        self.assertIn("Missing column", errors[0])

    def test_alerts_false_alarm_and_sensitivity(self):
        cats = {a["category"] for a in core.compute_alerts(self.tx, self.budgets, 0)}
        self.assertIn("eating out", cats)                      # the planted one-off causes an alert
        self.assertEqual(core.compute_alerts(self.tx, self.budgets, 100), [])   # less sensitive: none
        counts = [len(core.compute_alerts(self.tx, self.budgets, s)) for s in (0, 25, 50, 100)]
        self.assertEqual(counts, sorted(counts, reverse=True))  # never more alerts at a higher threshold

    def test_one_off_note_removes_the_alert_and_forgetting_restores_it(self):
        store = fresh_store()
        top = core.compute_alerts(self.tx, self.budgets, 0)[0]["top"]
        m = core.new_memory("episodic", "one-off", "stated", "marked", today=TODAY, ref=top["id"])
        store.add(m)
        eff = core.effective_transactions(self.tx, store, TODAY)
        self.assertNotIn("eating out", {a["category"] for a in core.compute_alerts(eff, self.budgets, 0)})
        store.forget(m.id)
        eff = core.effective_transactions(self.tx, store, TODAY)
        self.assertIn("eating out", {a["category"] for a in core.compute_alerts(eff, self.budgets, 0)})

    def test_inferences_include_the_planted_one_off(self):
        tags = {p["tag"] for p in core.propose_inferences(self.tx)}
        self.assertTrue({"eating-spike", "subs", "eating-avg"} <= tags)


class RetrievalAndContextTests(unittest.TestCase):
    def setUp(self):
        self.tx = core.make_transactions(today=TODAY)
        self.budgets = dict(core.DEFAULT_BUDGETS)

    def test_evidence_is_from_the_mentioned_category_and_latest_month(self):
        rows = core.retrieve_evidence("How much did I spend eating out?", self.tx)
        self.assertTrue(rows)
        self.assertEqual({t["category"] for t in rows}, {"eating out"})
        self.assertEqual(len({t["date"][:7] for t in rows}), 1)
        self.assertEqual(core.retrieve_evidence("How am I doing?", self.tx), [])

    def test_context_limits_memories_orders_sections_and_restates_task(self):
        mems = [core.new_memory("semantic", f"Note {i} about eating out", "stated", "told", today=TODAY)
                for i in range(7)]
        ctx = core.assemble_context("How much eating out?", mems, self.tx, self.budgets, [])
        self.assertEqual(len(ctx["memories_used"]), core.MAX_MEMORIES_IN_CONTEXT)
        self.assertEqual(ctx["record"]["memories_left_out"], 3)
        self.assertEqual([s["name"] for s in ctx["record"]["sections"]][-1], "task")
        self.assertTrue(ctx["prompt"].rstrip().endswith("How much eating out?"))

    def test_irrelevant_memory_is_not_sent_but_procedural_is(self):
        likes = core.new_memory("semantic", "I like cycling.", "stated", "told", today=TODAY)
        rule = core.new_memory("procedural", "Never suggest cancelling the gym.", "stated", "told", today=TODAY)
        ctx = core.assemble_context("How much eating out?", [likes, rule], self.tx, self.budgets, [])
        self.assertEqual([m.id for m in ctx["memories_used"]], [rule.id])

    def test_task_spec_has_all_six_parts(self):
        prompt = core.build_system_prompt()
        for part in ("TASK:", "AUDIENCE:", "CONSTRAINTS:", "EXAMPLES:", "SUCCESS CRITERIA:", "OUTPUT FORMAT:"):
            self.assertIn(part, prompt)


class ChecksTests(unittest.TestCase):
    def setUp(self):
        tx = core.make_transactions(today=TODAY)
        self.allowed = core.known_numbers(tx, core.DEFAULT_BUDGETS)

    def test_figures_grounded_vs_invented(self):
        good = "Your budget for eating out is £120."
        bad = "Your eating out was £999 last month."
        self.assertEqual(checks.match_figures(good, self.allowed)[1], [])
        self.assertEqual(checks.match_figures(bad, self.allowed)[1], [999.0])

    def test_reply_checks(self):
        res = checks.reply_checks("You spent £999.", self.allowed, inferred_used=False)
        self.assertFalse(res[0]["ok"])
        res = checks.reply_checks("It is a guess and may be out of date.", self.allowed, inferred_used=True)
        self.assertTrue(res[2]["ok"])
        res = checks.reply_checks("It was high.", self.allowed, inferred_used=True)
        self.assertFalse(res[2]["ok"])
        res = checks.reply_checks("I've changed your budget.", self.allowed, inferred_used=False)
        self.assertFalse(res[3]["ok"])
        res = checks.reply_checks("word " * 130, self.allowed, inferred_used=False)
        self.assertFalse(res[1]["ok"])


class ToolBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.budgets = dict(core.DEFAULT_BUDGETS)

    def test_valid_call(self):
        v = tools.validate_tool_call("propose_budget_change", {"category": "Eating Out", "amount": 150}, self.budgets)
        self.assertTrue(v["ok"])
        self.assertEqual(v["call"], {"category": "eating out", "amount": 150.0})
        self.assertEqual(v["warning"], "")

    def test_rejections(self):
        bad = [("delete_everything", {}), ("propose_budget_change", {"category": "holidays", "amount": 10}),
               ("propose_budget_change", {"category": "groceries", "amount": "lots"}),
               ("propose_budget_change", {"category": "groceries", "amount": -5}),
               ("propose_budget_change", {"category": "groceries", "amount": 10 ** 6}),
               ("propose_budget_change", None)]
        for name, args in bad:
            self.assertFalse(tools.validate_tool_call(name, args, self.budgets)["ok"], (name, args))

    def test_big_change_warns(self):
        v = tools.validate_tool_call("propose_budget_change", {"category": "eating out", "amount": 1200}, self.budgets)
        self.assertTrue(v["ok"])
        self.assertIn("big change", v["warning"])

    def test_schema_lists_the_categories(self):
        schema = tools.propose_budget_tool(self.budgets)
        self.assertEqual(schema["input_schema"]["properties"]["category"]["enum"], sorted(self.budgets))

    def test_parse_anthropic_response_separates_text_and_proposal(self):
        data = {"content": [{"type": "text", "text": "Here you go."},
                            {"type": "tool_use", "name": "propose_budget_change",
                             "input": {"category": "groceries", "amount": 200}}]}
        text, calls = core.parse_anthropic_response(data)
        self.assertEqual(text, "Here you go.")
        self.assertEqual(calls[0]["input"]["amount"], 200)
        with self.assertRaises(core.LLMError):
            core.parse_anthropic_response({"oops": 1})


@mock.patch.dict(os.environ, NO_KEYS)
class BehaviourTests(unittest.TestCase):
    def setUp(self):
        self.tx = core.make_transactions(today=TODAY)
        self.budgets = dict(core.DEFAULT_BUDGETS)
        self.store = fresh_store()
        self.recs = fresh_recordings()

    def ask(self, msg, today=TODAY):
        return core.answer(msg, self.store, self.tx, self.budgets, [], today=today, recordings=self.recs)

    def test_out_of_scope_is_declined_in_code(self):
        self.assertEqual(self.ask("Write me a poem about the sea")["mode"], "declined")

    def test_unknown_topic_is_nothing_found_not_a_guess(self):
        res = self.ask("How much did I spend on holidays last month?")
        self.assertEqual(res["mode"], "nothing_found")
        self.assertIn("don't have any data", res["text"])

    def test_demo_reply_is_grounded_and_passes_its_own_checks(self):
        res = self.ask("How much did I spend eating out last month?")
        self.assertEqual(res["mode"], "demo")
        self.assertTrue(res["text"].startswith("You asked about eating out."))
        self.assertTrue(res["evidence"])
        self.assertTrue(all(c["ok"] for c in res["checks"] if c["ok"] is not None))
        self.assertEqual(len(res["stages"]), 7)

    def test_memory_used_and_guess_flagged(self):
        self.store.add(core.new_memory("episodic", "Your eating-out spending jumped.", "inferred", "g", today=TODAY))
        res = self.ask("How much did I spend eating out?")
        self.assertEqual(len(res["memories_used"]), 1)
        self.assertIn("guess of mine", res["text"])

    def test_stale_memory_is_not_used_and_is_counted(self):
        self.store.add(core.new_memory("episodic", "Your eating-out spending jumped.", "inferred", "g", today=TODAY))
        res = self.ask("How much did I spend eating out?", today=TODAY + timedelta(days=60))
        self.assertEqual(res["memories_used"], [])
        self.assertEqual(res["stale_unused"], 1)

    def test_a_one_off_note_changes_the_answer(self):
        before = self.ask("How much did I spend eating out last month?")["text"]
        top = core.compute_alerts(self.tx, self.budgets, 0)[0]["top"]
        self.store.add(core.new_memory("episodic", "one-off", "stated", "m", today=TODAY, ref=top["id"]))
        after = self.ask("How much did I spend eating out last month?")["text"]
        self.assertNotEqual(before, after)

    def test_recorded_live_output_replays_without_a_key(self):
        key = core.Recordings.key("How much did I spend eating out last month?", [])
        self.recs.put(key, "Recorded answer: budget £120.", [], "test-model")
        res = self.ask("How much did I spend eating out last month?")
        self.assertEqual(res["mode"], "replay")
        self.assertIn("Recorded answer", res["text"])
        self.assertEqual(core.Recordings(self.recs.path).get(key)["model"], "test-model")

    def test_a_recorded_tool_call_becomes_a_validated_proposal_not_an_action(self):
        msg = "Could you lower my food budget to about 100?"
        key = core.Recordings.key(msg, [])
        self.recs.put(key, "", [{"name": "propose_budget_change", "input": {"category": "eating out", "amount": 100}}], "t")
        res = self.ask(msg)
        self.assertEqual(res["proposal"]["category"], "eating out")
        self.assertEqual(self.budgets["eating out"], 120.0)      # nothing changed
        self.assertIn("Nothing changes until you confirm", res["text"])

    def test_an_invalid_tool_call_is_ignored(self):
        msg = "Please sort out my money budget"
        key = core.Recordings.key(msg, [])
        self.recs.put(key, "Done.", [{"name": "wipe_account", "input": {}}], "t")
        res = self.ask(msg)
        self.assertIsNone(res["proposal"])
        self.assertIn("failed validation", res["text"])

    def test_commands_and_repair(self):
        self.assertEqual(core.parse_remember("Remember that I get paid on the 25th"), "I get paid on the 25th")
        self.assertIsNone(core.parse_remember("how much did I spend"))
        self.assertEqual(core.parse_budget_command("set eating out budget to 150"),
                         {"raw": "eating out", "category": "eating out", "amount": 150.0})
        self.assertEqual(core.parse_budget_command("Lower my food budget to £90")["category"], "eating out")
        unknown = core.parse_budget_command("set holidays budget to 90")
        self.assertIsNone(unknown["category"])
        self.assertTrue(core.suggest_categories("grocery", self.budgets)[0] == "groceries")
        self.assertEqual(sorted(core.suggest_categories("zzz", self.budgets)), sorted(self.budgets))

    def test_outcome_note_compares_with_last_month(self):
        note = core.outcome_note("subscriptions", 30, self.tx)
        self.assertIn("over this budget", note)


if __name__ == "__main__":
    unittest.main()
