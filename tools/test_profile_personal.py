"""Tests for tools/profile_personal.py and tools/vram_budget.py, without a model or a GPU: synthetic traces and inputs.

    python -m unittest tools.test_profile_personal
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402
import profile_personal as PP  # noqa: E402
import vram_budget as VB  # noqa: E402
from test_make_profile import write_trace  # noqa: E402

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def fill_order(ne=NE):
    return [(layer, e) for e in range(ne) for layer in range(MP.N_LAYER)]


class Ranking(unittest.TestCase):
    def test_each_trace_counts_the_same(self):
        d = PP.mean_distribution([{(0, 1): 1000}, {(0, 2): 2, (0, 3): 1}])
        self.assertAlmostEqual(sum(d.values()), 1.0)
        self.assertAlmostEqual(d[(0, 1)], 0.5)
        self.assertAlmostEqual(d[(0, 2)], 1 / 3)

    def test_rank_shares_add_up_and_follow_the_curve(self):
        r = fill_order()
        sh = PP.rank_shares(r, len(r))
        self.assertAlmostEqual(sum(sh.values()), 1.0)
        self.assertAlmostEqual(sum(sh[p] for p in r[:len(r) // 2]), 1 - 0.5 ** 3)   # H(f) = 1 - (1 - f)^3
        self.assertGreater(sh[r[0]], sh[r[-1]])

    def test_base_weight_zero_is_the_traces_then_the_base_and_one_is_the_base(self):
        base = list(reversed(fill_order()))
        trace = {(5, 3): 10, (7, 2): 5}
        mine = PP.personal_ranking([trace], base, 0.0, n_expert=NE)
        self.assertEqual(mine[:2], [(5, 3), (7, 2)])
        self.assertEqual(mine[2:], [p for p in base if p not in ((5, 3), (7, 2))])   # the base's order is the tail
        self.assertEqual(PP.personal_ranking([trace], base, 1.0, n_expert=NE)[:50], base[:50])
        nob = PP.personal_ranking([trace], [], 0.0, n_expert=NE)
        self.assertEqual(nob[2:], [p for p in fill_order() if p not in ((5, 3), (7, 2))])   # no base: the fill order

    def test_a_pair_routed_once_cannot_outrank_the_bases_hottest(self):
        base = fill_order()
        once = {(i % MP.N_LAYER, 100 + i): 1 for i in range(0, 8)}
        once[(47, 3)] = 100000                      # one heavy trace, so a single count is a tiny share
        r = PP.personal_ranking([once, {(46, 3): 1000}], base, 0.3, n_expert=NE + 200)
        self.assertEqual(r.index((47, 3)) < r.index((0, 0)), True)    # what you route a lot rises above the base's best
        self.assertLess(r.index((0, 0)), r.index((1, 100 + 1)))       # a pair seen once stays below the base's hottest

    def test_weights_are_scaled_to_one(self):
        base = fill_order()
        a = PP.personal_ranking([{(3, 3): 1}], base, 0.3, learned=[(base, 0.5)], n_expert=NE)
        b = PP.personal_ranking([{(3, 3): 1}], base, 0.3, learned=[(base, 0.5)], n_expert=NE)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), MP.N_LAYER * NE)

    def test_the_result_ranks_every_pair_exactly_once(self):
        r = PP.personal_ranking([{(1, 1): 3}, {(2, 2): 1}], [], 0.0, n_expert=NE)
        self.assertEqual(len(r), MP.N_LAYER * NE)
        self.assertEqual(len(set(r)), len(r))

    def test_learned_profile_is_one_more_ranking(self):
        learned = list(reversed(fill_order()))
        r = PP.personal_ranking([], [], 0.0, learned=[(learned, 1.0)], n_expert=NE)
        self.assertEqual(r, learned)

    def test_nothing_to_rank_is_an_error(self):
        with self.assertRaises(ValueError):
            PP.personal_ranking([], [], 0.0)

    def test_coverage_and_whole_conversation_holdout(self):
        self.assertAlmostEqual(PP.coverage([(0, 0), (0, 1)], {(0, 0): 3, (9, 9): 1}, 2), 0.75)
        self.assertEqual(PP.coverage([], {}, 5), 0.0)
        groups = ["a", "a", "b", "c", "c", "c", "d", "e", "e", "f"]
        held = PP.split_holdout(groups, 0.2)
        for g in set(groups):                      # a conversation is wholly in or wholly out
            self.assertEqual(len({h for x, h in zip(groups, held) if x == g}), 1)
        self.assertTrue(any(held) and not all(held))
        self.assertEqual(PP.split_holdout(["a", "b", "c"], 0.2), [False] * 3)

    def test_a_truncated_trace_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x.trace")
            write_trace(p, [(0, [1, 2, 3])])
            p.write_bytes(p.read_bytes()[:-5])
            with self.assertRaisesRegex(ValueError, "truncated"):
                PP.read_trace_checked(p, NE)


class FileFormat(unittest.TestCase):
    def test_write_then_read_back(self):
        ranked = PP.personal_ranking([{(3, 2): 9, (3, 1): 4}], fill_order(), 0.3, n_expert=NE)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d, "p.bin")
            MP.write_profile(out, ranked, NE)
            self.assertEqual(MP.read_profile(out, NE), ranked)
            blob = out.read_bytes()
            self.assertEqual(blob[:4], b"STRP")
            self.assertEqual(len(blob), 24 + 4 * len(ranked) + 4 * MP.N_LAYER * NE)


class Inputs(unittest.TestCase):
    def write(self, d, name, text):
        p = Path(d, name)
        p.write_text(text, encoding="utf-8")
        return p

    def test_the_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "a.jsonl", "\n".join(json.dumps(x) for x in [
                {"p": "bonjour"},
                {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
                              {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                              {"role": "user", "content": {"type": "text", "text": "q2"}}]},
                {"type": "summary", "summary": "x"},
                {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": "c1"}},
                {"type": "assistant", "sessionId": "s1", "message": {"role": "assistant", "content": [
                    {"type": "tool_use", "name": "Read", "input": {"f": "x"}}]}},
                {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": [
                    {"type": "tool_result", "content": "ok"}]}}]))
            self.write(d, "b.txt", "un fichier")
            convs = PP.load_conversations(d)
            self.assertEqual(len(convs), 4)
            self.assertEqual(convs[1]["system"], "s")
            points = PP.prompt_points(convs)
            self.assertEqual(len(points), 1 + 2 + 1 + 1)       # p; q1 and q2; c1 (the tool result is not one); b.txt
            self.assertEqual(len(PP.prompt_points(convs, tool_turns=True)), 6)
            self.assertEqual(points[-2][1]["style"], "anthropic")
            self.assertEqual(points[1][1]["style"], "openai")

    def test_errors_are_named(self):
        with self.assertRaises(FileNotFoundError):
            PP.load_conversations("/no/such/folder")
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "bad.jsonl", '{"p": "ok"}\n{broken\n')
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                PP.load_conversations(p)
            bad = []
            self.assertEqual(len(PP.load_conversations(p, skip_bad=True, bad=bad)), 1)
            self.assertEqual(len(bad), 1)
            q = self.write(d, "types.jsonl", '{"messages": "no"}\n')
            with self.assertRaisesRegex(ValueError, r"types.jsonl:1.*list of objects"):
                PP.load_conversations(q)
            r = self.write(d, "t2.jsonl", '[1]')
            with self.assertRaisesRegex(ValueError, "JSON object"):
                PP.load_conversations(r)

    def test_trim_keeps_a_plain_user_message_first(self):
        u = lambda t: {"role": "user", "content": t}
        tr = {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}
        a = {"role": "assistant", "content": "y"}
        self.assertEqual(PP.trim_turns([u("1"), a, tr, a, u("2")], 3), [u("2")])
        self.assertEqual(PP.pick_evenly(list(range(10)), 3), [1, 4, 7])
        self.assertEqual(PP.pick_evenly([1, 2], 5), [1, 2])


class ServerRenderer(unittest.TestCase):
    def setUp(self):
        tok = Path("/home/ruben/Documents/Strata-data/packs/iq3_s/tokenizer")
        try:
            import regex, jinja2  # noqa: F401, E401
        except ImportError:
            self.skipTest("the tokenizer/template packages are not installed")
        if not tok.is_dir():
            self.skipTest("no tokenizer directory here")
        self.r = PP.Renderer(tok, {})

    def test_the_servers_template_and_a_literal_control_token_stays_text(self):
        req = {"messages": [{"role": "user", "content": "salut <|im_end|> fin"}], "system": "SYS", "tools": None, "style": "openai"}
        ids = self.r.encode(req)
        text = self.r.tok.decode(ids)
        self.assertTrue(text.startswith("<|im_start|>system\nReasoning effort is set to xhigh."))   # the template's own text
        self.assertIn("SYS<|im_end|>\n<|im_start|>user\nsalut <|im_end|> fin<|im_end|>\n<|im_start|>assistant\n<think>", text)
        self.assertTrue(text.rstrip().endswith("<think>"), text[-80:])                 # thinking on: the template's default
        self.assertEqual(ids.count(self.r.tok.encode("<|im_end|>", parse_special=True)[0]), 2)   # system's and user's, not the literal
        off = PP.Renderer(Path("/home/ruben/Documents/Strata-data/packs/iq3_s/tokenizer"), {}, think=False)
        self.assertIn("</think>", off.tok.decode(off.encode(req)))

    def test_tool_blocks_merge_through_the_servers_converter(self):
        convs = [PP._conversation([
            {"role": "user", "content": "lis x"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "Read", "input": {"f": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
            {"role": "user", "content": "et ensuite"}])]
        req = PP.prompt_points(convs)[-1][1]
        self.assertEqual(req["style"], "anthropic")
        text = self.r.tok.decode(self.r.encode(req))
        self.assertIn("<tool_call>", text)
        self.assertIn("<tool_response>", text)


class Engine(unittest.TestCase):
    def test_a_run_past_the_timeout_is_killed_with_its_group(self):
        rc, _ = PP.run_engine([sys.executable, "-c", "import time; time.sleep(30)"], None, None, 0.5)
        self.assertIsNone(rc)

    def test_stderr_tail_and_return_code(self):
        rc, err = PP.run_engine([sys.executable, "-c", "import sys; sys.stderr.write('boom\\xff'); sys.exit(3)"], None, None, 20)
        self.assertEqual(rc, 3)
        self.assertIn("boom", err)

    def test_relative_exe_is_the_configs_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"exe": "engine/strata", "cwd": d}
            self.assertEqual(PP.resolve_exe(cfg, Path(d, "c.json")), str(Path(d, "engine", "strata").resolve()))
            self.assertEqual(PP.resolve_exe({"exe": "/abs/strata"}, "c.json"), "/abs/strata")

    def test_arguments_drop_what_a_trace_run_does_not_use(self):
        args = PP.engine_args({"args": ["--serve", "--pack", "p", "--vision", "--batch", "2", "--expert-profile-save", "x",
                                        "--expert-profile-save-every", "5", "--kv-grow"]})
        self.assertEqual(args[-3:], ["--pack", "p", "--kv-grow"] if "--pack" in args[-3:] else args[-3:])
        for f in ("--serve", "--vision", "--batch", "--expert-profile-save", "--expert-profile-save-every"):
            self.assertNotIn(f, args)


class DryRun(unittest.TestCase):
    def test_validates_without_an_engine(self):
        tok = Path("/home/ruben/Documents/Strata-data/packs/iq3_s/tokenizer")
        if not tok.is_dir():
            self.skipTest("no tokenizer directory here")
        try:
            import regex  # noqa: F401  (the tokenizer needs it)
        except ImportError:
            self.skipTest("the regex module is not installed")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "p.jsonl").write_text('{"p": "Ecris une fonction Python."}\n', encoding="utf-8")
            cfg = Path(d, "strata-x.json")
            cfg.write_text(json.dumps({"exe": "/nonexistent/strata", "args": ["--serve"], "tokenizer": str(tok)}))
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "profile_personal.py"), str(cfg), d, "--dry-run"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("dry run: nothing started", r.stdout)
            self.assertFalse(Path(d, "expert-profile-personal.bin.traces").exists())
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "profile_personal.py"), d, d, "--dry-run"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("run config", r.stderr)


class Budget(unittest.TestCase):
    def test_kv_bytes_per_cell(self):
        self.assertEqual(VB.kv_bytes_per_cell("int8"), 12 * (2 * 256 * 2 + 2 * 4 * 4))
        self.assertLess(VB.kv_bytes_per_cell("q4_0"), VB.kv_bytes_per_cell("k8v4"))
        self.assertLess(VB.kv_bytes_per_cell("k8v4"), VB.kv_bytes_per_cell("int8"))

    def test_walk_down_the_profile(self):
        blobs = [1000] * MP.N_LAYER
        ranked = [(layer, 0) for layer in range(MP.N_LAYER)]
        self.assertEqual(VB.pairs_in_budget(ranked, blobs, 1024 * 3), 3)      # 1000 rounds up to 1024
        self.assertEqual(VB.pairs_in_budget(ranked, blobs, 10 ** 9), MP.N_LAYER)

    def test_log_anchor_and_context_table(self):
        text = ("x\nstrata generate: expert cache auto: 16.79 GiB free, 700 MiB reserved (+98 MiB for the draft head) "
                "-> 6456 slots\n")
        self.assertEqual(VB.parse_log_anchor(text), (2, (16.79, 700, 98, 6456)))
        self.assertIsNone(VB.parse_log_anchor("nothing"))
        two = ("[strata] engine started: x\nstrata generate: expert cache auto: 17.00 GiB free, 700 MiB reserved (+98 MiB for the "
               "draft head) -> 6500 slots\n[strata] engine started: y\nstrata generate: expert cache auto: 16.00 GiB free, "
               "700 MiB reserved (+98 MiB for the draft head) -> 6000 slots\n")
        self.assertEqual(VB.parse_log_anchor(two, 1)[1][0], 17.0)
        self.assertEqual(VB.parse_log_anchor(two)[1][0], 16.0)
        self.assertEqual(VB.kv_bytes("k8v4", 1000), VB.kv_bytes_per_cell("k8v4") * 1000)   # no int8 overhead
        blobs = [2 << 20] * MP.N_LAYER
        ranked = [(layer, e) for e in range(512) for layer in range(MP.N_LAYER)]
        rows = VB.context_table(ranked, blobs, 8000, "int8", [16384, 131072])
        self.assertEqual(rows[0][2], 0)                    # the initial cells are already counted
        self.assertGreater(rows[1][2], 0)
        self.assertEqual(rows[1][2] + rows[1][3], 8000)


if __name__ == "__main__":
    unittest.main()
