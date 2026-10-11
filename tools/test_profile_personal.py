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

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def fill_order(ne=NE):
    return [(layer, e) for e in range(ne) for layer in range(MP.N_LAYER)]


class Ranking(unittest.TestCase):
    def test_each_trace_counts_the_same(self):
        long_t = {(0, 1): 1000}
        short_t = {(0, 2): 2, (0, 3): 1}
        self.assertEqual(PP.ranking_from_traces([long_t, short_t])[:3], [(0, 1), (0, 2), (0, 3)])

    def test_base_weight_zero_is_the_traces_and_one_is_the_base(self):
        base = fill_order()
        trace = {(5, 3): 10, (7, 2): 5}
        mine = PP.personal_ranking([trace], base, 0.0, n_expert=NE)
        self.assertEqual(mine[:2], [(5, 3), (7, 2)])
        self.assertEqual(PP.personal_ranking([trace], base, 1.0, n_expert=NE), base)

    def test_a_blend_keeps_the_bases_top_above_pairs_nobody_you_used(self):
        base = fill_order()
        trace = {(40, 3): 10}
        r = PP.personal_ranking([trace], base, 0.3, n_expert=NE)
        self.assertEqual(r[0], (40, 3))                       # your one pair wins at weight 0.7
        self.assertEqual(r[1], (0, 0))                        # then the base's first
        self.assertEqual(sorted(r), sorted(fill_order()))     # still every pair once

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

    def test_coverage_and_holdout(self):
        self.assertAlmostEqual(PP.coverage([(0, 0), (0, 1)], {(0, 0): 3, (9, 9): 1}, 2), 0.75)
        self.assertEqual(PP.coverage([], {}, 5), 0.0)
        train, held = PP.split_holdout(list(range(10)), 0.2)
        self.assertEqual(held, [4, 9])
        self.assertEqual(len(train), 8)
        self.assertEqual(PP.split_holdout([1, 2, 3], 0.2), ([1, 2, 3], []))


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
                              {"role": "user", "content": "q2"}]},
                {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": "c1"}},
                {"type": "assistant", "sessionId": "s1", "message": {"role": "assistant", "content": [
                    {"type": "tool_use", "name": "Read", "input": {"f": "x"}}]}},
                {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": [
                    {"type": "tool_result", "content": "ok"}]}}]))
            self.write(d, "b.txt", "un fichier")
            convs = PP.load_conversations(d)
            points = PP.prompt_points(convs)
            self.assertEqual(len(convs), 4)
            self.assertEqual(len(points), 1 + 2 + 2 + 1)          # p, q1+q2, c1 + the tool result (a user turn), b.txt
            self.assertEqual(points[2][-1], ("user", "q2"))
            self.assertIn("<tool_call>", points[-2][1][1])

    def test_errors_are_named(self):
        with self.assertRaises(FileNotFoundError):
            PP.load_conversations("/no/such/folder")
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "bad.jsonl", '{"p": "ok"}\n{broken\n')
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                PP.load_conversations(p)

    def test_render_and_pick(self):
        s = PP.render_prompt([("user", "a"), ("assistant", "b"), ("user", "c")], "SYS")
        self.assertTrue(s.startswith("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\na"))
        self.assertTrue(s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(PP.render_prompt([("user", "a"), ("assistant", "b"), ("user", "c")], None, 2).count("<|im_start|>user"), 1)
        self.assertEqual(PP.pick_evenly(list(range(10)), 3), [1, 4, 7])
        self.assertEqual(PP.pick_evenly([1, 2], 5), [1, 2])


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
        self.assertEqual(VB.parse_log_anchor(text), (16.79, 700, 98, 6456))
        self.assertIsNone(VB.parse_log_anchor("nothing"))
        blobs = [2 << 20] * MP.N_LAYER
        ranked = [(layer, e) for e in range(512) for layer in range(MP.N_LAYER)]
        rows = VB.context_table(ranked, blobs, 8000, "int8", [16384, 131072])
        self.assertEqual(rows[0][2], 0)                    # the initial cells are already counted
        self.assertGreater(rows[1][2], 0)
        self.assertEqual(rows[1][2] + rows[1][3], 8000)


if __name__ == "__main__":
    unittest.main()
