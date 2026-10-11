"""Tests of tools/strata_ab_linux.py's pure parts (no engine, no GPU, no network):

    python -m unittest tools.test_strata_ab_linux
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_ab_linux as ab  # noqa: E402

LOG = ("strata serve: prompt 31 tokens = 0 reused + 31 read in 412 ms (75.2 tok/s), 128 generated in 1069 ms (119.7 tok/s), "
       "drafts accepted 88 of 120, 3 checkpoints\n")


class Parse(unittest.TestCase):
    def test_request_line(self):
        r = ab.parse_engine_line(LOG)
        self.assertEqual((r["prompt_tokens"], r["reused"], r["read"], r["gen"], r["accepted"], r["drafted"]),
                         (31, 0, 31, 128, 88, 120))
        self.assertAlmostEqual(r["decode_tps"], 119.7)
        self.assertAlmostEqual(r["prefill_tps"], 75.2)

    def test_cancelled_read_and_no_drafts(self):
        r = ab.parse_engine_line("strata serve: prompt 900 tokens = 10 reused + 512 of 890 read in 100 ms (5120.0 tok/s), "
                                 "0 generated in 0 ms (0.0 tok/s), (cancelled)")
        self.assertEqual((r["read"], r["gen"], r["drafted"]), (512, 0, 0))

    def test_other_line(self):
        self.assertIsNone(ab.parse_engine_line("strata serve: decode expert cache hit rate: 91.0% (10 hits / 11 lookups)"))

    def test_startup(self):
        t = ("strata generate: expert cache 5120 slots, 21.40 GiB of VRAM; policy is\n"
             "strata serve: prompt chunk auto: 8192 tokens, a 384-slot ring\n")
        self.assertEqual(ab.parse_startup(t), {"expert_cache": 5120, "prefill_chunk": 8192})
        self.assertEqual(ab.parse_startup("nothing"), {})


class Pairing(unittest.TestCase):
    def test_median(self):
        self.assertEqual(ab.median([3, 1, 2]), 2)
        self.assertEqual(ab.median([1, 2, 3, 4]), 2.5)

    def test_hash_depends_on_text_and_count(self):
        self.assertEqual(ab.answer_hash("a", 3), ab.answer_hash("a", 3))
        self.assertNotEqual(ab.answer_hash("a", 3), ab.answer_hash("a", 4))
        self.assertNotEqual(ab.answer_hash("a", 3), ab.answer_hash("b", 3))

    def test_self_consistency(self):
        self.assertEqual(ab.self_consistency([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]), [])
        self.assertEqual(ab.self_consistency([["a", "b", "c"], ["a", "x", "c"], ["a", "b", "y"]]), [1, 2])
        self.assertEqual(ab.self_consistency([]), [])

    def test_cross_arm(self):
        self.assertEqual(ab.cross_arm(["a", "b", "c"], ["a", "x", "c"]), [1])

    def test_paired_ratio_and_exclusion(self):
        base = {0: dict(hash="h0", tps=[100.0]), 1: dict(hash="h1", tps=[50.0]), 2: dict(hash="h2", tps=[80.0]),
                3: dict(hash="h3", tps=[])}
        arm = {0: dict(hash="h0", tps=[110.0]), 1: dict(hash="DIFF", tps=[500.0]), 2: dict(hash="h2", tps=[72.0]),
               3: dict(hash="h3", tps=[10.0])}
        r = ab.paired_speed(base, arm)
        self.assertEqual(r["diverged"], [1])
        self.assertEqual(r["nospeed"], [3])
        self.assertEqual(sorted(r["ratios"]), [0, 2])
        self.assertAlmostEqual(r["ratios"][0], 1.1)
        self.assertAlmostEqual(r["ratios"][2], 0.9)
        self.assertAlmostEqual(r["median"], 1.0)
        self.assertEqual((r["faster"], r["n"]), (1, 2))

    def test_repeats_use_their_median_and_mixed_hash_is_marked(self):
        m = ab.merge_hash([("h", 100.0), ("h", 120.0), ("h", 110.0)])
        self.assertEqual(m["hash"], "h")
        self.assertEqual(ab.median(m["tps"]), 110.0)
        mixed = ab.merge_hash([("h", 1.0), ("g", 2.0)])
        self.assertTrue(mixed["hash"].startswith("*mixed*"))
        r = ab.paired_speed({0: m}, {0: mixed})
        self.assertEqual(r["diverged"], [0])
        self.assertEqual(r["n"], 0)

    def test_all_diverged(self):
        r = ab.paired_speed({0: dict(hash="a", tps=[1.0])}, {0: dict(hash="b", tps=[2.0])})
        self.assertEqual(r["n"], 0)
        self.assertTrue(r["median"] != r["median"])    # nan


class Config(unittest.TestCase):
    BASE = {"exe": "/old/strata", "args": ["--spec", "4", "--expert-cache", "auto", "--prefill", "auto", "--pcie-frac", "0.35",
                                          "--kv-grow", "--adapt-async", "1"], "env": {"A": "1"}, "port": 8080}

    def test_arm_parse(self):
        a = ab.parse_arm("pf50;exe=/x/strata;arg=--pcie-frac 0.5;arg=--kv-grow;env=STRATA_X=1")
        self.assertEqual(a, dict(name="pf50", exe="/x/strata", args={"--pcie-frac": "0.5", "--kv-grow": None},
                                 env={"STRATA_X": "1"}))
        for bad in ("exe=/x", "n;arg=pcie", "n;env=NOVALUE", "n;what=1", "n;arg=--a b c"):
            with self.assertRaises(ValueError, msg=bad):
                ab.parse_arm(bad)

    def test_apply_args(self):
        a = ab.apply_args(self.BASE["args"], {"--expert-cache": "5000", "--suffix-draft": "0", "--kv-grow": None})
        self.assertEqual(a[a.index("--expert-cache") + 1], "5000")
        self.assertEqual(a[a.index("--suffix-draft") + 1], "0")
        self.assertEqual(a.count("--kv-grow"), 1)
        self.assertEqual(self.BASE["args"][3], "auto")           # the input is not changed

    def test_apply_args_flag_gets_a_value_inserted_before_the_next_flag(self):
        self.assertEqual(ab.apply_args(["--a", "--b", "1"], {"--a": "x"}), ["--a", "x", "--b", "1"])

    def test_build_config_arm_wins_over_the_bundle(self):
        det = ab.det_bundle(5000, 8192, "", "7", True)
        arm = ab.parse_arm("t;arg=--expert-cache 4000;env=STRATA_ROUTE_TAIL_SKIP=0")
        cfg = ab.build_config(self.BASE, arm, det, "/new/strata", "/l.log", 18080)
        self.assertEqual(cfg["exe"], "/new/strata")
        self.assertEqual(cfg["args"][cfg["args"].index("--expert-cache") + 1], "4000")
        self.assertEqual(cfg["args"][cfg["args"].index("--prefill") + 1], "8192")
        self.assertEqual(cfg["env"]["STRATA_DETERMINISTIC"], "1")
        self.assertEqual(cfg["env"]["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertEqual(cfg["env"]["A"], "1")
        self.assertEqual((cfg["port"], cfg["log"]), (18080, "/l.log"))
        self.assertEqual(self.BASE["exe"], "/old/strata")

    def test_no_deterministic_bundle_is_empty_but_for_pins(self):
        det = ab.det_bundle(0, 0, "", "7", False)
        self.assertEqual(det, {"args": {}, "env": {}})
        self.assertNotIn("STRATA_DETERMINISTIC", ab.det_bundle(1, 1, "0.4", "7", False)["env"])


class Processes(unittest.TestCase):
    def test_descendants(self):
        table = {10: 1, 11: 10, 12: 10, 13: 11, 20: 1, 21: 20}
        self.assertEqual(sorted(ab.descendants(10, table)), [11, 12, 13])
        self.assertEqual(ab.descendants(99, table), [])

    def test_wait_until_times_out(self):
        t0 = time.monotonic()
        self.assertFalse(ab.wait_until("never", lambda: False, 0.3, 0.05))
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(ab.wait_until("now", lambda: True, 0.3, 0.05))
        self.assertFalse(ab.wait_until("aborted", lambda: False, 5.0, 0.05, abort=lambda: True))

    def test_proc_table_sees_this_process(self):
        self.assertEqual(ab.proc_table()[os.getpid()], os.getppid())

    def test_port_in_use(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        self.assertTrue(ab.port_in_use(port))
        s.close()
        self.assertFalse(ab.port_in_use(port))

    def test_help_runs(self):
        out = subprocess.run([sys.executable, str(ROOT / "tools" / "strata_ab_linux.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("parity", out.stdout)


if __name__ == "__main__":
    unittest.main()
