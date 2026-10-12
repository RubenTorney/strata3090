# RTX 3090 + Ryzen 5 5600X, Linux: the opt/3090 options measured (2026-10-12)

The write-up is `docs/RTX3090.md`. Files here, all from `tools/strata_ab_linux.py` (12 cold prompts, 256 tokens, a fresh
engine per arm, IQ3_S, context 262,144):

- `speed1.*`: each option alone on the real config, 2 rounds (the first engine of the series, `base` round 0, was slow)
- `speed2_det.*`: the same-output options paired under `STRATA_DETERMINISTIC=1 --adapt-swaps 0`
- `speed3_combo.*`: base / PCIe branch + 64-block fetch + GR_FAST / the same + IQ_ZEN3, 3 rotated rounds
- `speed4_profile.*`: the shipped expert profile against `expert-profile-frcode.bin` (built from `profile_prompts/`)
- `speed5_tune.*`, `speed6_tune2.*`: one-setting sweeps on top of the combination
- `parity_*.jsonl`: self-consistency checks of the deterministic mode
- `base_config.json`, `opt_config.json`, `config_rtx3090_max.json`: the configs (the last one is the retained one)
