# Issue #9774 performance-RSI health witness

This directory captures the exact JSON stdout from the repository's performance-RSI health target.

## Reproduce

From the repository root, run:

```bash
make performance-rsi-health > docs/_witnesses/issue-9774-performance-rsi-health/scorecard.json
```

## Expected current summary

- Loop health: grade `D`, score `62.6/100`, and `clean: false`.
- Performance-RSI debt: `9` total, comprising `8` `BEHIND` dimensions and `1` `UNKNOWN` dimension.
- Coverage: `15` of `16` dimensions measured.
- Dominant bottleneck: `cycle_time`.

This witness grades the health of the performance-RSI measurement and improvement loop. It does **not** prove that parent issue #9752 achieved its 100x target.
