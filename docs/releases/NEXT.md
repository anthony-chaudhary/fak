# fak vNext (targeting v0.51.1): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.51.1` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.51.1` (`patch` bump)
- **Base release tag:** `v0.51.0`
- **Commits in flight:** 7

## What changed

- *(No new user-visible features landed yet)*

## Reliability and correctness

- Run microcontext tool result elision before subturn yield valve check (#11636).

## Engineering quality and evidence

### Security & Governance
- Deconstruct TRUST_VIOLATION backwards tail-wagging into zero-trust capability boundaries (#11392).
- Add frontier model evolution audit for Astra and beyond.

### Autonomous Agent & Multi-Model Harness
- Empirical resource profile breakdown and roofline model (#11407).

### Serving Engine, Gateway & Kernel Acceleration
- Execute Hopper H100 kernel-lever benchmarks on GCP A3 (#10944).
- Expand benchmark suite for resolver indexing and fault paths.

### Developer Platform, Tooling & Evidence
- Add sandbox tiering and gym spec (#11534).
- Accelerated performance RSI loop and deterministic scorecard (`fak performance-rsi-scorecard`, #9752, #9779).

## Upgrade and breaking changes

- No manual migration required unless specified above.
