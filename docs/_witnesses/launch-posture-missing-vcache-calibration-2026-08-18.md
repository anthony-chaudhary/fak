# Missing vCache calibration launch-posture witness — 2026-08-18

**Verdict:** in a temporary non-FAK Git repository with an empty private ledger, `fak doctor launch-posture --entrypoint guard --harness codex --provider openai --json` exits 1 and reports `vcache-calibration` as `inert` with the action `run a real fak guard/serve session through this provider`.

The full machine-readable report is [`launch-posture-missing-vcache-calibration-2026-08-18.json`](launch-posture-missing-vcache-calibration-2026-08-18.json). This proves launch diagnostics no longer describe a configured vCache anchor as wholly healthy when the selected provider lacks a fresh dated calibration. It does not claim that calibrated constants steer runtime policy yet.
