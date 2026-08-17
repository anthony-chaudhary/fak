# Vertical value-chain audit

`fak value-chain audit` joins declared stack stages to measured turns, authoritative cost, and organization-defined outcomes. It keeps missing cost absent and reports the denominator covered by cost evidence.

```sh
fak value-chain audit \
  --manifest examples/value-chain/support-manifest.json \
  --observations examples/value-chain/support-observations.json \
  --selfcheck \
  --expect examples/value-chain/support-witness.txt
```

The example models one shared-context setup used by two agents. `cost_key` makes the setup a single charge rather than multiplying it per session. `pair_id` upgrades the baseline/candidate readout from observational to paired; without a shared pair ID, the command does not imply an experimental design.

The manifest is deliberately organization-defined: a support team can replace `ticket_resolved` with its own audited outcome and unit. Observation provenance should point to the billing, benchmark, CRM, or other authoritative artifact. An AgenticBench result packet can contribute its authoritative pass/fail count as an outcome observation while provider cost remains unknown until a billing join exists.


## Evidence rules

- `cost_usd` is optional. Missing cost stays unknown; derived `$ / turn` and `$ / outcome` rates appear only when billing evidence covers every measured turn in the arm.
- Shared setup cost uses `cost_key` so one setup is charged once rather than once per session.
- `pair_id` marks matched baseline/candidate units. Without matches, the comparison is explicitly observational.
- `--agentic-packet` accepts only a fully graduated `fak.agentic-benchmark-result-packet.v1`: benchmark-native parity, official grader, raw/fak arms, all metric categories, and checked evidence artifacts must be declared in addition to `PASS_RESULT`. The packet must carry explicit `value_chain` observations; no outcome is inferred from a headline status bit.
- Negative turns, cost, and outcomes fail closed.

The checked-in `--selfcheck` command is the minimal end-to-end spine witness: it runs the real CLI, reads the organization-defined manifest and observations, and compares the emitted report byte-for-byte with `support-witness.txt`.
