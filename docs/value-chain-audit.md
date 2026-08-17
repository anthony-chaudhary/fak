# Vertical value-chain audit

`fak value-chain audit` joins declared stack stages to measured turns, authoritative cost, and organization-defined outcomes. It keeps missing cost absent and reports the denominator covered by cost evidence.

```sh
fak value-chain audit \
  --manifest examples/value-chain/support-manifest.json \
  --observations examples/value-chain/support-observations.json
```

The example models one shared-context setup used by two agents. `cost_key` makes the setup a single charge rather than multiplying it per session. `pair_id` upgrades the baseline/candidate readout from observational to paired; without a shared pair ID, the command does not imply an experimental design.

The manifest is deliberately organization-defined: a support team can replace `ticket_resolved` with its own audited outcome and unit. Observation provenance should point to the billing, benchmark, CRM, or other authoritative artifact. An AgenticBench result packet can contribute its authoritative pass/fail count as an outcome observation while provider cost remains unknown until a billing join exists.

