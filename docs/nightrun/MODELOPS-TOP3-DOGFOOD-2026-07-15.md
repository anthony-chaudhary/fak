# Exact-model canary gate dogfood — 2026-07-15

Status: **ROLLBACK exercised; production promotion remains HOLD.**

This is the checked-in readout required by #4794. It records one clean-checkout run of the exact-model operations spine against this repository's checked-in top-three policy and outcome data. It is an operational dogfood witness, not a claim that the three-model portfolio is production-ready.

## Reproduction

Source revision: `87e176c0416865e024d974ede2176749e47de4da`

```powershell
git archive 87e176c0416865e024d974ede2176749e47de4da | tar -xf - -C $clean
Push-Location $clean
go build -o fak-dogfood.exe ./cmd/fak
./fak-dogfood.exe model canary-gate --input examples/modelops-top3-canary.json > modelops-decision.json
$LASTEXITCODE # 3: typed ROLLBACK
Pop-Location
```

Input SHA-256: `53c39ce19a69cea2e442ee8deb8d016ca0a12ae6d1bcf778a6db2a8fae7850b3`

Decision SHA-256: `e6c31cd478069b5d5ff7d8b9578a38d23b7f1b00614d6ba114078fdf5c90b59`

Stderr was empty. The command selected the first healthy capability-safe fallback and returned the documented rollback exit status rather than silently treating rollback as success.

## Captured decision

```json
{
  "schema": "fak.modelops.canary-decision/1",
  "action": "ROLLBACK",
  "candidate": "claude-opus-4-8",
  "selected": "claude-sonnet-4-6",
  "required_tier": 1,
  "reasons": [
    "success_rate 0.5000 < 0.9500",
    "selected first healthy capability-safe fallback"
  ],
  "outcome_counts": [
    {
      "model": "claude-opus-4-8",
      "promote": 0,
      "rollback": 1,
      "hold": 1,
      "total": 2
    },
    {
      "model": "claude-sonnet-4-6",
      "promote": 2,
      "rollback": 0,
      "hold": 0,
      "total": 2
    }
  ]
}
```

## Readout

- Exact IDs remain separate in both observations and outcome counters; tier aliases are not used as attribution keys.
- The Opus candidate violates the declared success-rate SLO, so the gate rolls back to Sonnet.
- Sonnet satisfies the required capability tier and all declared thresholds in this fixture.
- Haiku is not selected for a tier-1 request, preserving the fail-closed capability floor.
- Invocation outcome replay is represented by stable invocation IDs and counted once by the shipped evaluator.

The separately witnessed 40-run exact-ID capability campaign remains `HOLD`; this synthetic fault drill does not override that result or promote production traffic. Its scrubbed campaign readout is tracked in #4633/#4813.

## Defect disposition

This dogfood run surfaced no new implementation defect: decoding, exact-ID attribution, threshold evaluation, fallback selection, typed exit status, and JSON rendering matched the checked-in contract. The remaining production gaps are acceptance work already tracked by parent #4634 (live alert/dashboard ownership and a live provider-failure drill); they are not silently converted into a PASS here.
