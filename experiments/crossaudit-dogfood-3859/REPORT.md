# Cross-audit dogfood rollout — issue #3859

Date: 2026-07-15  
Run manifest: [`run-manifest.json`](run-manifest.json)  
Verified receipt ledger: [`receipts.jsonl`](receipts.jsonl)  
Scorecard: [`scorecard.json`](scorecard.json)

## Verdict

**ACTION — keep cross-audit advisory.** The bounded reciprocal backfill audited
3/5 preregistered subjects. All three receipts independently admitted author/
auditor identity. The run produced one PASS, one INCONCLUSIVE, and one REFUTE.
The REFUTE was independently confirmed by git read-back and filed as
[#4856](https://github.com/anthony-chaudhary/fak/issues/4856). The local/open-
weight and unknown-author quorum cells remain explicit `not-yet`; therefore the
background program is a dark/incomplete loop, not a production closure gate.

## Pre-registration and budget

The manifest capped the run at eight subjects and USD 3.00 external provider
spend, selected recent security/trust and runtime closures before seeing audit
outcomes, and required Claude-authored work to route to GPT, GPT/Codex-authored
work to Claude, and unknown author work to a diverse quorum. No private transcript,
credential, endpoint, or host detail is present in the public artifacts.

## Results

| Issue | Class | Author → auditor | Independence | Outcome | Independent adjudication |
|---|---|---|---|---|---|
| #3853 | accidental corpus | GPT/Codex → Claude | ADMIT | REFUTE | **Confirmed.** Git read-back shows PASS/FAIL markers are selected from the `corrupt` label and then regex-matched. Filed #4856. |
| #3854 | calibration | GPT/Codex → Claude | ADMIT | INCONCLUSIVE | No corruption counted. The bundle omitted captured `claim-check`/selfcheck outputs and truncated the relevant test patch; committed-tip CI/DOS remain green. |
| #4852 | guard transport | Claude → GPT xhigh | ADMIT | PASS | Closing patch, 40 KiB byte-preservation tests, CI/DOS references, and live Windows canary supported closure. |
| #4853 | model runtime | unknown → diverse quorum | not-yet | not-yet | Author provenance was unresolved and the single-audit CLI cannot claim quorum. |
| local/open-weight | local route | local → hosted reciprocal | not-yet | not-yet | No authenticated sanctioned-node bridge was active; no local result was fabricated. |

Receipt digests:

- #3853: `sha256:19bbf160dfa1af6991144c56668d6800bd84e9197c6d41523ba9b5746108e442`
- #3854: `sha256:d5c0f90874b1e998e1faab4c0ecbbf7b2095578b047ca1a6a4e5780924b78528`
- #4852: `sha256:4946ec177baf48e96d167f0e856c5d3ce2793608296a27ac05beadda84babeed`

`fak audit verify` independently confirmed three hash-chained rows, three unique
audits, and head hash
`sha256:3b2ad512a96a9d48910264eb82cad0eac208526dcf5dd577a65a31cff77761c5`.

## Scorecard

The shipped `loopindex.ScoreCrossAudit` fold over the same receipt envelopes
reported:

- coverage: 3/5 audited (`0.60`), two missing/not-yet;
- independence: 3/3 admitted (`1.00`) for attempted subjects;
- outcomes: PASS 1, REFUTE 1, INCONCLUSIVE 1;
- finding yield: 1/3 completed (`0.333` overall; 1.0 in the one accidental-corpus subject);
- author mix: GPT 2, Claude 1; auditor mix: Claude 2, GPT 1;
- pairwise cost telemetry: 64,916 reported Claude tokens across two receipts;
  GPT token/cost and all dollar-cost cells remain `not-yet` because the receipt
  adapter did not report them;
- mean audit latency: 34.62 seconds across three samples;
- health: `dark_loop=true`, one unavailable provider (local), two pending cells;
- grade/verdict: `F / ACTION` with debts `dark-loop`, `coverage-incomplete`, and
  `open-findings`.

This is a dogfood yield observation, **not** an accuracy estimate. The denominator
is intentionally tiny and stratified, the confirmed finding was already in a
new synthetic corpus, and model disagreement is not counted as corruption.

## Non-model finding witness

The #3853 REFUTE was checked without trusting auditor narration:

```powershell
git merge-base --is-ancestor a34dda0dc603 origin/main
dos commit-audit a34dda0dc603
git show a34dda0dc603:internal/modelroute/crossaudit_accidental_fixtures.go
```

The committed source computes `outcome := "PASS"`, changes it to `"FAIL"` under
`if corrupt`, stores `external-witness:<pair>:<outcome>`, and has the selfcheck
regex-match that generated marker. Thus the purported external witness is a
function of the ground-truth label rather than an independently executed
behavior. #4856 names the executable-witness replacement and done condition.

## Reproduction

```powershell
# Verify immutable receipts
fak audit verify experiments/crossaudit-dogfood-3859/receipts.jsonl

# Re-derive the scorecard from the exact public receipt envelopes
go run ./cmd/crossauditdogfood `
  --receipts experiments/crossaudit-dogfood-3859/receipt-envelopes.json `
  --out experiments/crossaudit-dogfood-3859/scorecard.json
```

The live audits used the committed-tip `fak issue audit` command with the checked-in
author manifests/identity roster and appended only verified receipts to the ledger.

## Limitations and rollout decision

The sample is not representative, unknown-author quorum was not exercised, local
fallback was not available, dollar cost was not receipt-bound, and no production
false-positive rate can be inferred. The one confirmed finding demonstrates useful
yield, but also demonstrates that closing issues solely from selfchecks can miss
circular ground truth. Keep the audit loop advisory, fix #4856, restore a running
background loop and local provider, then rerun a larger preregistered sample before
considering #3860 enforcement.
