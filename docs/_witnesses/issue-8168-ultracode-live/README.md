# Issue #8168 live paired artifact

Verdict: **ABSTAIN**. This is a live provider run, not the offline selfcheck. It supports no Ultracode value claim because effective treatment activation was not observable, the fleet outcome had no joined effect/witness/reconciliation receipt, the declared token envelope was not enforced, and subscription spend was unavailable.

The control and treatment read the same archived tree and public task with `gpt-5.6-sol`, xhigh reasoning, the same Windows/x64 environment, and the same declared 180-second / 65,536-token / $1 envelopes. Both returned the same 3/3 accepted effects with zero contradictions. The fleet was slower (34.576s versus 16.277s) and used more provider-reported input/output tokens (237,724 versus 59,371); cache share rose from 65.01% to 73.21%, which is not a token or value gain. Those observations remain descriptive because billed tokens, spend, and activation were not verified.

## Replay the captured pair

From the repository root:

```powershell
fak ultracode bench --pair docs/_witnesses/issue-8168-ultracode-live/pair.json --json
go test ./docs/_witnesses/issue-8168-ultracode-live
```

The first command must return `ABSTAIN`. `pair.json` preserves the legacy numeric compatibility projection, where `billed_tokens` contains provider total input-plus-output tokens and `spend_usd: 0` does not prove zero charge. Its joined `fak.ultracode.accounting.v1` receipts are authoritative: raw input/output/cache axes remain provider-usage observations, while billed tokens and spend are typed `unavailable`. The evaluator ignores the compatibility placeholders for cost verdicts and reports no numeric cost gain.

## Rerun the live campaign after #8488

Use a clean `git archive` of one committed tree for each arm; do not run against the peer-dirty checkout. Allocate an external scratch directory, read the public task from `task.txt`, and run the control through the same `fak guard` / Codex model route:

```powershell
$issue8168Task = (Get-Content -Raw docs/_witnesses/issue-8168-ultracode-live/task.txt).Trim()
$issue8168Scratch = Join-Path ([IO.Path]::GetTempPath()) ("fak-issue-8168-" + [guid]::NewGuid())
$issue8168Audit = Join-Path $issue8168Scratch single-guard.audit.jsonl
New-Item -ItemType Directory -Path $issue8168Scratch | Out-Null
fak guard --codex-loop-gate off --provider openai-responses --audit $issue8168Audit --expose-profile headless -- codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --dangerously-bypass-hook-trust --json -c 'model="gpt-5.6-sol"' -c 'model_reasoning_effort="xhigh"' $issue8168Task
```

From the second archive of the identical tree, run the treatment and read its terminal status:

```powershell
$issue8168Task = (Get-Content -Raw docs/_witnesses/issue-8168-ultracode-live/task.txt).Trim()
fak ultracode --task-text $issue8168Task --max-workers 4 --max-tokens 65536 --launch --json
fak ultracode status --json
```

Reduce only `turn.completed.usage`, elapsed timing, and the final public JSON answer into the redacted arm receipts. Join #8488's activation coverage, update `pair.json`, then run the captured-pair and artifact-test commands above. The campaign remains ABSTAIN unless every gate below passes.

Replace the redacted receipts and pair only when all of these gates pass:

- task digest, model, committed tree, environment, and declared wall/token/spend envelopes are equal;
- the launcher enforces the envelopes or records an ABSTAIN when either arm exceeds one;
- every treatment child has an effective-activation receipt and coverage is 3/3;
- both arms have provider-authoritative input/output/cache/billed-token/spend receipts, or unavailable accounting forces ABSTAIN;
- accepted effects are independently graded, deduplicated, reconciled, and contradiction-counted;
- retries, exits, unequal outcomes, missing witnesses, or noisy timing force ABSTAIN.

Raw logs are intentionally excluded because they can contain private harness context. The committed receipts retain only timings, token counters, accepted public answers, and SHA-256 evidence bindings; they contain no local paths, account, host, credential, session, or run identifiers.
