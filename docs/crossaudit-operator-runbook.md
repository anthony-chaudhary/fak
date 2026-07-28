---
title: "Independent cross-model issue audits"
description: "The operator front door for fak's reciprocal audit program: how an authored change routes to a family-diverse auditor, and why a display name is not identity."
---

# Independent cross-model issue audits

This runbook is the public operator front door for the reciprocal audit program in
[#3846](https://github.com/anthony-chaudhary/fak/issues/3846). It covers the shipped
`fak issue audit`, `fak issue audit-loop`, `fak issue finding`, and `fak audit
verify` surfaces. It does **not** launch a worker or grant a model authority to
close, edit, or override an issue.

The trust rule is simple: the auditor must be independently identified and
admitted. A Claude-authored change routes to a configured GPT-family or
open-weight auditor; a GPT/Codex-authored change routes to a configured Claude-
family or open-weight auditor; an unknown author requires a provider/family-
diverse quorum in the routing layer. A model's display name is not evidence of
identity.

## Public/private boundary

Keep these in the public repository:

- model/provider/family aliases and non-secret capability metadata;
- generic OpenAI-compatible endpoint examples such as `http://127.0.0.1:8080/v1`;
- author manifests, bounded audit bundles, typed receipts, and scrubbed findings;
- ledger hashes and reason codes.

Keep credentials, private endpoints, hostnames, bridge commands, and raw private
evidence out of public files and receipts. Put API keys only in an operator-chosen
environment variable and pass its **name** with `--auditor-api-key-env`. For lab
hardware, begin at [the public private-channel stub](private-comms-channel.md),
which points to the private companion repository. If evidence cannot be made
public safely, record it as unavailable/private and do not paste it into a finding.

## 1. Inspect the shipped command contracts

Run these from the repository root. Help exits nonzero because these commands use
Go's flag usage path; the emitted flag list is the contract.

```powershell
.\fak.exe issue audit --help
.\fak.exe issue audit-loop --help
.\fak.exe issue finding --help
.\fak.exe audit --help
```

Do not copy model IDs from this document as availability promises. Replace every
`EXAMPLE_*` value with an identity present in your configured roster.

## 2. Declare author provenance and the identity roster

An author manifest binds the resolving commit/range to the observed author
identity and names independently readable evidence:

```json
{
  "schema": "fak-crossaudit-author/v1",
  "author": {
    "harness": "claude-cli",
    "provider": "example-claude-provider",
    "family": "example-claude-family",
    "model": "EXAMPLE_AUTHOR_MODEL",
    "endpoint_class": "hosted-cli",
    "account_class": "cli-auth",
    "reasoning_posture": "unknown"
  },
  "source_evidence": [
    {"kind": "git-commit", "ref": "COMMIT_SHA"}
  ],
  "commit_range": "COMMIT_SHA"
}
```

The identity roster resolves operator-facing aliases to canonical provider,
family, model, and weights facts. It is configuration, not model output:

```json
{
  "schema": "fak-audit-identity-roster/v1",
  "aliases": [
    {
      "alias": "EXAMPLE_GPT_AUDITOR",
      "canonical_model": "EXAMPLE_GPT_AUDITOR",
      "provider": "example-gpt-provider",
      "family": "example-gpt-family",
      "weights_revision": "configured-revision",
      "provenance_source": "operator-roster"
    },
    {
      "alias": "EXAMPLE_CLAUDE_AUDITOR",
      "canonical_model": "EXAMPLE_CLAUDE_AUDITOR",
      "provider": "example-claude-provider",
      "family": "example-claude-family",
      "provenance_source": "operator-roster"
    },
    {
      "alias": "EXAMPLE_LOCAL_AUDITOR",
      "canonical_model": "EXAMPLE_LOCAL_AUDITOR",
      "provider": "local",
      "family": "example-open-weight-family",
      "weights_revision": "configured-content-digest",
      "provenance_source": "operator-roster"
    }
  ]
}
```

Never claim independence merely because harnesses differ. Admission compares
provider/family and the configured identity axes. Unknown or conflicting author
identity must remain typed as unresolved/conflicting; it is not permission to
choose the cheapest single auditor.

## 3. Dry-run the bounded evidence route

`--bundle-only` performs the credential-free GitHub/git read and emits the exact
bounded bundle an auditor would receive. It does not call a model and does not
append a receipt:

```powershell
.\fak.exe issue audit `
  --issue ISSUE_NUMBER `
  --repo OWNER/REPO `
  --bundle-only `
  --bundle-commit COMMIT_SHA `
  --json > _scratch/crossaudit-bundle.json
```

Read the bundle before live inference. Confirm the issue, resolving commit,
diff digest, and bounded evidence are the intended subject. A missing closing
commit is recovered by supplying the independently verified `--bundle-commit`;
it is not guessed from issue prose.

For the durable loop, a snapshot is also dry-run by default. This example plans
two eligible subjects and one typed ineligible subject without creating the
ledger or cursor:

```json
[
  {"issue_number": 100, "marker_key": "example-100", "risk": "high", "eligible": true},
  {"issue_number": 101, "marker_key": "example-101", "risk": "medium", "eligible": true},
  {"issue_number": 102, "eligible": false, "ineligible_reason": "not a closed leaf"}
]
```

```powershell
.\fak.exe issue audit-loop `
  --snapshot _scratch/crossaudit-snapshot.json `
  --ledger _scratch/crossaudit-receipts.jsonl `
  --cursor _scratch/crossaudit-cursor.json `
  --batch-cap 2 --scan-cap 500 --json
```

A dry-run report has `dry_run: true`; it must not create the ledger or cursor.
The loop command currently exposes planning/status/replay. Live auditor execution
is supplied by the scheduled host integration, not armed by a hidden CLI flag.

## 4. Run one reciprocal audit

### Claude-authored work to configured GPT/Codex

The CLI driver requires an explicit reasoning posture. Use `xhigh` when the
configured Codex model supports it:

```powershell
.\fak.exe issue audit `
  --issue ISSUE_NUMBER --repo OWNER/REPO `
  --author-manifest _scratch/author.json `
  --identity-roster _scratch/identity-roster.json `
  --auditor example-gpt-provider/example-gpt-family/EXAMPLE_GPT_AUDITOR `
  --auditor-driver codex --auditor-reasoning xhigh `
  --auditor-weights configured-revision `
  --ledger _scratch/crossaudit-receipts.jsonl --json
```

### GPT/Codex-authored work to configured Claude

```powershell
.\fak.exe issue audit `
  --issue ISSUE_NUMBER --repo OWNER/REPO `
  --author-manifest _scratch/author.json `
  --identity-roster _scratch/identity-roster.json `
  --auditor example-claude-provider/example-claude-family/EXAMPLE_CLAUDE_AUDITOR `
  --auditor-driver claude --auditor-reasoning high `
  --ledger _scratch/crossaudit-receipts.jsonl --json
```

### Local/open-weight fallback

An OpenAI-compatible local service uses the HTTP driver. HTTP intentionally does
not accept `--auditor-reasoning`; the receipt records `provider-default`:

```powershell
.\fak.exe issue audit `
  --issue ISSUE_NUMBER --repo OWNER/REPO `
  --author-manifest _scratch/author.json `
  --identity-roster _scratch/identity-roster.json `
  --auditor local/example-open-weight-family/EXAMPLE_LOCAL_AUDITOR `
  --auditor-driver http `
  --auditor-endpoint http://127.0.0.1:8080/v1 `
  --auditor-weights configured-content-digest `
  --ledger _scratch/crossaudit-receipts.jsonl --json
```

For a hosted HTTP endpoint, set a credential only in the process environment and
name that variable, for example `CROSSAUDIT_API_KEY`; never put its value in argv,
JSON, shell history, or a committed file:

```powershell
$env:CROSSAUDIT_API_KEY = '<set outside committed files>'
.\fak.exe issue audit ... `
  --auditor-driver http `
  --auditor-endpoint https://api.example.invalid/v1 `
  --auditor-api-key-env CROSSAUDIT_API_KEY
```

## 5. Budget and cost controls

`fak issue audit` bounds one inference with `--auditor-timeout` (default ten
minutes). The durable loop bounds discovery and work with `--scan-cap` (maximum
500) and `--batch-cap` (default eight). The routing API additionally evaluates
configured token estimates, per-million-token prices, health, capacity, cooldown,
tier, effort, and quorum diversity before admitting a candidate.

There is no CLI flag that silently exceeds an operator's monetary budget. Enforce
the account/provider spend ceiling outside the process, keep `batch-cap` small,
and inspect the dry-run bundle/snapshot first. Provider unavailability is not a
reason to weaken independence: use a rostered local candidate, wait for cooldown,
or leave the audit unsettled.

## 6. Verify receipts and inspect status

A live audit with `--ledger` appends only a `Verify()`-valid receipt to the
hash-chained `fak-crossaudit-receipt-ledger/v2` ledger. Independently read it back:

```powershell
.\fak.exe audit verify _scratch/crossaudit-receipts.jsonl
.\fak.exe issue audit-loop `
  --status `
  --ledger _scratch/crossaudit-receipts.jsonl `
  --cursor _scratch/crossaudit-cursor.json `
  --json
```

`fak audit verify` must report `OK`, the row count, unique-audit count, verdict
counts, and head hash. `TAMPERED/BROKEN` is terminal evidence failure: retain the
file for diagnosis and do not fold findings from it.

Loop states are typed:

- `ADVANCING`: at least one eligible issue reached terminal `PASS` or `REFUTE`;
- `WAIT`: no eligible work is due now, or a dry-run has plannable work;
- `STALLED`: eligible work exhausted its retry budget without a terminal receipt;
- `DRAIN`: discovery returned no eligible work and no retries remain.

`UNAVAILABLE` and `INCONCLUSIVE` do not advance the cursor. They enter bounded
retry state; after the configured maximum they appear in the cursor's dead-letter
set and the loop reports `STALLED`. Use `--replay ISSUE_NUMBER` only after fixing
the cause. Replay is an explicit, visible re-audit request, not deletion of the
prior receipt.

## 7. Review and project findings

A verified `PASS` supports the audited claim; it never overrides failing tests,
DOS witnesses, policy denial, or git ancestry. A `REFUTE` is actionable only after
reviewing its evidence and production envelope. Plan finding work without GitHub
mutation first:

```powershell
.\fak.exe issue finding `
  --ledger _scratch/crossaudit-receipts.jsonl `
  --target-envelope 'production target' `
  --witnessed-envelope 'currently witnessed scope' `
  --parent-baseline-points BASELINE_POINTS `
  --completion-standard production `
  --dedupe-cap 500 --max-apply 10 --json
```

Only after reviewing the plan should an operator add `--live`. Live mode requires
the bounded dedupe proof and mutation cap and uses the governed issue actuator.
Do not paste private evidence into the generated issue.

## 8. Typed recovery table

| Observation | Typed reason/state | Recovery |
|---|---|---|
| Author identity is absent or contradictory | `IDENTITY_UNRESOLVED`, `IDENTITY_CONFLICT`, `NO_ROUTE_AUTHOR_IDENTITY_CONFLICT` | Repair the author manifest/roster from independent evidence; unknown authors require a diverse quorum. |
| Candidate resolves to the author family or fails configured axes | `INDEPENDENCE_NOT_ADMITTED`, `NO_ROUTE_NO_INDEPENDENT_AUDITOR` | Select a rostered different-family/provider candidate; never relabel the same model. |
| Unknown author lacks a diverse quorum | `QUORUM_NOT_DIVERSE`, `NO_ROUTE_DIVERSIFIED_QUORUM` | Add healthy rostered candidates from distinct families/providers or wait. |
| Provider is down or health is unknown | `PROVIDER_UNHEALTHY`, `PROVIDER_HEALTH_UNKNOWN`, `NO_ROUTE_NO_HEALTHY_PROVIDER`, receipt `UNAVAILABLE` | Use an admitted local/open-weight fallback or wait; retain retry state. |
| Capacity/cooldown blocks a candidate | `CAPACITY_SATURATED`, `CAPACITY_UNKNOWN`, `COOLDOWN_ACTIVE`, `COOLDOWN_UNKNOWN`, `NO_ROUTE_NO_CAPACITY`, `NO_ROUTE_COOLDOWN` | Reduce the bounded batch, wait, or choose another admitted candidate. |
| Candidate cannot satisfy tier/effort | `BELOW_TIER_FLOOR`, `BELOW_EFFORT_FLOOR`, `NO_ROUTE_TIER_FLOOR`, `NO_ROUTE_EFFORT_FLOOR` | Choose a capable rostered auditor; do not lower a high-risk floor to clear the queue. |
| Evidence is private or cannot be scrubbed | receipt `UNAVAILABLE` / unresolved audit | Keep the evidence in the private companion, publish only a scrubbed digest/reference, or leave the audit unsettled. |
| Receipt ledger verification fails | `TAMPERED/BROKEN` | Quarantine the ledger, diagnose from an immutable copy, and do not consume its findings. |
| Retries are exhausted | loop `STALLED` plus dead-letter cursor entry | Correct the typed cause, then explicitly `--replay` the issue. |

## Break glass and override

The current audit commands expose **no break-glass bypass** for identity,
independence, receipt verification, or structural denial. That absence is
intentional. An emergency operator may proceed with the underlying operational
work only through its separately governed actuator, while recording that the
audit is missing/unavailable in the durable decision trail. Do not forge `PASS`,
edit a receipt, delete a dead-letter row, or use same-family output as an
independent audit. A future high-risk closure gate must make any override explicit,
time-bounded, and audited; this runbook does not claim that gate is already on.

## Fresh-operator completion check

Before relying on an audit, verify all of the following:

- the bundle-only dry-run names the intended issue, resolving commit, and digest;
- author and auditor aliases resolve through the authoritative roster;
- provider/family independence (or unknown-author diverse quorum) is admitted;
- inference is bounded by timeout, batch, scan, and external spend controls;
- no credential, private endpoint, hostname, or raw private evidence is persisted;
- `fak audit verify` confirms the receipt ledger after append;
- loop status has no unexplained `STALLED`/dead-letter subjects;
- findings are reviewed in dry-run before any bounded `--live` projection;
- model output never overrides structural tests, policy, DOS, or git witnesses.
