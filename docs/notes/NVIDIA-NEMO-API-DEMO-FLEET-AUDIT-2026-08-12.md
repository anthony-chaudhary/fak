# NVIDIA-hosted Nemo API demo fleet audit — 2026-08-11..12

> **Verdict: capability grade T4 (lowest / restricted). Not eligible for autonomous issue selection, top-level engineering, or general fleet work.**

## Name the system correctly

The run directory used the shorthand `nemotron`, but the durable name of this experiment is:

**NVIDIA-hosted Nemo API demo / OpenCode fallback fleet**

The process invoked model id `nvidia/nvidia/nemotron-3-super-120b-a12b` through the
NVIDIA-hosted OpenAI-compatible endpoint `https://integrate.api.nvidia.com/v1` using
OpenCode. This was an NVIDIA-hosted API-demo endpoint, not a locally operated Nemotron
service, not a Codex worker, and not a worker launched through `fak fleet-accounts`.
“Nemo” here names the NVIDIA model family/demo route; it is not a fak subsystem or agent
class.

## What happened

A 14-hour issue-resolution run attempted to keep about five workers active after the
original Claude seats reached provider limits. Two OpenCode config homes named for
DeepSeek V4 Pro and Kimi K2.6 pointed at NVIDIA's API endpoint. Those configured models
were unavailable, so launch commands overrode both homes with
`nvidia/nvidia/nemotron-3-super-120b-a12b`.

The fallback was started directly:

```text
XDG_CONFIG_HOME=<opencode seat> opencode run \
  --model nvidia/nvidia/nemotron-3-super-120b-a12b --format json
```

That command bypassed account resolution and launched a generic OpenCode coding harness.
It did not select a Codex-style worker even though the account roster later showed an
available tier-1 Codex account (`gpt-5.6-sol`).

## Reproducible low-level evidence

The checked-in witness is [`artifact-index.json`](../witnesses/nvidia-nemo-api-demo-2026-08-11/artifact-index.json).
It records every matching local artifact's basename, byte count, SHA-256 digest, artifact
kind, JSONL event count, and tool-call count. Raw `.goal-runs` files remain ignored because
they contain machine-local paths and large provider transcripts.

Regenerate or verify the manifest on the source machine:

```text
go run ./cmd/nvidia-nemo-api-demo-audit \
  --dir .goal-runs --pattern '*nemotron*' \
  --generated-utc 2026-08-12T17:14:39Z \
  --out docs/witnesses/nvidia-nemo-api-demo-2026-08-11/artifact-index.json

git diff --exit-code -- \
  docs/witnesses/nvidia-nemo-api-demo-2026-08-11/artifact-index.json
```

Captured totals:

| Evidence | Count |
|---|---:|
| artifacts | 263 |
| bytes | 30,509,102 |
| trajectory outputs | 59 |
| tool calls | 3,450 |
| shell calls | 3,037 |
| direct edit/write calls | 46 |
| read/search calls | 366 |

The artifact set includes prompts, output/error JSONL, worktree markers, and PID markers.
The manifest proves what was inspected without pretending that activity was delivery.

## Actual engineering result

**Verified fleet-attributed trunk ships: 0.**

The workers searched, routed issues, edited files, and ran focused tests. Independent
inspection rejected the candidate changes because they included syntax corruption,
duplicated code, source truncation, regression failures, scratch-file pollution,
forbidden frozen-ABI edits, nontrivial edits to grandfathered Python tooling, whitespace
no-ops, or changes without a reproducing test and issue witness.

Two detached commits survived as Git objects but are not ancestors of `main`:

| Commit | Effect | Shipped? |
|---|---|---|
| `131800d769` | one-line adjudicator comment typo | no |
| `766cd43f79` | arg-index test isolation edit | no |

A trajectory pushed a shared-trunk merge, but no authored issue fix from this agent was in
that merge. Campaign-era commits from other sessions are not attributed to this run.

## Capability classification

This report uses a **local dispatch capability grade**, deliberately separate from fak's
architecture tiers and from `fleet-accounts` model tiers:

| Grade | Meaning |
|---|---|
| T1 | trusted for ambiguous, repository-wide autonomous engineering |
| T2 | trusted for bounded engineering with normal witness gates |
| T3 | simple implementation only, with an explicit issue and narrow file contract |
| **T4** | **mechanical, obvious, straight-line work only; every effect independently reviewed** |

The observed NVIDIA-hosted Nemo API demo agent is **T4**. The grade follows from the
measured outcome, not the model's marketing label: 3,450 calls and 59 trajectories yielded
zero verified trunk ships, while repeated malformed edits required supervisor cleanup.

### Allowed assignments

Only assign work when all of these are true:

- exact issue number and acceptance test are supplied;
- permitted paths are enumerated;
- change is mechanical and local (for example, a literal typo or deterministic fixture);
- no architecture, issue selection, Git reconciliation, release action, or policy judgment
  is required;
- an independent higher-grade worker reviews the diff and runs the witness;
- the agent cannot commit, land, push, or close an issue directly.

### Forbidden assignments

Never use this agent for:

- selecting “top” issues or interpreting a broad backlog;
- fleet dispatch, supervision, or autonomous refill;
- top-level features, debugging, security, policy, ABI, CI/CD, or cross-package refactors;
- shared-trunk Git operations;
- deciding that its own work is complete;
- fallback substitution for a Codex/Claude coding worker merely because capacity exists.

## Why a Codex-style agent was not used

This was a routing failure, not evidence that Codex was unavailable:

1. Claude seats reached capacity.
2. The operator manually selected two OpenCode config homes.
3. Their configured NVIDIA-hosted models were stale/unavailable.
4. The command overrode the model inside OpenCode instead of asking the account router for
   another capable worker product.
5. Therefore the run inherited OpenCode's generic harness and Nemotron's behavior, rather
   than Codex's coding-agent harness, repository discipline, and account-bound launch path.

At audit time `fak fleet-accounts list --json` discovered 61 accounts: four Claude workers,
one Codex worker, and five OpenCode workers. One tier-1 Codex worker was marked available.
The direct fallback never asked that roster to resolve the task.

## Account-system defect and correction

The account system already supports Claude, Codex, and OpenCode discovery and tier-aware
routing. The campaign bypassed it. It also contained a dangerous stale default: seats named
`nim-deepseek-v4-pro` and `nim-kimi-k26` were hard-coded as tier 1 with positive route
weights, despite their configured models being unavailable and despite arbitrary model
overrides being possible through those config homes.

The companion change to `internal/fleetaccounts` demotes those legacy seat names to tier 3
(`TierOther`) and zero route weight. This is the strongest restriction the existing router
supports without inventing a fourth account tier: tier-3 accounts are reachable only when
an operator explicitly requests tier 3, never as fallback for hard or light work. The
T4 behavioral grade in this note remains stricter: even explicit use must obey the narrow
assignment rules above.

## Follow-up controls

1. [#6534](https://github.com/anthony-chaudhary/fak/issues/6534): require all fallback launches to resolve through `fak fleet-accounts`; reject raw `XDG_CONFIG_HOME=... opencode run` in fleet launch paths.
2. [#6535](https://github.com/anthony-chaudhary/fak/issues/6535): add model identity and endpoint read-back so a config-home name cannot claim DeepSeek/Kimi while the command runs another model.
3. [#6536](https://github.com/anthony-chaudhary/fak/issues/6536): add measured capability grade, `fleet_eligible`, allowed task classes, and trajectory outcome ingestion to account records.
4. As part of #6536, forbid T4 agents from issue selection, Git mutation, broad path leases, or self-witnessing completion.
5. Prefer a currently available Codex-style or Claude coding worker for engineering; if no eligible coding worker exists, stop rather than silently degrading to an API demo model.

## Bottom line

The accurate statement is not “Nemotron ran an overnight fleet.” It is:

> An NVIDIA-hosted Nemo API demo model was exercised through OpenCode after bypassing the
> account router. It generated substantial activity but no verified issue ship, so it is
> capability-grade T4 and excluded from autonomous fleet engineering.
