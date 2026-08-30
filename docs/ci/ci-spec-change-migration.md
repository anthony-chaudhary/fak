---
title: "Changing a CI/CD spec — migrate every consumer safely"
description: "Why a CI/CD spec is a contract across files, and how to change one side without breaking the other later in a scheduled job or someone else's PR."
---

# Changing a CI/CD spec: migrate every consumer, and state what will happen

**Rule of thumb:** a CI/CD spec is a *contract between two sides that live in different files*.
When you change one side, the other side does **not** break loudly at edit time — it breaks
later, in a scheduled job, in someone else's PR, as a red trunk or a silently-wrong Slack card.
So every CI/CD spec change carries two obligations that are easy to skip and expensive to miss:

1. **Migrate every consumer in lockstep** — find *all* the other sides, not just the obvious one.
2. **Write down what will happen** — the impact, the cutover, and the rollback — so a reviewer
   (or the next agent) can see the blast radius without re-deriving it.

This page says what "a CI/CD spec" means here, gives the checklist, and shows a worked example.

---

## What counts as a "CI/CD spec" (the contracts that bite)

A change is a *spec* change — not just an edit — when something **outside the file you touched**
depends on the shape of what you changed. In this repo the recurring contract classes are:

| Contract | Producer side | Consumer side | Silent-break symptom |
|---|---|---|---|
| **Tool `--json` / output schema** | a repo tool (`tools/issue_triage.py --json`, `fak <verb> --json`, a step writing `$GITHUB_OUTPUT`) | a workflow step that `jq`-reads specific field paths | `jq` returns `null` → grade/threshold logic runs on garbage; card posts wrong numbers |
| **Required check / job name** | a job's `name:` / a reusable workflow's identity | `needs:`, `workflow_run.workflows`, branch-protection required checks | downstream job never runs, or PR merge is blocked on a check that no longer exists |
| **Secret / env / vars** | repo/org settings, `env:` blocks | `${{ secrets.* }}` / `${{ vars.* }}` references | step fails auth at runtime, only on the branch/schedule that uses it |
| **Runner / image / action version** | `runs-on:` labels, container images, `uses: …@vX` pins | any job assuming that capability exists | job queues forever (missing self-hosted label) or breaks on an action major bump |
| **Artifact / cache / cross-workflow** | `upload-artifact` name, cache key, a workflow's `name:` | `download-artifact` name, `workflow_run` trigger, cache restore | empty download, cache never hits, trigger never fires |
| **External service coupling** | Slack channel IDs, webhooks, scoreboard endpoints, project-board IDs | the `*-feed.yml` / `*-signal.yml` posters | posts to the wrong channel, or 404s, with a green job |

If your change touches the *left* column, someone on the *right* column has to move too.

> **The tell:** if you can't answer "who reads this?" from memory, you have a spec change on your
> hands, not a local edit. Grep for the field name / job name / secret before you assume it's local.

---

## The migration checklist

**Before you change the spec**
- [ ] **Enumerate consumers.** Grep the *whole* live tree for the field path / job name / secret /
  artifact name — across `.github/workflows/*.yml`, `tools/*.py`, Go (`cmd/`, `internal/`), and
  docs. Exclude frozen snapshots (`docs/_audits/`, `.dos/`, `_*`) but *read* everything else and
  classify each hit: **genuine consumer** (must migrate) vs **independent** (same name, own logic)
  vs **stale prose/comment** (fix if now misleading).
- [ ] **Confirm direction.** Is the producer change already landed, or landing in the same commit?
  A producer that ships ahead of its consumers is a red trunk waiting for the next scheduled run.

**When you change it**
- [ ] **Migrate producer and every consumer together**, ideally in one commit (or a tightly ordered
  pair). Never leave a `jq` reading a field the tool no longer emits.
- [ ] **Guard the parse.** Workflows here already defend integer fields (`case "$n" in ''|*[!0-9]*)
  … =0`); keep that pattern so a *future* schema gap degrades instead of exploding.
- [ ] **Keep the generated maps honest.** If you added/removed/renamed a branch/tag reference,
  regenerate `docs/ci/workflow-branch-audit.md` (`fak workflow-audit --write-doc`) — the
  `--check-doc` gate reds the trunk otherwise.

**After you change it**
- [ ] **Prove the trunk is green from the *committed* tip, not the peer-dirty tree** —
  `fak ci-preflight [--json]` archives the committed tip to a throwaway checkout and runs
  `go build ./...` + `gofmt -l` there, so a peer's uncommitted WIP can neither mask nor fabricate
  a red. Run `make ci` (build · vet · test · claims-lint; native-Windows → `./test.ps1` under WSL).
- [ ] **Write the impact statement** (below) into the PR/commit body.
- [ ] **Watch the first real run.** A schedule-only or branch-only job won't run in the PR — check
  the next scheduled fire (or trigger it) and confirm the card/artifact is correct, not just green.

---

## "State what will happen" — the impact statement

Put this in the commit/PR body for any spec change. It is three lines and it is the difference
between a reviewer trusting the change and a reviewer having to reverse-engineer it:

```
CI-SPEC CHANGE: <what contract changed, producer -> consumer>
CONSUMERS MIGRATED: <every file that reads the old shape, and how each was handled>
IMPACT / CUTOVER / ROLLBACK: <what a run does differently now; when it takes effect
  (next push? next schedule?); how to revert if the first real run is wrong>
```

Name the consumers you checked *and found not to need changing* too — "the TUI's `LikelyDup`
counter is independent and left as-is" tells the reviewer you looked, so a silent break can't hide
behind "I assumed it was unrelated."

---

## Worked example: removing `likely_dup` from the triage contract

The change: `tools/issue_triage.py` (producer) stopped emitting `counts.likely_dup` in its `--json`
output. The consumer `.github/workflows/backlog-feed.yml` reads that JSON with `jq` to build the
weekly #backlog digest card.

What the migration actually required (this is the point — it was *not* just the one workflow):

- **Producer:** drop `likely_dup` from the `counts` object in `tools/issue_triage.py`, and update
  `tools/issue_triage_test.py` so it no longer asserts the field.
- **Consumer (the workflow):** in `backlog-feed.yml`, remove the `likelydup=$(jq … .counts.likely_dup)`
  extraction, the `$GITHUB_OUTPUT` line, and the `· likely-dup …` fragment from **both** the
  dry-run and the real `--detail` string. Leave the integer-guard `for` loop covering the
  *remaining* fields.
- **Other named consumers — checked, classified, and either migrated or deliberately left:**
  `cmd/fak/tui_types.go` (`LikelyDup int \`json:"likely_dup"\``) and `cmd/fak/tui_issues_garden.go`
  (`c.LikelyDup++`), plus prose in `internal/gardenbundle/walk.go` and a prompt string in
  `tools/issue_gardener_worker.py`. Each had to be traced to decide *consumer vs independent* —
  a struct that unmarshals the tool's JSON must move; a TUI that classifies issues itself does not.
- **Frozen:** `docs/_audits/issue-triage-*.md` snapshots keep the old field — historical records,
  left as-is.

The trap this page exists to prevent: shipping the producer removal and *only* the obvious
workflow, while a second consumer keeps reading a field that is now always `null`/zero — green
build, silently wrong output, discovered days later in a scheduled card.

### The contracts most likely to bite (audit the workflows before you assume "local")

An inventory of all ~38 workflows surfaced these as the highest-risk couplings — the ones where
one side can change and red (or silently mislead) the pipeline. If your change is anywhere near
these, treat it as a spec change:

1. **`release-cadence.yml` hard-reads `decision.json["decision"]`** from `tools/release_decide.py`
   (a bracket index, not `.get`). Drop/rename that key → a `KeyError` **reds the whole release
   run**. Highest risk: it hard-fails, and the producer is an independently-evolving tool.
2. **`release_decide.py` gates "is CI green?" on the workflow *filenames* `ci.yml` / `ci-fast.yml`**
   (`FAK_RELEASE_FAST_CI_WORKFLOW`, default `ci-fast.yml`). Rename either file, or restructure
   ci-fast's decisive job, → `CI_STATE_UNKNOWN` → **auto-release silently stops shipping**, no red
   anywhere. Renaming a workflow file is a spec change.
3. **`release-container.yml` jq contract against `server.json`** (`.name`, `.version`,
   `.packages[] | select(.registryType=="oci") | .identifier/.version`). Restructure the MCP
   manifest → empty jq → **hard-red verify job** blocks the container release.
4. **`release-notify.yml` couples to `release-cadence.yml` by workflow *name* + six step *names***
   (`"Release cadence"`, `'Cut release commit'`, …). Rename any → **silent false "all clear"**:
   failures stop alerting #releases/#blockers while the pipeline looks green (fail-*open* — the
   dangerous direction). Renaming a job or step is a spec change.
5. **`issue_triage.py --json` → `backlog-feed.yml` field paths** (`.counts.*`, `.rows[].number`) —
   the worked example below. Demonstrably mutable, and the poster fails *soft* (blank card, no
   red), so drift here is invisible until someone notices a channel went quiet.

**A real, currently-open drift of this kind:** `FAK_SCOREBOARD_CHANNEL` is read as a **secret** in
`ci.yml` but as a **var** in the feeders (`scoreboard-feed.yml`, `slack-beat.yml`,
`slack-watchdog.yml`). Set only one and the other side silently drops to dry-run. Split-brain
config is a spec mismatch even though no single file is "wrong."

**Fail-soft is not fail-safe.** Most posters here guard with `// default` or
`if: … != ''` and degrade to a blank/dry-run instead of a red. That protects the *build* but hides
the *drift* — the card is quiet or wrong for days. When you change a producer, don't lean on the
consumer's soft-fail; migrate it.


---

## Where this is enforced (so it isn't just etiquette)

- `make ci` / `./test.ps1` — build · vet · test · claims-lint gate every commit/push.
- `fak ci-preflight` — committed-tip build/gofmt truth, immune to the peer-dirty shared tree.
- `fak workflow-audit --check-doc` — reds the trunk if `docs/ci/workflow-branch-audit.md` drifts
  from the branch/tag references actually in the workflows.
- The commit-msg / file-admission / trunk guards (`fak hooks …`) run automatically at commit/push.

If you change a CI/CD spec and can't point to the migrated consumers and the impact statement,
the change isn't done — it's armed.

## Current `ci-fast` queue and parallelism contract

The release-critical `ci-fast.yml` gate converges on the newest commit for each GitHub ref:
its concurrency group is `github.workflow` plus `github.ref`, with in-progress supersession
enabled. On a hot `main`, an obsolete run is cancelled so the available runner advances to the
newest head instead of spending the full test budget on every ancestor. Pull requests remain
isolated by their `refs/pull/<number>/merge` ref. Cancelled runs are not decisive release signals;
the release decision remains fail-closed until a completed fast gate is green.

The job keeps `GOMAXPROCS=2` and `GOFLAGS=-p=2`. That is bounded two-way Go package/compiler
parallelism, not a package-set change: build remains `go build ./...`, vet remains
`go vet ./...`, and the correctness step remains `go test ./...` without `-race`. Its existing
30-minute step limit, heartbeat, status file, log file, cache keys, workflow name, and job/check
name are unchanged.

**Impact:** superseded runs for the same ref stop consuming the queue, and two independent Go
packages may progress at once within the job's explicit two-way package bound. **Cutover:** the contract
takes effect on the first push containing the workflow change; that push's run may cancel an
older `main` run and becomes the head-convergent release signal. **Rollback:** restore the prior
SHA-keyed concurrency group and `-p=1`; this is safe but reintroduces ancestor retention and the
observed risk that the full package set does not complete within the existing step budget.
