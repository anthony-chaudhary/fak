# Curated Policy-Template Presets

A small, **documented, versioned, round-trip-gated** set of capability-floor
starting points. Each preset is a reviewable `fak-policy/v1` manifest an adopter
copies, narrows, and witnesses with `fak policy --check` / `fak preflight`
before putting it in front of an agent. The goal is a *good allow-list* instead
of a blank one (issue #578).

> **What a preset is, and is not.** A preset encodes the **capability floor** —
> which tool *names* and *argument values* the agent may invoke. It is a
> permissions floor, not a detection guarantee: it does not make the agent "safe"
> from content-level prompt injection, only from invoking effects the allow-list
> omits. Regex `deny_regex` arg-rules are best-effort keyword guards (a
> determined agent can hide a keyword), not parsers; the load-bearing refusals
> are the **structural** ones — `DEFAULT_DENY` for any unlisted tool, and the
> in-kernel rungs (`gitgate`, self-modify, IFC) that decide on shape, not text.

## Round-trip gate (the "can't rot" property)

Every preset shipped here is enforced by
`internal/policy.TestPresetsRoundTrip`, which asserts that each manifest

1. **loads** through `ParseRuntime` — i.e. it passes `fak policy --check`, with
   every `deny` citing a closed-vocabulary reason; and
2. **round-trips exactly** — loaded to a `Policy` and re-rendered with
   `FromPolicy` (the path `fak policy --dump` uses), it reproduces the SAME
   floor, and the re-rendered bytes equal the file on disk.

A hand-edit that drifts from canonical form, or that introduces a field the
loader does not carry, fails the build instead of silently shipping a floor
different from the one reviewed. Loose, undocumented manifests that are not yet
part of this gate live in the parent [`../`](../README.md) `examples/` directory
(parse-gated by `internal/policy.TestExamplePoliciesParse`).

```bash
# Validate any preset in this pack:
go run ./cmd/fak policy --check examples/presets/coding-agent-safe.json
```

---

## `coding-agent-safe.json` — hardened coding agent (built on `gitgate`)

The one NEW preset in this pack, and the recommended starting point for a coding
agent that edits a repository through a shell (`Bash`).

**Allows.** The standard coding-agent tool surface — `Bash`, `BashOutput`,
`KillShell`, `Read`, `Edit`, `Write`, `NotebookEdit`, `Glob`, `Grep`, `LS`,
`TodoWrite`, `Task`, `WebFetch`, `WebSearch`, `ExitPlanMode`, `Skill`,
`SlashCommand` — plus the `read_`/`get_`/`search_`/… read-shaped prefixes.

**Refuses (argument-level, on `Bash.command`):**
- the **`gitgate` trunk-discipline hazards**, mirrored as `deny_regex` so the
  floor holds even with `FAK_GITGATE=off`: `push --force`/`--force-with-lease`,
  `push --no-verify`, `push --delete`, `commit --amend`, `commit --no-verify`,
  `commit --no-gpg-sign`, `commit -a/--all`, `add -A/--all`, `add -u/--update`,
  `tag -f/--force`, `tag -d/--delete`, `rebase -i/--interactive`;
- **destructive / system** commands: `rm -rf`, `sudo`, `mkfs`/`dd if=`/device
  redirect, the fork-bomb `:(){`, `curl|sh` pipe-to-shell;
- **out-of-tree writes**: any redirect/copy/`-o`/`--output` targeting `../`.

**Refuses (structural):** `exfiltrate` (→ `SECRET_EXFIL`); any write into the
kernel/policy spine (`internal/abi/`, `internal/kernel/`, …), `.git/`, `.ssh/`,
`VERSION`, `id_rsa`, `/etc/` is `SELF_MODIFY`; `password`/`secret`/`api_key`/
`token`/`authorization` args are redacted; `lint_writes: true` refuses a
whole-file write of unparseable Go/JSON with `MALFORMED` before it lands.

**Threat model.** An agent that has been steered (by prompt injection or its own
misjudgement) into rewriting shared history, force-pushing the trunk, staging a
peer's files, or wiping the worktree. The `gitgate` pairing is **defense in
depth**: the in-kernel `gitgate` rung refuses these at the call boundary with a
repairable reason; this manifest refuses them at the *policy* layer too, so the
same hazards are blocked even by a `fak` build with the rung unregistered.

```bash
go run ./cmd/fak preflight --policy examples/presets/coding-agent-safe.json \
  --tool Bash --args '{"command":"git push --force origin main"}'   # DENY  POLICY_BLOCK
go run ./cmd/fak preflight --policy examples/presets/coding-agent-safe.json \
  --tool Bash --args '{"command":"git push origin main"}'           # ALLOW (push itself is not blocked)
```

---

## Curated existing templates (by use case)

The rest of the pack groups the manifest templates already shipped in
[`../`](../README.md) by the use case they encode. Each is a documented
allow-list; copy the closest one, delete what you do not need, and witness the
most important dangerous actions with `fak preflight`.

| Use case | File | Allows | Refuses (the threat it encodes) |
|---|---|---|---|
| **Customer support (readonly)** | [`../customer-support-readonly-policy.json`](../customer-support-readonly-policy.json) | read/search/lookup + `create_support_ticket`; escalates to `transfer_to_human_agents` | `refund_payment`, `delete_account`, `transfer_funds`, `send_customer_email`, `export_customer_data` (SECRET_EXFIL) — an agent paying out, deleting accounts, or exfiltrating the customer DB |
| **Finance / booking (readonly)** | [`../flight-booking-agent-policy.json`](../flight-booking-agent-policy.json) | search/book/read flights | `refund_payment`, `cancel_booking`, PNR export, fund transfer — plus a `deny_regex` price cap; an agent refunding or transferring funds without a human |
| **Healthcare (PHI)** | [`../healthcare-phi-policy.json`](../healthcare-phi-policy.json) | read EHR / search ICD / drug-interaction / notes / appointments | `export_patient_data` (SECRET_EXFIL), `email_phi`, record delete; heavy `redact_fields`; trusted-EHR vs untrusted-inbox provenance |
| **DevOps (dry-run only)** | [`../devops-dryrun-policy.json`](../devops-dryrun-policy.json) | plan / diff / template / `kubectl get` / validate | `terraform_apply`, `deploy_production`, `kubectl_delete`/`exec`, `drop_database`, `shell` — an agent applying infra or running prod commands |
| **Repo guard (coding, structural)** | [`../repo-guard-policy.json`](../repo-guard-policy.json) | the coding-agent tool surface | destructive shell + out-of-tree writes via `deny_regex`; the structural `gitgate`-paired coding floor above is its documented, round-trip-gated successor |

### Honest fences

- **Stateless limits.** Some trunk laws (`OFF_TRUNK`, the shared-tree staging
  sweep, a peer's in-flight `MERGE_HEAD`) depend on repo *state*; a stateless
  manifest cannot decide them. The `gitgate` rung deliberately DEFERS on those
  (they stay with the witness resolver and the git hooks); this preset's regexes
  inherit the same boundary.
- **Regex is not structure.** The `deny_regex` rules here are the textual
  analogue of a structural check — they catch the obvious form of a hazard. The
  load-bearing refusals are structural: an unlisted tool is `DEFAULT_DENY`, and a
  write into the spine is `SELF_MODIFY`. Treat the regexes as defense-in-depth,
  not a complete parser.
