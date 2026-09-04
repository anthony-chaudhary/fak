# POLICY.md — the deployable capability floor

> **`fak`'s thesis is "permissions as the floor."** This file is how you *deploy*
> that floor: the set of tools your agent may call is a **declarative manifest you
> edit and a reviewer can diff** — not a Go literal you fork the kernel to change.

Security is the secondary benefit here, not the headline: this floor rides along
on the *same* checkpoint that delivers the token-savings, KV-cache reuse, and
right-model-per-call wins — so this page just documents the safety you get for
free on that seam.

In v0.1 the floor was `adjudicator.DefaultPolicy()`, a compiled-in Go table.
Adopting `fak` meant editing Go and recompiling. The policy manifest closes that
gap: `fak` loads the floor from a JSON file at startup, so a coding agent, an
ops bot, and a customer-support agent each ship a *different* manifest against
the *same* binary.

## The workflow: dump → edit → check → load

```bash
# 1. Dump the built-in default as a starting point.
fak policy --dump > policy.json

# 2. Edit policy.json — add the tools your agent legitimately needs,
#    deny the irreversible ones, keep everything else default-denied.

# 3. Validate BEFORE it gates a run: every deny must cite a closed-vocabulary
#    reason, no unknown keys, a known schema version.
fak policy --check policy.json

# 4. Run with it. The floor is now your file, not the binary's default.
fak agent     --policy policy.json --offline
fak run       --policy policy.json --trace trace.json
fak preflight --policy policy.json --tool delete_account --args '{}'
```

Long-lived gateways can reload that same file without dropping the process,
warm vDSO cache, or IFC ledger:

```bash
fak serve --policy policy.json --addr 127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/v1/fak/policy/reload
```

If `--require-key-env` is set, the reload route requires the same bearer token as
the other `/v1/fak/*` routes.

The same served lifecycle surface can clear one trace's IFC high-water mark after
an operator-approved session boundary:

```bash
curl -X POST http://127.0.0.1:8080/v1/fak/trace/reset \
  -H 'Content-Type: application/json' \
  -d '{"trace_id":"gw-123"}'
```

`fak preflight --policy policy.json --tool NAME --args JSON` is the per-call
oracle: it prints the exact verdict (`ALLOW` / `DENY` + reason) your manifest
gives one tool call — the cheapest way to answer *"does my policy let X
through?"* before deploying.

## The manifest schema (`fak-policy/v1`)

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow":        ["search_web", "create_ticket"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny":         { "delete_account": "POLICY_BLOCK", "exfiltrate": "POLICY_BLOCK" },
  "self_modify_globs": [".git/", ".dos/", "policy.json"],
  "redact_fields":     ["password", "secret", "api_key", "token"],
  "egress": { "deny_hosts": ["metadata.corp.internal"], "research_allow_hosts": ["arxiv.org", "docs.python.org"] },
  "rate_limit":   { "max_calls": 50, "max_cost": 0, "key": "trace", "retry_after_ms": 1000 }
}
```

| Field | Meaning |
|---|---|
| `version` | Schema tag. Omit it (current is assumed) or set `fak-policy/v1`. A different **major** (e.g. `fak-policy/v2`) is refused; a newer v1 **minor** — written `fak-policy/v1.x`, e.g. `fak-policy/v1.3`, and matched by the `fak-policy/v1` prefix — is forward-accepted, so any binary that speaks v1 tolerates any v1-minor manifest (there is no per-minor support matrix). |
| `posture` | Adjudication posture: `default_open` (default for `fak guard`), `fail_closed` (strict default-deny; default for policy manifests), or `admit_and_log` (audit/telemetry mode). See [Posture modes](#posture-modes) below. |
| `allow` | Tool names affirmatively permitted (exact match). |
| `allow_prefix` | A call is permitted if its tool name **starts with** any of these — the read-only family (`read_`, `get_`, `search_`, …). |
| `deny` | Explicit provable refusals: `tool → reason`. The reason **must** be a name from the closed refusal vocabulary (below), and it is a **static label** stamped on the refusal — never a runtime condition. A `deny` entry refuses *every* call to that tool name unconditionally; picking a detector-shaped code like `SECRET_EXFIL` does **not** make the deny fire only when a secret is present (that taint-conditional path is the live detector, not this static map). Prefer a structural code such as `POLICY_BLOCK` here so the label-not-condition reading is obvious. |
| `self_modify_globs` | Path fragments that prove a `SELF_MODIFY` attempt (the agent editing its own kernel/config). Checked on **both** write paths: a write-shaped call's target *argument* (`Edit`/`Write`), **and** a shell write whose target lives *inside the command string* (`Bash`: `sed -i`, a `>`/`>>` redirect, `tee`, `git apply`/`git checkout`, an in-place `perl -i`/`ruby -i`/`awk -i`, `python -c`/`node -e` inline writes, `find … -delete`, archive extraction). A shell *read* of a guarded file (`cat`/`grep`) is not a self-modify. |
| `redact_fields` | Arg keys whose value is stripped (`[REDACTED]`, a `TRANSFORM`) before dispatch — secret hygiene at the call boundary. |
| `egress` | Network-egress policy. `deny_hosts` adds exact host/IP refusals on top of the hardwired cloud-metadata/link-local block. `research_allow_hosts` is an opt-in `WebFetch` research path: only `http://` or `https://` URLs whose host exactly matches or is a subdomain of one of these entries are allowed, non-matching hosts are refused with `POLICY_BLOCK`, and fetched bytes still enter result admission as untrusted data. |
| `arg_rules` | Per-tool **argument-value** denials: a list of `{ "tool", "arg", "deny_regex", "reason" }`. If an allow-listed `tool`'s decoded string `arg` matches `deny_regex` (RE2 — no backreferences), the call is refused with `reason` (a closed-vocabulary code). Regex-only and best-effort — it inspects one decoded string, not the resolved effect — but enough to deny `rm -rf`, `git push`, or a write whose path escapes the repo (`-o ../…`). See [`examples/dogfood-claude-policy.json`](examples/dogfood-claude-policy.json) and [`examples/repo-guard-policy.json`](examples/repo-guard-policy.json); the path-resolving structural complement is [`tools/repo_guard.py`](tools/repo_guard.py) (see [`docs/repo-guard.md`](docs/repo-guard.md)). |
| `rate_limit` | Declarative throughput/cost cap (issue #699). An object `{ "max_calls", "max_cost", "key", "retry_after_ms" }` applied to the governor at boot and on `--policy` hot-reload. `max_calls` is a per-key admitted-call quota, `max_cost` a cumulative-cost budget (arg bytes ≈ tokens); set either or both (at least one is required). `key` is the bucketing dimension `trace` (default) / `tool` / `global`. An over-cap call is refused with `RATE_LIMITED`, whose disposition is `WAIT` carrying an advisory `retry_after` — back off like HTTP 429, not a reservation (this is a fixed-ceiling quota with no time window, so the hint is advisory; `retry_after_ms` overrides the default). Omit the block entirely to leave the limiter inert. The `FAK_RATELIMIT_*` env vars are the fallback when no `--policy` is given; a policy load is authoritative over them. |

<a id="posture-modes"></a>
## Posture modes: `default_open`, `fail_closed`, and `admit_and_log`

The policy adjudicator operates under one of three posture modes:

| Posture | Behavior on unlisted tools | Default in |
|---|---|---|
| `default_open` | **Admitted** (`ALLOW` with `posture=default_open`). Explicit denies, arg rules, self-modification, egress restrictions, and the Dangerous Gotchas catalog remain fail-closed. | `fak guard` (when no explicit policy manifest is specified) |
| `fail_closed` | **Refused** (`DEFAULT_DENY`). Only tools explicitly listed in `allow` or matching `allow_prefix` (or permitted by `egress.research_allow_hosts`) may execute. | Policy manifests (`fak policy`, `fak preflight`) |
| `admit_and_log` | **Admitted for read-shaped queries** (`read_`, `get_`, `search_`, `list_`, `lookup_`, `find_`, `calc`, `calculate`) with metadata `would_deny=DEFAULT_DENY`. Write-shaped calls and explicit denials still fail closed. | Opt-in via manifest or `--posture admit_and_log` |

### `default_open`: Developer convenience bounded by the safety floor

`fak guard` defaults to `default_open`. Under this posture, unlisted benign tools (custom scripts, CLI helpers like `gh` or `jq`, ad-hoc MCP server tools, and local utilities) are admitted automatically without requiring developers or operators to maintain an exhaustive allowlist of every command an agent might invoke.

At the same time, `default_open` does **not** grant unrestricted execution. It strictly enforces:
1. **The compiled-in FROZEN safety floor**: SSRF/cloud-metadata egress blocks (`169.254.169.254`, `fd00:ec2::254`), out-of-tree write escapes (`OUT_OF_TREE_WRITE`), and secret redactions cannot be weakened or bypassed by any posture.
2. **Explicit deny rules and argument predicates**: Rules defined in `deny`, `arg_rules`, and `self_modify_globs` are enforced unconditionally.
3. **The Dangerous Gotchas catalog**: Known catastrophic command patterns are blocked fail-closed before any allow verdict is minted.

This design implements the doctrine of **positive workspace management**: construct an affirmative, productive environment bounded by an un-bypassable structural floor, rather than trapping agents in punitive default-deny doom-loops. See [`docs/positive-workspace-management.md`](docs/positive-workspace-management.md) for the foundational architecture.

### The Dangerous Gotchas catalog

Under `default_open`, before any command runs, `fak guard` inspects tool arguments (including shell command strings passed to `Bash` or `PowerShell`) against the fail-closed Dangerous Gotchas catalog (`internal/adjudicator/gotchas.go`). Matching calls are immediately denied with `POLICY_BLOCK` (`by: monitor/gotchas`):

| Gotcha category | ID | Blocked patterns | Remedy / Alternative |
|---|---|---|---|
| **Destructive file deletions** | `destructive_deletion` | Recursive or forced deletion (`rm -rf`, `Remove-Item -Recurse -Force`, `shred`, `truncate`, `srm`) targeting workspace roots or project directories. | Delete individual files without `-r`/`-f`, move to trash, or delete inside declared scratchpad roots. |
| **Raw disk/volume destruction** | `raw_disk` | Block device formatting or partition table manipulation (`mkfs.*`, `fdisk`, `dd` writing directly to `/dev/sd*` or raw disk devices). | Raw storage and block volume operations require human operator approval. |
| **Host/shell evasion** | `host_shell_evasion` | Pipe-to-shell remote code execution (`curl ... \| sh`, `wget ... \| bash`), fork bombs (`:(){ :\|:& };:`), and subshell escape vectors. | Download scripts to disk and inspect before execution; avoid piping unverified network streams directly to interpreters. |
| **Privilege escalation** | `privilege_escalation` | Local privilege escalation commands (`sudo`, `doas`, `su`, PowerShell `Start-Process -Verb RunAs`). | Execute without elevation or ask the operator to perform privileged setup steps out-of-band. |
| **Cloud/infra teardown** | `infra_teardown` | Blanket cloud or cluster resource teardown (`terraform destroy` without read-only plan, `kubectl delete all`, `aws s3 rb --force`). | Preview infrastructure changes with `terraform plan -destroy` or `kubectl diff`/dry-run. |
| **Critical system disruption** | `system_disruption` | Terminating system init (PID 1) or supervisor services (`kill -9 1`, `pkill systemd`, `killall systemd`). | Do not target core OS init or supervisor processes. |

### Hard-won carveouts

Blanket regexes on dangerous keywords cause painful false positives in normal software engineering tasks. The gotchas catalog includes specific, battle-tested carveouts that distinguish legitimate development work from catastrophic actions:

- **Declared scratchpad roots (`FAK_GUARD_SCRATCHPAD_ROOTS`)**: A recursive deletion whose targets sit strictly below declared scratchpad roots (e.g. `FAK_GUARD_SCRATCHPAD_ROOTS=/tmp/claude` or `_scratch`) is admitted without operator intervention. Cleaning up temporary session scratch is routine engineering, not an existential threat.
- **Remote SSH sudo (`ssh host 'sudo ...'`)**: While local privilege escalation is blocked, executing `sudo` inside a remote SSH session is permitted because it targets an explicitly designated remote machine rather than escalating privileges on the local host.
- **Read-only terraform plan (`terraform plan -destroy`)**: While `terraform destroy` is blocked fail-closed, running `terraform plan -destroy` is recognized as a read-only speculative plan and permitted.
- **Single literal file degradation / force-only delete**: A force-only delete (`rm -f <path>` without `-r`) of a single explicit, non-glob literal path is not treated as a blanket gotcha block; instead, it falls through to the reversibility preview gate (`REQUIRE_WITNESS`).

### DOS verification and bytes-not-authored protection

The adjudicator uses DOS (Decentralized Operations Substrate) evidence discipline to govern file deletions and mutations based on provenance:

- **Self-authored untracked files (`trace-authored-git-untracked`)**: Untracked files that were authored during the current session carry cryptographic write receipts tracked by the kernel (`adj.ObserveResult`). When the agent deletes an untracked file it created during the current trace, the deletion is admitted with `witness: trace-authored-git-untracked` (`by: monitor/reversibility`). An agent is always allowed to clean up its own newly created files.
- **Tracked files and bytes not authored by the agent**: If a file is tracked in Git, or is untracked but was not authored by the agent in this session, deleting it is an irreversible destruction of user state. The adjudicator refuses immediate execution and holds the call behind a preview confirmation token (`REQUIRE_WITNESS`, `by: monitor/reversibility`). The agent or operator must review the preview payload and supply the returned `_fak_confirm` token to proceed.

### Strict default-deny for compliance and enterprise teams

For security-sensitive deployments, compliance audits, or enterprise environments where every permitted tool must be strictly enumerated in an allowlist, you can enforce `fail_closed`:

1. **Via CLI flag**:
   ```bash
   fak guard --posture fail_closed -- claude
   ```
2. **Via environment variable**:
   ```bash
   export FAK_GUARD_POSTURE=fail_closed
   fak guard -- claude
   ```
3. **Via policy manifest**:
   Set `"posture": "fail_closed"` in your policy JSON file, and pass `--policy <file>`:
   ```bash
   fak guard --policy my-policy.json -- claude
   ```
4. **Dump the strict policy manifest**:
   To inspect or export the strict fail-closed capability floor directly:
   ```bash
   fak guard --dump-strict-policy > strict-policy.json
   ```

Under `fail_closed`, any tool not explicitly listed in `allow` or matched by `allow_prefix` is refused with `DEFAULT_DENY`. An empty manifest (`{}`) is valid and represents the maximally paranoid floor where all tools are refused. `fak policy --check` calls this out explicitly so you never deploy an empty floor by accident.

## The closed refusal vocabulary

Every `deny` reason must be one of these names (a refusal cites a code, never
free text, so a deny is verifiable and a deny-loopback can derive a disposition
from it). Run `fak policy --check` to have an unknown reason rejected with the
full list:

```
DEFAULT_DENY  POLICY_BLOCK  SELF_MODIFY  LEASE_HELD  TRUST_VIOLATION  MALFORMED
MISROUTE  RATE_LIMITED  SECRET_EXFIL  UNWITNESSED  OVERSIZE  UNKNOWN_TOOL
RESULT_SECRET_DISCOVERED  SECRET_REDACTED  SHELL_DIALECT  PII_REDACTED  PII_EXFIL
```

(See `internal/abi/reasons.go` — the same set DOS's `dos_refuse_reasons`
exposes. It is additive: a later minor may add a code; an older binary renders an
unknown code as `REASON_<n>` rather than failing.)

## What the floor does and does NOT bound (honest scope)

- It bounds **which tools** run — deny-by-default on the tool *name*. An
  irreversible tool you do not allow-list is refused *regardless of what is in
  context*, including an injection that talks the model into calling it. This is
  the structural guarantee.
- It does **not** bound the **arguments** of an allow-listed tool. An
  allow-listed `send_email` with attacker-chosen recipients still leans on the
  detection layer (the context-MMU + `normgate`), not on this floor. Keep
  irreversible/exfil-shaped tools *off* the allow-list and let `DEFAULT_DENY`
  hold them.
- `redact_fields` and `self_modify_globs` are best-effort call-boundary hygiene,
  not a guarantee — they inspect decoded args by key/substring (and, for the shell
  write path, the `Bash` `command` string by substring). The shell guard is a
  conservative substring floor, not a full shell parser: it errs toward refusing a
  guarded path named alongside a write verb (a false refusal into a kernel tree is
  cheap; a false *allow* is the self-grading-homework failure the floor exists to stop).
- It adjudicates a **whole turn**, not a live token stream. The floor's verdict is
  computed over the *complete* tool-call set the upstream proposed — a call cannot
  be allowed/denied/repaired until its arguments have fully arrived, and a turn
  where every call is refused rewrites the in-band content. So `fak serve` does
  **not** pass through live decode: a `stream:true` request is adjudicated in full,
  then re-serialized as a well-formed SSE sequence (the wire is identical to a real
  stream; partial tokens are never emitted). This is a property of the enforcement
  model, not a missing feature — adopters wiring an interactive harness to the
  gateway should expect full-turn latency, not token-by-token streaming. See the
  "SSE is buffered rather than token-streaming" note in `GETTING-STARTED.md`.

## Safety properties of the loader

- **Fail-loud on config errors.** A malformed manifest, an unknown reason,
  unknown posture, or an unknown JSON field (e.g. `"allows"` for `"allow"`) is a
  **fatal startup error** — `fak` does not silently fall back to a more
  permissive default.
- **Replace, not merge.** A loaded manifest *is* the whole floor. `--dump` gives
  you the complete default to edit from, so you never lose a baked-in protection
  by omission.
- **Round-trip stable.** `fak policy --dump | fak policy --check` is exact: the
  manifest the binary emits parses back to the identical floor (enforced by
  `TestRoundTrip`).

Runnable demonstration of all three properties plus the empty-manifest warning (above,
"An empty manifest (`{}`) is valid"):
[`examples/policy-loader-properties/`](examples/policy-loader-properties/README.md).
Enforced by `internal/policy`'s `TestRoundTrip`, `TestLoadedPolicyIsLoadBearing`, and
`TestUnknownDenyReasonRejected`.

## Roadmap

- A YAML reader (comments + anchors) as a thin front-end over the same schema —
  kept out of v0.1 to preserve the zero-dependency, single-static-binary
  property.
- Richer argument-level constraints. A regex form (`arg_rules`, above) already
  ships, so the floor can bound *what* a permitted tool does, not only *that* it
  may run; the roadmap is structured value predicates (path-resolution,
  numeric/range, allow-list-by-arg) beyond a single `deny_regex`.
- SIGHUP and signed manifests for long-lived deployments. HTTP reload is already
  available through `POST /v1/fak/policy/reload` when `serve` starts with
  `--policy FILE`.

## Related authorities

- [`docs/positive-workspace-management.md`](docs/positive-workspace-management.md) — construction over punitive default-deny and the FROZEN safety floor doctrine
- [`GETTING-STARTED.md`](GETTING-STARTED.md) — install, offline verification, and guard setup
- [`docs/cli-reference.md`](docs/cli-reference.md) — comprehensive CLI flag reference for `fak guard`, `fak policy`, and `fak serve`

