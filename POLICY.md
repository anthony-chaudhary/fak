# POLICY.md — the deployable capability floor

> **`fak`'s thesis is "permissions as the floor."** This file is how you *deploy*
> that floor: the set of tools your agent may call is a **declarative manifest you
> edit and a reviewer can diff** — not a Go literal you fork the kernel to change.

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
  "rate_limit":   { "max_calls": 50, "max_cost": 0, "key": "trace", "retry_after_ms": 1000 }
}
```

| Field | Meaning |
|---|---|
| `version` | Schema tag. Omit it (current is assumed) or set `fak-policy/v1`. A different **major** (e.g. `fak-policy/v2`) is refused; a newer v1 **minor** — written `fak-policy/v1.x`, e.g. `fak-policy/v1.3`, and matched by the `fak-policy/v1` prefix — is forward-accepted, so any binary that speaks v1 tolerates any v1-minor manifest (there is no per-minor support matrix). |
| `posture` | Default-deny posture. Omit it or set `fail_closed` for the normal floor. Set `admit_and_log` only for unattended/batch runs that should admit low-risk read-shaped `DEFAULT_DENY` calls while logging `would_deny=DEFAULT_DENY`. |
| `allow` | Tool names affirmatively permitted (exact match). |
| `allow_prefix` | A call is permitted if its tool name **starts with** any of these — the read-only family (`read_`, `get_`, `search_`, …). |
| `deny` | Explicit provable refusals: `tool → reason`. The reason **must** be a name from the closed refusal vocabulary (below). |
| `self_modify_globs` | Path fragments that prove a `SELF_MODIFY` attempt (the agent editing its own kernel/config). Checked on **both** write paths: a write-shaped call's target *argument* (`Edit`/`Write`), **and** a shell write whose target lives *inside the command string* (`Bash`: `sed -i`, a `>`/`>>` redirect, `tee`, `git apply`/`git checkout`, an in-place `perl -i`/`ruby -i`/`awk -i`, `python -c`/`node -e` inline writes, `find … -delete`, archive extraction). A shell *read* of a guarded file (`cat`/`grep`) is not a self-modify. |
| `redact_fields` | Arg keys whose value is stripped (`[REDACTED]`, a `TRANSFORM`) before dispatch — secret hygiene at the call boundary. |
| `arg_rules` | Per-tool **argument-value** denials: a list of `{ "tool", "arg", "deny_regex", "reason" }`. If an allow-listed `tool`'s decoded string `arg` matches `deny_regex` (RE2 — no backreferences), the call is refused with `reason` (a closed-vocabulary code). Regex-only and best-effort — it inspects one decoded string, not the resolved effect — but enough to deny `rm -rf`, `git push`, or a write whose path escapes the repo (`-o ../…`). See [`examples/dogfood-claude-policy.json`](examples/dogfood-claude-policy.json) and [`examples/repo-guard-policy.json`](examples/repo-guard-policy.json); the path-resolving structural complement is [`tools/repo_guard.py`](tools/repo_guard.py) (see [`docs/repo-guard.md`](docs/repo-guard.md)). |
| `rate_limit` | Declarative throughput/cost cap (issue #699). An object `{ "max_calls", "max_cost", "key", "retry_after_ms" }` applied to the governor at boot and on `--policy` hot-reload. `max_calls` is a per-key admitted-call quota, `max_cost` a cumulative-cost budget (arg bytes ≈ tokens); set either or both (at least one is required). `key` is the bucketing dimension `trace` (default) / `tool` / `global`. An over-cap call is refused with `RATE_LIMITED`, whose disposition is `WAIT` carrying an advisory `retry_after` — back off like HTTP 429, not a reservation (this is a fixed-ceiling quota with no time window, so the hint is advisory; `retry_after_ms` overrides the default). Omit the block entirely to leave the limiter inert. The `FAK_RATELIMIT_*` env vars are the fallback when no `--policy` is given; a policy load is authoritative over them. |

**Anything not in `allow` / `allow_prefix` and not explicitly denied resolves to
the fail-closed `DEFAULT_DENY`.** An *empty* manifest (`{}`) is valid — it is the
maximally paranoid floor where every call is denied. `fak policy --check` calls
this out explicitly so you never deploy an empty floor by accident.

The one opt-in exception is `"posture": "admit_and_log"`: after explicit deny,
self-modify, redaction, and arg-rule checks have passed, a read-shaped default
deny (`read_`, `get_`, `search_`, `list_`, `lookup_`, `find_`, `calc`, or `calculate`) is
admitted with verdict metadata `posture=admit_and_log` and
`would_deny=DEFAULT_DENY`. Write-shaped calls and explicit denials still fail
closed.

## The closed refusal vocabulary

Every `deny` reason must be one of these names (a refusal cites a code, never
free text, so a deny is verifiable and a deny-loopback can derive a disposition
from it). Run `fak policy --check` to have an unknown reason rejected with the
full list:

```
DEFAULT_DENY  POLICY_BLOCK  SELF_MODIFY  LEASE_HELD  TRUST_VIOLATION  MALFORMED
MISROUTE  RATE_LIMITED  SECRET_EXFIL  UNWITNESSED  OVERSIZE  UNKNOWN_TOOL
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
