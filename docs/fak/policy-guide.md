---
title: "fak Policy Authoring Guide: Capability Floor Manifests"
description: "Author a fak capability floor: write, check, and preflight a default-deny policy manifest that bounds which tools an AI agent may call."
---

# Policy authoring guide (with worked examples)

This is the **task-oriented** companion to [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md). POLICY.md
is the schema reference — the fields and their meaning. This page shows you how to *build a
real capability floor for a real agent*, with **example manifests you can copy** and the
**actual `fak` output each one produces** (every output block below was captured from a
clean build).

> **The one idea.** The capability floor answers exactly one question: **which tools may
> this agent call?** It is a JSON file you edit and a reviewer can diff — not a Go literal
> you recompile. Anything not affirmatively allowed is refused (`DEFAULT_DENY`). You make an
> agent safe mostly by *what you leave off the allow-list*, not by what you deny.

---

## The loop: dump → edit → check → preflight → load

```sh
fak policy --dump > policy.json          # 1. start from the built-in default
# 2. edit policy.json (below)
fak policy --check policy.json           # 3. validate (catches typos before prod)
fak preflight --policy policy.json --tool delete_account --args '{}'   # 4. spot-check a call
fak serve --policy policy.json --addr 127.0.0.1:8080                   # 5. deploy
```

Steps 3 and 4 are the whole safety story for authoring: **`--check`** proves the file is
well-formed, and **`preflight`** answers *"does my floor let X through?"* for any single
call — the cheapest possible test, no model, no server.

---

## Worked example 1 — a coding agent

**Goal:** an agent that can read the repo and run tests, but **cannot** push, deploy, or
delete, and never leaks a secret in a tool argument.

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["run_tests", "git_status", "git_diff"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny": {
    "git_push": "POLICY_BLOCK",
    "deploy": "POLICY_BLOCK",
    "delete_file": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", "policy.json", ".github/workflows/"],
  "redact_fields": ["password", "secret", "api_key", "token", "ssh_key"]
}
```

Validate it. `fak policy --check` doesn't just say "ok" — it prints **the exact floor it
admits**, so you review what you're about to deploy:

```
OK  coding-agent.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 3 tool(s)
allow (prefix)     : read_, get_, search_, list_
deny (explicit)    : 3 tool(s)
                     delete_file -> POLICY_BLOCK
                     deploy -> POLICY_BLOCK
                     git_push -> POLICY_BLOCK
self-modify globs  : .git/, policy.json, .github/workflows/
redact arg fields  : password, secret, api_key, token, ssh_key
arg rules          : 0 rule(s)
ifc safe sinks     : (none)
ifc authorize      : 0 rule(s)
ifc sources        : 0 tool(s)
```

Now spot-check four representative calls with `preflight`:

```
$ fak preflight --policy coding-agent.json --tool read_file  --args '{"path":"main.go"}'
verdict=ALLOW reason=NONE by=monitor          # matched allow_prefix "read_"

$ fak preflight --policy coding-agent.json --tool run_tests  --args '{}'
verdict=ALLOW reason=NONE by=monitor          # exact allow-list hit

$ fak preflight --policy coding-agent.json --tool git_push   --args '{}'
verdict=DENY  reason=POLICY_BLOCK  by=monitor  # explicit, named refusal

$ fak preflight --policy coding-agent.json --tool send_email --args '{}'
verdict=DENY  reason=DEFAULT_DENY  by=monitor  # never listed -> fail-closed
```

The last line is the important one: `send_email` isn't denied *explicitly* — it's denied
because **it was never allowed**. You didn't have to anticipate it. That's the structural
guarantee: an irreversible tool you forgot to think about is refused by default.

### Secret hygiene at the boundary (`redact_fields` → `TRANSFORM`)

`redact_fields` strips a secret-shaped argument *before the call is dispatched*. A call that
carries one comes back as a **`TRANSFORM`** (admitted, but rewritten) rather than `ALLOW`:

```
$ fak preflight --policy coding-agent.json --tool read_config \
    --args '{"path":"x","api_key":"sk-live-12345"}'
verdict=TRANSFORM reason=NONE by=monitor
```

The `api_key` value is replaced with `[REDACTED]` on the way through, so the secret never
reaches the downstream tool or the model's context. This is best-effort key/substring
hygiene, not a cryptographic guarantee — see the honest-scope section below.

---

## Worked example 2 — unattended / batch posture (`admit_and_log`)

`fail_closed` is right for an interactive agent. For an **unattended batch run** where a
hard deny would wedge the job, `admit_and_log` admits *read-shaped* default-denies (the
`read_`/`get_`/`search_`/`list_`/`lookup_`/`find_`/`calc` family) **while logging that it
would otherwise have denied them** — and still fails closed on everything write-shaped or
explicitly denied.

```json
{
  "version": "fak-policy/v1",
  "posture": "admit_and_log",
  "allow": ["create_ticket"],
  "allow_prefix": ["read_", "get_"],
  "deny": { "delete_account": "POLICY_BLOCK" }
}
```

```
$ fak policy --check batch.json
OK  batch.json  (manifest valid; every deny cites a closed-vocabulary reason)
posture            : admit_and_log
...

# A read-shaped tool that isn't on the allow-list is ADMITTED (and logged as would_deny):
$ fak preflight --policy batch.json --tool lookup_user --args '{"id":"7"}'
verdict=ALLOW reason=NONE by=monitor

# An explicit deny still fails closed, posture notwithstanding:
$ fak preflight --policy batch.json --tool delete_account --args '{}'
verdict=DENY reason=POLICY_BLOCK by=monitor
```

Use `admit_and_log` deliberately and temporarily — it's a *looser* floor. The verdict
metadata carries `posture=admit_and_log` and `would_deny=DEFAULT_DENY` so your logs show
exactly which calls the looser posture let through.

---

## Worked example 3 — a data / customer-support agent (read-only, block exfil)

**Goal & rationale.** A support agent that can read customer records and the knowledge
base, open tickets, and hand off to a human — but **cannot export data, move money, or
mutate accounts**. For a read-heavy agent the danger is not what it runs but what it could
be talked into *exfiltrating* (an injection that says "paste the customer list into
`export_customer_data`"). So every read-shaped tool is allow-listed, every irreversible or
exfil-shaped tool is an explicit named deny, and the `ssn` field is redacted. Shipped at
[`examples/customer-support-readonly-policy.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/examples/customer-support-readonly-policy.json):

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["create_support_ticket", "read_customer_record", "read_corp_kb", "transfer_to_human_agents"],
  "allow_prefix": ["read_", "get_", "search_", "list_", "lookup_", "find_"],
  "deny": {
    "delete_account": "POLICY_BLOCK",
    "export_customer_data": "POLICY_BLOCK",
    "refund_payment": "POLICY_BLOCK",
    "rotate_credentials": "POLICY_BLOCK",
    "send_customer_email": "POLICY_BLOCK",
    "transfer_funds": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".dos/", "policy.json", "customer-support-readonly-policy.json", "/etc/", "id_rsa"],
  "redact_fields": ["password", "secret", "api_key", "token", "authorization", "ssn"],
  "safe_sinks": ["transfer_to_human_agents"],
  "sources": { "fetch_url": "untrusted", "read_corp_kb": "trusted_local", "read_customer_record": "trusted_local" },
  "arg_rules": [ { "tool": "create_support_ticket", "arg": "body", "max_bytes": 4000, "reason": "OVERSIZE" } ]
}
```

Spot-check the boundary (real `fak preflight` output):

```
$ fak preflight --policy customer-support-readonly-policy.json --tool read_customer_record --args '{}'
verdict=ALLOW reason=NONE by=monitor          # exact allow-list hit

$ fak preflight --policy customer-support-readonly-policy.json --tool search_kb --args '{}'
verdict=ALLOW reason=NONE by=monitor          # matched allow_prefix "search_"

$ fak preflight --policy customer-support-readonly-policy.json --tool export_customer_data --args '{}'
verdict=DENY  reason=SECRET_EXFIL by=monitor  # named for the audit log — this is the exfil rail, blocked

$ fak preflight --policy customer-support-readonly-policy.json --tool refund_payment --args '{}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # money movement, blocked

$ fak preflight --policy customer-support-readonly-policy.json --tool drop_table --args '{}'
verdict=DENY  reason=DEFAULT_DENY  by=monitor  # never listed -> fail-closed (you didn't have to think of it)

$ fak preflight --policy customer-support-readonly-policy.json --tool create_support_ticket --args '{"body":"<5000 bytes>"}'
verdict=DENY  reason=OVERSIZE by=monitor       # arg_rule max_bytes caps ticket injection surface

$ fak preflight --policy customer-support-readonly-policy.json --tool read_customer_record --args '{"id":"1","ssn":"123-45-6789"}'
verdict=TRANSFORM reason=NONE by=monitor       # the ssn is redacted to [REDACTED] on the way through
```

**Testing checklist for this floor.**
- [ ] `fak policy --check` passes — every deny cites a closed-vocabulary reason.
- [ ] `preflight` every `deny` entry → `POLICY_BLOCK` (not `DEFAULT_DENY` — you want the *named* refusal in the audit log).
- [ ] `preflight` an unlisted exfil-shaped tool (e.g. `dump_table`) → `DEFAULT_DENY` (the structural backstop).
- [ ] `preflight create_support_ticket` with an oversize body → `OVERSIZE`.
- [ ] `preflight read_customer_record` carrying an `ssn` arg → `TRANSFORM` (secret stripped before dispatch).

---

## Worked example 4 — a DevOps agent (plan & dry-run, block prod mutations)

**Goal & rationale.** A DevOps agent that can inspect clusters, render templates, validate
Terraform, and *plan* changes — but **cannot apply, deploy, exec into pods, or touch
prod**. The distinctive lever here is **argument-level**: `plan_deploy` is allowed (planning
is the job), but its `environment` argument is regex-blocked from `prod|production`, so the
agent can plan a staging deploy and is refused — by value, not by tool name — the moment it
aims the *same* tool at prod. Shipped at
[`examples/devops-dryrun-policy.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/examples/devops-dryrun-policy.json):

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["create_change_request", "diff_infra", "helm_template", "kubectl_get", "plan_deploy", "validate_terraform"],
  "allow_prefix": ["read_", "get_", "list_", "lookup_", "describe_", "dryrun_"],
  "deny": {
    "deploy_production": "POLICY_BLOCK", "drop_database": "POLICY_BLOCK",
    "kubectl_delete": "POLICY_BLOCK", "kubectl_exec": "POLICY_BLOCK",
    "rotate_credentials": "POLICY_BLOCK", "run_command": "POLICY_BLOCK", "shell": "POLICY_BLOCK",
    "terraform_apply": "POLICY_BLOCK", "transfer_funds": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".dos/", ".kube/config", "policy.json", "devops-dryrun-policy.json", "terraform.tfstate", "secrets/", "/etc/", "id_rsa"],
  "redact_fields": ["password", "secret", "api_key", "token", "authorization", "kubeconfig"],
  "safe_sinks": ["create_change_request"],
  "sources": { "fetch_runbook": "trusted_local", "read_repo": "trusted_local", "read_ticket": "untrusted" },
  "arg_rules": [
    { "tool": "create_change_request", "arg": "body", "max_bytes": 8000, "reason": "OVERSIZE" },
    { "tool": "plan_deploy", "arg": "environment", "deny_regex": "(?i)^(prod|production)$", "reason": "POLICY_BLOCK" }
  ]
}
```

Spot-check the boundary (real `fak preflight` output):

```
$ fak preflight --policy devops-dryrun-policy.json --tool kubectl_get --args '{}'
verdict=ALLOW reason=NONE by=monitor          # exact allow-list hit

$ fak preflight --policy devops-dryrun-policy.json --tool describe_pod --args '{}'
verdict=ALLOW reason=NONE by=monitor          # matched allow_prefix "describe_"

$ fak preflight --policy devops-dryrun-policy.json --tool plan_deploy --args '{"environment":"staging"}'
verdict=ALLOW reason=NONE by=monitor          # planning staging is the job

$ fak preflight --policy devops-dryrun-policy.json --tool plan_deploy --args '{"environment":"prod"}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # SAME tool, refus'd by the arg value — prod is off-limits even for planning

$ fak preflight --policy devops-dryrun-policy.json --tool terraform_apply --args '{}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # state mutation, blocked

$ fak preflight --policy devops-dryrun-policy.json --tool create_change_request --args '{"body":"<9000 bytes>"}'
verdict=DENY  reason=OVERSIZE by=monitor       # arg_rule caps the CR body (injection surface)

$ fak preflight --policy devops-dryrun-policy.json --tool shell --args '{}'
verdict=DENY  reason=POLICY_BLOCK by=monitor   # the obvious one — but DEFAULT_DENY would hold it even if you forgot
```

**Testing checklist for this floor.**
- [ ] `fak policy --check` passes.
- [ ] `preflight` every prod-shaped mutation (`terraform_apply`, `deploy_production`, `kubectl_delete`, `kubectl_exec`, `shell`) → `POLICY_BLOCK`.
- [ ] `preflight plan_deploy` with `environment` = `prod`, `production`, `PROD` → all `DENY` (the regex is case-insensitive).
- [ ] `preflight plan_deploy` with `environment` = `staging` → `ALLOW` (the tool itself is fine; only the prod value is refused).
- [ ] `preflight` a `dryrun_` / `describe_` tool you did *not* list exactly → `ALLOW` (prefix family).
- [ ] `preflight create_change_request` with an oversize body → `OVERSIZE`.

---

## Worked example 5 — a research agent (web & search, block execution)

**Goal & rationale.** A research agent that can fetch URLs, read webpages, search, and take
notes — but **cannot run commands, post, email, upload, or delete**. Two argument-level
guards do real work here: `fetch_url` is blocked from `file://`/`ftp://`/`ssh://` schemes
(SSRF and local-file escape — an injection that points the fetcher at `file:///etc/shadow`
is refused by value), and `create_note` is confined to a `notes/**` glob so notes can't
overwrite arbitrary repo paths. It runs under `admit_and_log` so a long batch crawl isn't
wedged by one unlisted read-shaped tool. Shipped at
[`examples/research-agent-policy.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/examples/research-agent-policy.json):

```json
{
  "version": "fak-policy/v1",
  "posture": "admit_and_log",
  "allow": ["cite_source", "create_note", "fetch_url", "read_webpage", "summarize_document"],
  "allow_prefix": ["read_", "get_", "search_", "list_", "lookup_", "find_", "calc"],
  "deny": {
    "delete_file": "POLICY_BLOCK", "drop_table": "POLICY_BLOCK",
    "exfiltrate": "POLICY_BLOCK", "post_message": "POLICY_BLOCK",
    "run_command": "POLICY_BLOCK", "send_email": "POLICY_BLOCK",
    "transfer_funds": "POLICY_BLOCK", "upload_file": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".dos/", "policy.json", "research-agent-policy.json", "/etc/", "id_rsa"],
  "redact_fields": ["password", "secret", "api_key", "token", "authorization"],
  "sources": { "fetch_url": "untrusted", "read_file": "trusted_local", "read_webpage": "untrusted" },
  "arg_rules": [
    { "tool": "create_note", "arg": "path", "allow_glob": "notes/**", "reason": "POLICY_BLOCK" },
    { "tool": "fetch_url",   "arg": "url",  "deny_regex": "(?i)^(file|ftp|ssh)://", "reason": "POLICY_BLOCK" }
  ]
}
```

Spot-check the boundary (real `fak preflight` output):

```
$ fak preflight --policy research-agent-policy.json --tool search_web --args '{}'
verdict=ALLOW reason=NONE by=monitor          # matched allow_prefix "search_"

$ fak preflight --policy research-agent-policy.json --tool fetch_url --args '{"url":"https://example.com"}'
verdict=ALLOW reason=NONE by=monitor          # http(s) fetch is the job

$ fak preflight --policy research-agent-policy.json --tool fetch_url --args '{"url":"file:///etc/shadow"}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # arg_rule deny_regex — local-file/SSRF escape refused by value

$ fak preflight --policy research-agent-policy.json --tool create_note --args '{"path":"notes/april.md","body":"hi"}'
verdict=ALLOW reason=NONE by=monitor          # inside the notes/** glob

$ fak preflight --policy research-agent-policy.json --tool create_note --args '{"path":"etc/passwd","body":"hi"}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # outside the glob — can't write just anywhere

$ fak preflight --policy research-agent-policy.json --tool run_command --args '{}'
verdict=DENY  reason=POLICY_BLOCK by=monitor  # execution, blocked

$ fak preflight --policy research-agent-policy.json --tool exfiltrate --args '{}'
verdict=DENY  reason=SECRET_EXFIL by=monitor  # named exfil rail, blocked

$ fak preflight --policy research-agent-policy.json --tool exec_payload --args '{}'
verdict=DENY  reason=DEFAULT_DENY  by=monitor  # an execution tool you never thought to name -> fail-closed

$ fak preflight --policy research-agent-policy.json --tool read_file --args '{"path":"x","api_key":"sk-live-12345"}'
verdict=TRANSFORM reason=NONE by=monitor       # the api_key is redacted before the call dispatches
```

**Testing checklist for this floor.**
- [ ] `fak policy --check` passes.
- [ ] `preflight fetch_url` with `file://`, `ftp://`, `ssh://` → all `DENY` (regex is case-insensitive — try `FILE://` too).
- [ ] `preflight create_note` inside `notes/**` → `ALLOW`; outside (e.g. `etc/x`, `../x`) → `DENY`.
- [ ] `preflight` an unlisted execution tool (e.g. `exec_payload`, `eval_code`) → `DEFAULT_DENY`.
- [ ] `preflight` every `deny` entry → the named reason (`POLICY_BLOCK` / `SECRET_EXFIL`).
- [ ] `preflight read_file` carrying an `api_key`/`token` arg → `TRANSFORM`.

> **Five complete floors, one pattern.** Examples 1–5 all follow the same shape: a short
> `allow` / `allow_prefix` for what the agent's job actually needs, an explicit `deny` for
> the rails you want named in the audit log, `DEFAULT_DENY` holding everything else, and
> `arg_rules` / `redact_fields` / `self_modify_globs` for the value-level and self-modify
> guards. Copy the closest one and edit.

---

## The check step catches mistakes *before* production

`fak policy --check` is **fail-loud**: a malformed manifest is a fatal error, not a silent
fallback to something more permissive. Two real examples:

**A typo'd field name** (you wrote `allows`, the schema wants `allow`):

```
$ fak policy --check bad.json
fak policy: policy bad.json: invalid manifest: json: unknown field "allows"
$ echo $?
1
```

**An invalid deny reason** — every deny must cite a name from the closed refusal
vocabulary, and `--check` prints the whole valid set when you miss:

```
$ fak policy --check bad2.json
fak policy: policy bad2.json: unknown deny reason(s): foo="NOT_A_REAL_REASON"; valid reasons:
  DEFAULT_DENY, LEASE_HELD, MALFORMED, MISROUTE, OVERSIZE, POLICY_BLOCK, RATE_LIMITED,
  SECRET_EXFIL, SELF_MODIFY, TRUST_VIOLATION, UNKNOWN_TOOL, UNWITNESSED
$ echo $?
1
```

Wire `fak policy --check policy.json` into CI as a required check — a non-zero exit fails the
build, so a broken floor can never ship.

---

## Authoring patterns (a checklist)

| Pattern | Do this |
|---|---|
| **Start from the default** | `fak policy --dump` — never hand-write from scratch, or you'll drop a baked-in protection. The manifest *replaces* the default; it does not merge. |
| **Allow-list the verbs, not the nouns** | Permit `search_kb`, not "the agent." Keep the list short and boring. |
| **Lean on `allow_prefix` for read families** | `read_`/`get_`/`search_`/`list_` cover most safe tools in one line. |
| **Keep irreversible tools OFF the list** | `DEFAULT_DENY` is stronger than an explicit `deny` — you don't have to enumerate every dangerous tool, only allow the safe ones. Use explicit `deny` for *documentation/auditing* of the ones you specifically want a named refusal for. |
| **Redact secret-shaped args** | `redact_fields` for `password`/`token`/`api_key`/… so a secret never crosses the boundary. |
| **Guard self-modification** | `self_modify_globs` for `.git/`, your CI workflow dir, and the policy file itself. |
| **Validate in CI** | `fak policy --check` as a required, fail-the-build check. |
| **Spot-check the scary calls** | `fak preflight` on `delete_*`, `deploy`, `transfer_*` before every deploy. |

---

## Argument-level constraints (`arg_rules`, `redact_fields`, `self_modify_globs`)

The capability floor bounds *which tools run*. Three manifest fields bound **how** an
allow-listed tool may be called, and every example above uses at least one of them.

### `deny_regex` — refuse an allow-listed tool when an argument matches

The tool is on the allow-list; the *value* is off-limits. The two recurring patterns:

```jsonc
// DevOps: plan_deploy is allowed, but never aimed at prod (example 4)
{ "tool": "plan_deploy", "arg": "environment", "deny_regex": "(?i)^(prod|production)$", "reason": "POLICY_BLOCK" }

// Research: fetch_url is allowed, but never at a local-file scheme (example 5)
{ "tool": "fetch_url", "arg": "url", "deny_regex": "(?i)^(file|ftp|ssh)://", "reason": "POLICY_BLOCK" }
```

Regexes are Go `regexp` (RE2) — case-insensitive via `(?i)`. Common shapes worth copying
(from [`examples/repo-guard-policy.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/examples/repo-guard-policy.json), which
guards a shell tool):

```jsonc
{ "tool": "Bash", "arg": "command", "deny_regex": "\\brm\\s+-[A-Za-z]*[rRfF]", "reason": "POLICY_BLOCK" },   // recursive/forced delete
{ "tool": "Bash", "arg": "command", "deny_regex": "\\bsudo\\b",              "reason": "POLICY_BLOCK" },   // privilege escalation
{ "tool": "Bash", "arg": "command", "deny_regex": "\\bmkfs\\b|\\bdd\\s+if=", "reason": "POLICY_BLOCK" },   // disk wipe
{ "tool": "Bash", "arg": "command", "deny_regex": ":\\(\\)\\s*\\{",          "reason": "POLICY_BLOCK" },   // fork-bomb
{ "tool": "Bash", "arg": "command", "deny_regex": "\\b(curl|wget)\\b[^|]*\\|\\s*(sudo\\s+)?(ba)?sh\\b", "reason": "POLICY_BLOCK" } // pipe-to-shell
```

### `allow_glob` — confine an allow-listed tool to a path prefix

A self-modify-shaped tool may be allow-listed for one directory and refused everywhere
else. Example 5 confines note-writing to `notes/**`:

```jsonc
{ "tool": "create_note", "arg": "path", "allow_glob": "notes/**", "reason": "POLICY_BLOCK" }
```

### `max_bytes` — cap an argument's size

Bounds the injection surface on a free-text argument (ticket body, change-request body):

```jsonc
{ "tool": "create_support_ticket", "arg": "body", "max_bytes": 4000, "reason": "OVERSIZE" }
```

### `redact_fields` and `self_modify_globs`

- **`redact_fields`** strips a secret-shaped argument *before* dispatch, turning the verdict
  into a `TRANSFORM` rather than an `ALLOW` (see the `api_key`/`ssn` runs above). It is
  best-effort key/substring hygiene on decoded args, not a cryptographic guarantee.
- **`self_modify_globs`** protects the agent's own trust boundary — `.git/`, the policy
  file, CI workflow dirs, credential files, and the frozen-ABI directories an agent must
  never edit. A call that would write inside one of these is refused.

> **The closed refusal vocabulary.** Every `deny` reason and every `arg_rules[].reason`
> must come from this set (printed in full when `--check` rejects a bad one):
> `DEFAULT_DENY, LEASE_HELD, MALFORMED, MISROUTE, OVERSIZE, POLICY_BLOCK, RATE_LIMITED,
> SECRET_EXFIL, SELF_MODIFY, TRUST_VIOLATION, UNKNOWN_TOOL, UNWITNESSED`. Pick the most
> specific one — `SECRET_EXFIL` on `export_customer_data` says more in the audit log than
> `POLICY_BLOCK` would.

---

## Throughput & cost caps (`rate_limit` → a WAIT deny with retry-after)

The capability floor bounds *which* tools run; `rate_limit` bounds *how fast* (and how
expensively) an allow-listed tool may be called. It is the one top-level manifest field
that is **runtime config, not an adjudicator rule** — it tunes fak's rank-8 load-shed
governor (`internal/ratelimit`) rather than adding a deny. The manifest block is
authoritative over the `FAK_RATELIMIT_*` env fallback and is re-applied on every
`--policy` hot-reload, so the cap is versioned with the rest of the floor instead of
living only in the environment.

```jsonc
{
  "version": "fak-policy/v1",
  "allow": ["search_kb", "summarize"],
  "rate_limit": {
    "max_calls": 200,        // per-key admitted-call quota   (0 = no call cap)
    "max_cost": 500000,      // per-key cumulative cost budget (0 = no cost cap)
    "key": "trace",          // the bucket the quota counts in: trace | tool | global
    "retry_after_ms": 1000   // advisory back-off on the over-cap WAIT (0 = limiter default, 1s)
  }
}
```

- **At least one cap must be declared** (`max_calls` *or* `max_cost`); an empty block is
  inert. When both are set, a call must satisfy both. **Cost** is the call's argument byte
  length — a resolver-free stand-in for tokens — unless the caller supplies a real count in
  `Meta["fak.ratelimit.cost"]`.
- **`key`** selects the bucket the quota is counted in: `trace` (per agent run — the
  default), `tool` (per tool name), or `global` (one shared bucket across the process).

**The WAIT / retry-after deny contract.** When a key exceeds a cap, the governor returns a
`DENY` citing the closed reason `RATE_LIMITED`, and the kernel maps `RATE_LIMITED` to a
**WAIT disposition** — a *recoverable* deny. The agent loop backs off and retries rather
than treating it as a hard refusal, the way `errno` pairs `EAGAIN` with a retry window. The
refusal is a **value, not an exception**: its `meta` carries `retry_after` (a duration, e.g.
`1s`) and `retry_after_ms` (the same back-off in milliseconds), alongside `cap` (which cap
fired — `max_calls` / `max_cost`), `limit`, and `key`. The back-off is the operator-declared
`retry_after_ms` when set, else a built-in `1s` default. Disclosure is bounded — the deny
names the cap and the back-off, never the call's arguments. With no `rate_limit` block (and
no `FAK_RATELIMIT_*` env), the governor Defers on every call, so the limiter is inert until
a cap is set.

---

## Testing policies

`preflight` is a single-call test with no model and no server — cheap enough to script.
Run the **same five-call battery** against every manifest before it ships:

```sh
# A 30-line CI gate for one policy. Fails the build on any unexpected verdict.
POLICY=examples/your-policy.json
expect() { # expect <tool> <args-json> <want-verdict>
  got=$(fak preflight --policy "$POLICY" --tool "$1" --args "$2" | grep -o 'verdict=[A-Z]*' | cut -d= -f2)
  [ "$got" = "$3" ] || { echo "FAIL $1: want $3 got $got"; exit 1; }
}
expect read_x           '{}'                 ALLOW
expect delete_x         '{}'                 DENY      # POLICY_BLOCK or DEFAULT_DENY — either is a pass
expect unlisted_tool    '{}'                 DENY      # the DEFAULT_DENY backstop
```

**The universal testing checklist (run for every floor).**
- [ ] `fak policy --check <manifest>` exits 0 (CI gate — a broken floor never ships).
- [ ] `preflight` one representative tool per `allow` / `allow_prefix` entry → `ALLOW`.
- [ ] `preflight` every `deny` entry → its **named** reason (not `DEFAULT_DENY`).
- [ ] `preflight` a tool you deliberately did *not* list → `DEFAULT_DENY` (the backstop).
- [ ] For each `arg_rules[]` entry: `preflight` a value that trips it (→ the rule's reason)
      *and* a value that passes (→ `ALLOW`) — both directions, every rule.
- [ ] `preflight` an allow-listed call carrying a `redact_fields` key → `TRANSFORM`.
- [ ] `preflight` a call whose target path lands inside a `self_modify_globs` entry → refused.

**Common pitfalls.**
- **Hand-writing from scratch.** Always start from `fak policy --dump` — the manifest
  *replaces* the default, it does not merge, so a from-scratch file silently drops every
  baked-in protection.
- **Denying instead of omitting.** A giant `deny` list is a smell: `DEFAULT_DENY` already
  holds anything you didn't allow. Use explicit `deny` only for the rails you want *named*
  in the audit log; keep dangerous tools off the `allow`/`allow_prefix` lists instead.
- **Trusting the detector as the floor.** `deny_regex` and `redact_fields` are best-effort
  value hygiene layered *on top* of the lock — they do not replace "keep the tool off the
  allow-list." The structural guarantee is still `DEFAULT_DENY`.
- **Forgetting the prefix family.** `allow_prefix: ["read_"]` admits `read_file`,
  `read_webpage`, *and* a future `read_secrets`. Re-run `preflight` on every newly-added
  tool before redeploy.
- **One policy for all environments.** Dev and prod deserve different floors (example 4
  blocks the prod value by argument, not by tool) — see production policies below.

---

## Production policies

### Multi-team: one manifest per role, not one giant manifest

A floor is most honest when it is short and boring. Prefer **one small manifest per agent
role** (coding, data, devops, research — the five above) over one sweeping manifest a
reviewer can't eyeball. Each team owns its own file; the diff between revisions *is* the
security review.

### Environment-specific: gate the dangerous *value*, not just the tool

The same agent often runs in dev and prod. Rather than two tool lists, allow the tool and
refuse the dangerous argument value. Example 4's `plan_deploy` works in staging and is
refused at `environment=prod` — one tool, one regex, environment-scoped by value. Generalize
it for any tool whose blast radius depends on an argument (`cluster`, `region`,
`account_id`, …).

### Rotation and reload

A long-lived gateway reloads the same file without dropping the process, the warm vDSO
cache, or the IFC ledger — see [server-config.md](server-config.md) for the
`/v1/fak/policy/reload` endpoint and bearer/auth requirements:

```sh
fak serve --policy policy.json --addr 127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/v1/fak/policy/reload   # bearer token if --require-key-env is set
```

Because `--check` is fail-loud, the safe rotation is **always** `fak policy --check
new.json` *before* the reload — a malformed manifest is a fatal error, never a silent
fallback to something more permissive.

---

## Honest scope — what the floor does and does *not* bound

Carried verbatim from [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) because it matters:

- ✅ **Bounds which tools run.** An irreversible tool you don't allow-list is refused
  *regardless of context* — including an injection that talks the model into calling it.
  This is the structural guarantee.
- ⚠️ **Does NOT bound the resolved effect** of an allow-listed tool's arguments.
  `arg_rules` (above) gives you regex / glob / size predicates on one decoded string —
  enough to deny `rm -rf`, block `environment=prod`, or cap a body — but it is *best-effort*
  and inspects the arg, not the resolved effect. An allow-listed `send_email` with
  attacker-chosen recipients still leans on the *detection* layer (context-MMU +
  `normgate`). Keep exfil-shaped tools off the allow-list and let `DEFAULT_DENY` hold them.
  (Richer *structured* value predicates — path-resolution, numeric/range — are on the
  [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) roadmap.)
- ⚠️ `redact_fields` and `self_modify_globs` are **best-effort** key/substring hygiene, not a
  cryptographic guarantee — they inspect decoded args.

The floor is the *lock* (the lever was never wired up). The detector is a *helpful bonus*
layered on top — never the floor. See [Policy in the kernel](../explainers/policy-in-the-kernel.md)
for why putting the check on the same call path (default-deny, fail-closed) is the whole game.

---

## See also

- [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the field-by-field schema reference (`arg_rules`, IFC `sources`/`safe_sinks`, the closed reason vocabulary) and the roadmap.
- [`examples/`](https://github.com/anthony-chaudhary/fak/tree/main/examples) — the five worked manifests above plus `repo-guard-policy.json`, `dogfood-claude-policy.json`, and `policy.example.json`, all copy-ready.
- [server-config.md](server-config.md) — the `/v1/fak/policy/reload` endpoint and gateway auth/network config.
- [tutorial.md §1.5](tutorial.md) — authoring a floor in the guided first session.
- [security.md](security.md) — hardening the deployed gateway (auth, network, defense-in-depth).
- [Policy in the kernel](../explainers/policy-in-the-kernel.md) — the design rationale.
