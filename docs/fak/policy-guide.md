# Policy authoring guide (with worked examples)

This is the **task-oriented** companion to [`fak/POLICY.md`](../../POLICY.md). POLICY.md
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

## Honest scope — what the floor does and does *not* bound

Carried verbatim from [`POLICY.md`](../../POLICY.md) because it matters:

- ✅ **Bounds which tools run.** An irreversible tool you don't allow-list is refused
  *regardless of context* — including an injection that talks the model into calling it.
  This is the structural guarantee.
- ⚠️ **Does NOT bound the arguments** of an allow-listed tool. An allow-listed `send_email`
  with attacker-chosen recipients leans on the *detection* layer (context-MMU + `normgate`),
  not on this floor. Keep exfil-shaped tools off the allow-list and let `DEFAULT_DENY` hold
  them. (Argument-level value predicates are on the roadmap.)
- ⚠️ `redact_fields` and `self_modify_globs` are **best-effort** key/substring hygiene, not a
  cryptographic guarantee — they inspect decoded args.

The floor is the *lock* (the lever was never wired up). The detector is a *helpful bonus*
layered on top — never the floor. See [Policy in the kernel](../explainers/policy-in-the-kernel.md)
for why putting the check on the same call path (default-deny, fail-closed) is the whole game.

---

## Reloading a live floor

A long-lived gateway reloads the same file without dropping the process, the warm vDSO
cache, or the IFC ledger:

```sh
fak serve --policy policy.json --addr 127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/v1/fak/policy/reload   # bearer token if --require-key-env is set
```

---

## See also

- [`fak/POLICY.md`](../../POLICY.md) — the field-by-field schema reference and the closed reason vocabulary.
- [tutorial.md §1.5](tutorial.md) — authoring a floor in the guided first session.
- [security.md](security.md) — hardening the deployed gateway (auth, network, defense-in-depth).
- [Policy in the kernel](../explainers/policy-in-the-kernel.md) — the design rationale.
