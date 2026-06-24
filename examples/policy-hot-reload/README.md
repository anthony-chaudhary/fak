# Hot-reload the capability floor with no restart (`POST /v1/fak/policy/reload`)

A long-lived `fak serve` gateway can adopt a **new** capability floor without
restarting. You edit the served manifest, validate it, `POST` the reload, and the
running process swaps the floor under itself — the verdict for the same call
changes, while the process, its warm vDSO cache, and its IFC taint ledger all
survive. This example is the runnable proof of that operator loop.

It is the deployment-surface companion to [`../../POLICY.md`](../../POLICY.md)
("The workflow" → the `fak serve … / curl … /policy/reload` block), which
describes the motion; this directory *runs* it and asserts each step.

## Run it

```bash
examples/policy-hot-reload/run.sh         # build fak, serve, run 10 witnesses, teardown
```

Needs only Go (to build `fak`) and `curl` — **no model, key, or GPU**. It **runs in a
few seconds** once `fak` is built — no model, no network. The verdict
surface (`/v1/fak/adjudicate`) and the lifecycle routes do not touch an upstream,
so the result is **deterministic** on every run. A captured run is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What you see

The run prints the ten witnesses above in order, each tagged with its verdict. The tells
that the hot-reload worked: the **same** `delete_account` call flips from `DENY`
(`POLICY_BLOCK`) under floor **A** to `ALLOW` under floor **B**; the trace stays
`quarantined` across the swap; and `start_time_unix` + PID are identical before and after
(only `uptime_seconds` rises) — proof the process never restarted. The script ends with a
one-line `PASS`/`FAIL` summary and gates the exit code on it.

## The operator loop the script walks

| # | Step | Witness |
|---|---|---|
| 1 | Adjudicate `delete_account` under floor **A** | `DENY` (`POLICY_BLOCK`) |
| 2 | Quarantine a poisoned result onto a session trace | trace → `quarantined` (warm state raised) |
| 3 | Record the gateway's `start_time_unix` + PID | the no-restart baseline |
| 4 | Edit the served file: **A → B** (also allow `delete_account`) | — |
| 5 | `fak policy --check` the new file **before** reloading | `valid` (fail-loud gate) |
| 6 | `POST /v1/fak/policy/reload` | `reloaded:true` |
| 7 | Adjudicate the **same** call under floor **B** | `ALLOW` — the floor swapped |
| 8 | Re-observe the trace | **still** `quarantined` — IFC ledger survived |
| 9 | Re-read `start_time_unix` + PID | **unchanged**, uptime higher — no restart |
| 10 | `POST` reload of a **broken** manifest | `400`, and floor **B** still holds |

## Why this proves "no restart, no dropped state"

The decisive witnesses are not cosmetic — a restart would change them:

- **`start_time_unix` is identical before and after**, and `uptime_seconds` only
  rises. A restarted process gets a fresh boot epoch; an in-place reload cannot.
  (`fak` reads these from [`/debug/vars`](../../internal/gateway/debug.go); the PID
  the launcher tracks is likewise unchanged.)
- **The IFC taint ledger survives.** A session that was `quarantined` before the
  reload is **still** `quarantined` after it. A restart would reset every trace to
  the clean `trusted` default — so the surviving high-water mark is direct
  evidence the in-process ledger was never dropped. (Same for the warm vDSO cache:
  it is process-global state that a reload leaves untouched and a restart would
  zero.)
- The reload itself only calls `adjudicator.Default.SetPolicy(...)` on the running
  server — it never rebinds the listener or re-boots the kernel.

This is the property the three green tests in
[`../../internal/gateway/policy_reload_test.go`](../../internal/gateway/policy_reload_test.go)
cover at the route layer:
`TestPolicyReloadRouteInvokesConfiguredReloader` (reload calls the wired loader),
`TestPolicyReloadRouteDisabledWithoutCallback` (404 when `serve` started with no
`--policy`), and `TestPolicyReloadRouteReportsLoaderFailure` (a bad manifest is a
`400`, not a silent swap). This example is the end-to-end walkthrough those unit
tests stand in for.

## Fail-loud, before *and* during the swap

The floor is validated twice, and a bad floor never lands:

- **Before reload (step 5).** `fak policy --check <file>` is the pre-flight gate
  POLICY.md prescribes: every `deny` must cite a closed-vocabulary reason, no
  unknown keys, a known schema version. Run it in your edit→deploy pipeline so a
  typo is caught at the desk, not at the gateway.
- **During reload (step 10).** Even if you skip the check, the reload route is
  fail-loud: a malformed manifest (here the typo `"allows"` for `"allow"`) is
  rejected with `400` and the **last-good floor stays in force** — the gateway
  never silently falls back to a more permissive default. The same call that was
  `ALLOW` under the good floor B is still `ALLOW`; the bad file changed nothing.

## The auth gate

This example serves on loopback with **no** authentication so it is a one-command
demo. In a real deployment you pass `--require-key-env VAR`, and then **the reload
route requires the same bearer token as every other `/v1/fak/*` route** — there is
no separate admin key, so locking the gateway locks its control plane too:

```bash
fak serve --policy policy.json --addr 0.0.0.0:8080 --require-key-env FAK_TOKEN
curl -X POST http://host:8080/v1/fak/policy/reload \
  -H "Authorization: Bearer $FAK_TOKEN"          # or:  -H "x-api-key: $FAK_TOKEN"
```

Without the header the reload is `401`, same as a model request would be. The
operator-route auth contract — and both accepted header shapes — is proven
end-to-end in the sibling [`../auth-hardening/`](../auth-hardening/) example
(witnesses 5–6 there gate `/metrics`; the policy reload route rides the identical
gate).

## Scope — what this demo does **not** claim (what hot-reload does **not** change)

This demo proves the *operator loop* (edit → check → reload → swapped verdict, no restart, no
dropped state); it does **not** claim that hot-reload is a security boundary on its own, and
it does **not** demonstrate multi-node propagation or auth (auth is the sibling
`auth-hardening/` example). Concretely, hot-reload itself:


- **It replaces the whole floor, it does not merge.** The reloaded manifest *is*
  the new floor. Start from `fak policy --dump` so you never drop a baked-in
  protection by omission. (See "Replace, not merge" in
  [`../../POLICY.md`](../../POLICY.md).)
- **It does not re-key in-flight sessions.** The vDSO cache and IFC ledger survive
  the swap by design (that is the point). The new floor governs calls adjudicated
  *after* the reload; it does not retroactively re-judge a result already admitted.
- **It is the floor, not auth.** Reloading the capability floor changes *what* any
  caller may do; it never changes *who* may call. Rotate the auth secret by
  changing `$FAK_TOKEN` and restarting (see the auth-hardening notes).

## Where this fits

- **The workflow, in context:** [`../../POLICY.md`](../../POLICY.md) — "The workflow: dump → edit → check → load"
- **The route-level tests:** [`../../internal/gateway/policy_reload_test.go`](../../internal/gateway/policy_reload_test.go)
- **The auth gate the reload route inherits:** [`../auth-hardening/`](../auth-hardening/)
- **The roadmap item this closes the example gap for:** [`../../POLICY.md`](../../POLICY.md) "Roadmap" (HTTP reload via `POST /v1/fak/policy/reload`)
