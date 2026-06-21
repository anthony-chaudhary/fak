# Security best practices for deploying `fak`

This is the operator's hardening guide: how to deploy `fak serve` so the gate it provides
is actually load-bearing. It covers the **threat model**, the **two independent gates**,
**auth**, **network exposure**, and an **honest statement of what `fak` does and does not
protect against**. Every output block was captured from a clean build of `fak` v0.30.0.

> **Read the honest scope first.** `fak` is built to survive a skeptic reading the code.
> Its security value is **structural** (a refused tool was never wired up) plus
> **containment** (a poisoned result never reaches the model) — *not* a smarter classifier.
> The detector layer is **≈100% evadable by design** and is a helpful bonus, never the
> floor. If you deploy `fak` expecting a content filter that catches clever attacks, you've
> deployed the wrong layer. Deploy it for the lock and the wall.

---

## 1. The threat model: two gates, not one classifier

An attacker has to beat **two independent gates**, and neither is a detector you can talk
past:

| Gate | What it is | Why it holds |
|---|---|---|
| **The lock** (capability floor) | a default-deny allow-list of tools/args | an irreversible tool you didn't allow-list is refused *regardless of context* — the lever was never built |
| **The wall** (result quarantine) | poisoned tool results held out of the model's context | the bytes never reach attention, so an injection inside them can't influence the next turn |

The evadable part — the *detector* that flags suspicious results — sits **on top** of the
wall. If it misses, the result is still quarantined by policy; if it fires, that's a bonus.
The security floor does not depend on it.

→ Background: [Policy in the kernel](../explainers/policy-in-the-kernel.md) ·
[`README.md` "Security: the lock, not the screener"](../../README.md).

---

## 2. Deploy a reviewed capability floor (never the bare default in prod)

The single most important deployment step: run with an explicit `--policy` that you have
reviewed, not the built-in default. Build it with the [policy guide](policy-guide.md), and
make these checks part of the deploy:

### Keep irreversible tools OFF the allow-list

`DEFAULT_DENY` is stronger than an explicit deny — a tool you never name is refused, even
one you didn't anticipate. Verify with `preflight` before shipping:

```
$ fak preflight --policy floor.json --tool wire_transfer --args '{}'
verdict=DENY reason=DEFAULT_DENY by=monitor      # never listed -> fail-closed
```

### Gate dangerous *arguments* of coarse tools (`arg_rules`)

A coarse tool like `Bash` is one capability, but its `command` value is gated by RE2
deny-patterns, so the floor separates `ls` from `rm -rf` without forking the kernel. The
shipped Claude Code dogfood floor demonstrates it — real output:

```
$ fak preflight --policy examples/dogfood-claude-policy.json --tool Bash --args '{"command":"rm -rf /tmp/x"}'
verdict=DENY  reason=POLICY_BLOCK by=monitor      # destructive removal, denied by argument value

$ fak preflight --policy examples/dogfood-claude-policy.json --tool Bash --args '{"command":"ls -la"}'
verdict=ALLOW reason=NONE        by=monitor        # everyday command, allowed

$ fak preflight --policy examples/dogfood-claude-policy.json --tool Bash --args '{"command":"sudo rm x"}'
verdict=DENY  reason=POLICY_BLOCK by=monitor      # privilege escalation, denied
```

### Block self-modification (`self_modify_globs`)

The agent must not be able to rewrite its own kernel, CI, or secrets. A write into a
guarded path is refused with `SELF_MODIFY`:

```
$ fak preflight --policy examples/dogfood-claude-policy.json --tool Edit --args '{"file_path":".git/config"}'
verdict=DENY reason=SELF_MODIFY by=monitor
```

Always guard at least `.git/`, `.ssh/`, your CI workflow directory, the kernel/config
source, and the **policy file itself**.

### Validate the floor in CI (fail the build on a broken policy)

`fak policy --check` exits non-zero on a malformed manifest — wire it into CI so a floor
that widens silently, or cites an invalid reason, can never ship. (Examples of the
fail-loud errors are in the [policy guide](policy-guide.md#the-check-step-catches-mistakes-before-production).)

---

## 3. Require authentication on every non-liveness route

By default the gateway is unauthenticated — fine for `127.0.0.1` experiments, **not** for
anything reachable by another host. `--require-key-env VAR` requires a bearer token on
every route except `/healthz`. Real behavior:

```sh
FAK_TOKEN=… fak serve --addr 0.0.0.0:8080 --base-url … --model … \
  --policy floor.json --require-key-env FAK_TOKEN
```

```
# no token            -> rejected
$ curl -s -o /dev/null -w '%{http_code}\n' http://host:8080/v1/models
401

# liveness stays open (for probes/load balancers)
$ curl -s -o /dev/null -w '%{http_code}\n' http://host:8080/healthz
200

# Authorization: Bearer  (OpenAI / fak-native clients)
$ curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer …' http://host:8080/v1/models
200

# x-api-key  (Claude Code / Anthropic SDKs) — both header styles accepted
$ curl -s -o /dev/null -w '%{http_code}\n' -H 'x-api-key: …' http://host:8080/v1/models
200
```

Notes:
- The token is read from an **environment variable**, never a flag — it won't land in your
  shell history or the process arg list.
- Auth also covers `/metrics`, `/debug/vars`, and the `/v1/fak/*` routes, so a hardened
  process doesn't serve its internals open. Confirm with `gateway.auth_required: true` in
  the `/debug/vars` snapshot.
- `/healthz` is intentionally unauthenticated and reveals only `{engine, model, ok}` — no
  secrets.

---

## 4. Control network exposure

| Setting | Recommendation |
|---|---|
| **Bind address** | Prefer `--addr 127.0.0.1:PORT` and reach it via a reverse proxy / sidecar. Bind `0.0.0.0` only when you must, and only *with* `--require-key-env`. (Containers must bind `0.0.0.0` — the default `CMD` does, so keep auth on.) |
| **TLS** | `fak serve` speaks plain HTTP. Terminate TLS at a reverse proxy (nginx/Caddy/Envoy) or a service mesh — never expose the raw port across a network boundary. |
| **Metrics scraping** | Keep `/metrics` and `/debug/vars` on an internal interface or behind auth; they expose build/version/runtime detail useful to an attacker fingerprinting you. |
| **Rate limiting / quotas** | Enforce at the proxy. `fak` adjudicates correctness, not request volume. |

---

## 5. Secret hygiene

- **Redact secret-shaped arguments** with `redact_fields` (`password`, `token`, `api_key`,
  …) so a secret in a tool call is stripped to `[REDACTED]` *before* dispatch and never
  enters context. A call carrying one returns `verdict=TRANSFORM` (see the
  [policy guide](policy-guide.md#secret-hygiene-at-the-boundary-redact_fields--transform)).
- **Upstream API keys** for proxy mode are passed by env-var name (`--api-key-env`), not on
  the command line.
- **The access log never logs request bodies, tool arguments, or result content** — only
  tool *names*, verdicts, and timings — so shipping logs to a SIEM doesn't create a new
  leak path. See [observability.md](observability.md#1-the-access-log-per-request-audit-trail).

---

## 6. Defense in depth — `fak` is a layer, not the whole stack

`fak` is the **call-boundary** layer. Pair it with the controls it does *not* provide:

| Concern | Owned by |
|---|---|
| Which tools/args may run; result containment | **`fak`** (this layer) |
| TLS, rate limits, IP allow-lists, WAF | your reverse proxy / gateway |
| OS-level sandboxing of the tool executor (the thing that actually runs an allowed call) | container/VM/seccomp — `fak` decides *whether* a call runs, your runtime decides *how safely* it runs |
| Secret storage / rotation | a secrets manager (Vault, cloud KMS) |
| Authn/z of the human/agent behind the request | your identity layer (the gateway checks a bearer token, not identity) |

> **`fak` never executes your tools.** On `/v1/chat/completions` it adjudicates the calls
> the upstream model proposes and returns only the admitted ones; *your client* runs the
> survivors. So the tool executor still needs its own sandbox — `fak` controls the
> decision, not the blast radius of an allowed call.

---

## 7. Honest scope — what `fak` does NOT protect against

Stated plainly so you don't over-trust it:

- ❌ It does **not** bound the *arguments* of an allow-listed tool unless you wrote an
  `arg_rule` for it. An allow-listed `send_email` with attacker-chosen recipients is *not*
  stopped by the floor — keep exfil-shaped tools off the allow-list and let `DEFAULT_DENY`
  hold them.
- ❌ The detector that flags poisoned results is **evadable by design** — never treat a
  "clean" detector verdict as proof a result is safe. The quarantine *policy*, not the
  detector, is the protection.
- ❌ `redact_fields` / `self_modify_globs` are **best-effort** key/substring hygiene, not a
  cryptographic guarantee.
- ❌ It is not a TLS terminator, a WAF, a rate limiter, or an OS sandbox (see §6).
- ❌ The in-kernel model (Tier 2) is a *correctness reference*, not a hardened multi-tenant
  serving engine. For production serving, front a real engine via Tier 1.

Per-capability status (`[SHIPPED]` / `[SIMULATED]` / `[STUB]`) is tracked honestly in
[`fak/CLAIMS.md`](../../CLAIMS.md). To report a vulnerability, see the repository's
security policy.

---

## Deployment checklist

- [ ] Run with an explicit, **reviewed** `--policy` (not the bare default).
- [ ] Irreversible tools are **absent** from the allow-list (verified with `preflight`).
- [ ] Coarse tools (`Bash`, `Edit`) have `arg_rules` / `self_modify_globs` guarding them.
- [ ] `fak policy --check` runs in CI as a required, build-failing check.
- [ ] `--require-key-env` is set; the token comes from an env var, not a flag.
- [ ] Bound to loopback behind a TLS-terminating proxy, or `0.0.0.0` **with auth**.
- [ ] `/metrics` + `/debug/vars` are auth-gated or on an internal interface.
- [ ] `redact_fields` covers your secret-shaped argument keys.
- [ ] The tool executor that runs admitted calls has its **own** OS sandbox.

---

## See also

- [policy-guide.md](policy-guide.md) — building the capability floor (with worked examples).
- [observability.md](observability.md) — the audit log + metrics, and securing those surfaces.
- [server-config.md](server-config.md) — every flag and env var.
- [`SECURITY-capability-floor-2026-06-18.md`](../notes/SECURITY-capability-floor-2026-06-18.md) — the floor visual + the dogfood verdict matrix.
- [Policy in the kernel](../explainers/policy-in-the-kernel.md) — the design rationale for default-deny in the call path.
