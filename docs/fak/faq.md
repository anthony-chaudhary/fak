# fak FAQ and common issues

Short, honest answers to the questions people ask most about `fak` — what it is, what it
guarantees, what it explicitly does **not**, and how to get unstuck. Every answer links to
the deeper doc that proves it.

> **The one-sentence version.** `fak serve` is a kernel you put *between* the model and the
> tools it wants to call: a tool that isn't on a reviewed allow-list is refused **by
> structure**, a malformed call is grammar-repaired, and a poisoned tool result is walled
> off before it reaches the model — and `fak` never executes your tools, your client does.

Jump to: [Core concepts](#core-concepts) · [Capabilities](#capability-questions) ·
[Comparisons](#comparison-questions) · [Operations](#operational-questions) ·
[Limitations](#limitations) · [Where to go next](#where-to-go-next)

---

## Core concepts

### What is `fak`?

`fak` treats the model as an untrusted program and a tool call as a **syscall**. `fak serve`
is an OpenAI- and Anthropic-compatible HTTP gateway that interposes a capability kernel
between "the model proposed a tool call" and "the tool runs." Every proposed call is
adjudicated against a reviewable policy; the gateway returns only the admitted (or repaired)
calls, plus a `fak` extension describing every decision. You can also run the kernel with no
model and no network at all (`fak preflight`, `fak run`) to test policy decisions offline.

→ [tutorial.md](tutorial.md) (zero to first adjudicated call) · [`fak/ARCHITECTURE.md`](../../ARCHITECTURE.md)

### What is `fak` vs llama.cpp vs vLLM?

They solve different problems and compose:

| | What it is | What it gives you |
|---|---|---|
| **llama.cpp / vLLM / SGLang** | inference engines | run the forward pass; per-session / per-instance KV cache reuse |
| **`fak`** | a tool-call **kernel** (and an optional in-kernel engine) | a default-deny capability floor, result quarantine, an audit trail, and cross-worker / cross-session KV reuse |

`fak` is **not** primarily an inference engine — the recommended deployment puts `fak serve`
*in front of* llama.cpp or vLLM (point `--base-url` at it) so you keep your model and gain
the kernel boundary. `fak` does ship an in-kernel engine that can load a GGUF directly, but
that path is a *correctness reference*, not a production serving engine (see
[Does `fak` work with any model?](#does-fak-work-with-any-model)). The infrastructure-level
differences (cross-worker / cross-session reuse, measured 20–24× vs naive re-prefill) are
quantified in [`docs/fak-vs-alternatives-comparison.md`](../../docs/fak-vs-alternatives-comparison.md).

### Why a kernel for tool adjudication?

Because the decision belongs at the **call boundary**, not in a prompt. A capability kernel
makes a refusal *structural*: a tool you never allow-listed is refused regardless of what is
in context, including an injection that talks the model into asking for it. The lever was
never built. The design rationale — default-deny in the call path, provable refusals — is in
[Policy in the kernel](../explainers/policy-in-the-kernel.md).

### What does "fail-closed" mean?

Two things:

1. **Default-deny.** Anything not in `allow` / `allow_prefix` and not explicitly denied
   resolves to `DEFAULT_DENY`. A tool you never named is refused — even one you didn't
   anticipate. An empty manifest (`{}`) is the maximally paranoid floor where *everything* is
   denied.
2. **Fail-loud on bad config.** A malformed manifest, an unknown refusal reason, an unknown
   posture, or an unknown JSON field is a **fatal startup error** — `fak` does not silently
   fall back to a more permissive default.

`posture: "fail_closed"` is the normal floor; the one opt-in relaxation is
`"admit_and_log"`, which admits *read-shaped* default-denies while logging
`would_deny=DEFAULT_DENY` (write-shaped calls and explicit denials still fail closed).

→ [`fak/POLICY.md`](../../POLICY.md) · [policy-guide.md](policy-guide.md)

### How does quarantine work?

Quarantine is the **second, independent gate** (the "wall"). A tool *result* the kernel
judges suspicious is held **out of the model's context** — the bytes never reach attention,
so an injection inside them can't influence the next turn. On the wire a quarantined result
shows up as a `result_admissions` entry with `verdict.kind == "QUARANTINE"`. The crucial
point: the protection is the quarantine **policy**, not the detector that flags the result.
The detector is the evadable part (see [the limitations section](#the-detector-is-evadable-by-design));
the wall holds even when the detector misses. A finished session can be persisted as a
durable core dump with `fak recall` if you need the quarantined state to survive the process.

→ [security.md §1](security.md) · [`README.md` "the lock, not the screener"](../../README.md)

### What are the "two gates" I keep reading about?

An attacker has to beat **two independent gates**, and neither is a detector you can talk
past:

| Gate | What it is | Why it holds |
|---|---|---|
| **The lock** (capability floor) | a default-deny allow-list of tools | an irreversible tool you didn't allow-list is refused *regardless of context* |
| **The wall** (result quarantine) | poisoned results held out of context | the bytes never reach attention |

The evadable detector sits *on top of* the wall; if it misses, the result is still
quarantined by policy. → [security.md](security.md)

---

## Capability questions

### Can `fak` prevent all malicious actions?

**No — and it is built so you don't have to trust that it does.** `fak`'s value is
*structural* (a refused tool was never wired up) plus *containment* (a poisoned result never
reaches the model) — not a smarter classifier. It cannot stop a malicious action performed by
a tool you **did** allow-list, it does not bound the *arguments* of an allow-listed tool, and
its result detector is evadable by design. The safe pattern is to keep irreversible / exfil-
shaped tools **off** the allow-list and let `DEFAULT_DENY` hold them. Read
[the limitations section](#limitations) before you rely on it.

### Does `fak` work with any model?

For the **gateway** (the recommended path): any OpenAI- or Anthropic-compatible upstream —
Ollama, vLLM, llama.cpp's `llama-server`, or a cloud provider — works by pointing
`--base-url` at it. The one real requirement is a **tool-calling model**: a base completion
model that never emits `tool_calls` gives the kernel nothing to adjudicate.

For the **in-kernel engine** (`fak serve --gguf …`, no `--base-url`): it loads a GGUF and runs
the forward pass inside the kernel's address space. This is a *correctness reference* proven
bit-exact against a HuggingFace oracle — **not** a hardened, production-optimized chat engine.
For chat-quality serving at scale, front a real engine via the gateway. Per-capability
status (`[SHIPPED]` / `[SIMULATED]` / `[STUB]`) is tracked honestly in
[`fak/CLAIMS.md`](../../CLAIMS.md).

→ [migration-guide.md](migration-guide.md) · [server-config.md](server-config.md)

### Can I use `fak` without Claude Code?

Yes. `fak` is client-agnostic. It speaks three wire surfaces on one port:

- **OpenAI Chat Completions** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- **Anthropic Messages** (`/v1/messages`)
- **fak-native / MCP** (`/v1/fak/*`, `/mcp`)

So the OpenAI SDK, LangChain, AutoGen, OpenAI Codex, Cursor, a raw `curl`, or your own loop
all work by redirecting the base URL. You can also skip clients entirely and use the offline
kernel verbs (`fak preflight`, `fak run`).

→ [migration-guide.md](migration-guide.md) · [api-reference.md](api-reference.md) ·
[`docs/integrations/claude.md`](../integrations/claude.md) ·
[`docs/integrations/openai-codex.md`](../integrations/openai-codex.md)

### What's the performance overhead?

The adjudication **decision itself is sub-millisecond** — a captured access-log line shows a
policy `DENY` adjudication at `duration_ms: 0.511`. In gateway/chat mode the dominant cost is
your upstream model, which is unchanged; `fak` adds the adjudication step and a local fast
path (the "vDSO") that can serve repeat decisions without touching the model. Two honest
notes:

- **Streaming is buffered.** `fak` buffers the whole upstream turn, adjudicates it, then
  re-emits a well-formed SSE stream. The wire is identical, but partial tokens are never
  passed through *before* adjudication — so a streamed response can look "burstier."
- **Measure it yourself.** `fak bench` runs the vDSO ablation (in-process vs spawned-hook),
  and the live `kernel` counters in `/debug/vars` show how much load the fast path served.

→ [observability.md](observability.md) · [`docs/fak-vs-alternatives-comparison.md`](../../docs/fak-vs-alternatives-comparison.md)

### Does `fak` execute my tools?

**No.** This is the single most important thing to internalize. On `/v1/chat/completions` and
`/v1/messages`, `fak` adjudicates the calls the model proposes and returns only the admitted
ones; **your client runs the survivors**, exactly as it does today. `fak` controls *whether*
a call runs, not the blast radius of one that does — so the executor that actually runs an
admitted call still needs its own OS sandbox.

→ [security.md §6](security.md)

---

## Comparison questions

> Across these, the recurring theme is **layer, not rival**: `fak` is the call-boundary
> decision layer. It composes with the inference engine below it and the sandbox/runtime
> around it.

### `fak` vs LangChain tools

LangChain already executes tools client-side and talks to models through a base-URL-
overridable chat client — both a perfect fit. LangChain itself has **no structural deny
floor**: it asks the model what to do and runs the tool it asked for. Putting `fak` in front
adds the kernel boundary in one line (change `base_url`); your `@tool` definitions,
`AgentExecutor` / LangGraph loop, and prompts are unchanged, and denied calls simply never
appear in the model's tool-call list.

→ [migration-guide.md → Migrating from LangChain](migration-guide.md#migrating-from-langchain)

### `fak` vs E2B sandbox

Different layers that pair well. **E2B** is an execution *sandbox* — it gives an allowed call
a safe, isolated place to *run*. `fak` decides *whether* a call runs at all and contains
poisoned *results*; it is explicitly **not** an OS sandbox and never executes your tools. The
intended composition is: `fak` for the capability decision and result containment, a sandbox
(E2B, a container, a microVM, seccomp) for the blast radius of the calls `fak` admits.

→ [security.md §6 "Defense in depth"](security.md)

### `fak` vs a Replit-Agent-style built-in guard

A built-in agent guard is a proprietary, in-product guardrail tied to one vendor's agent and
runtime. `fak` is the same *idea* — keep an agent from doing something irreversible — built as
an **open, self-hostable, model- and client-agnostic** boundary you put in front of *any*
stack, with a default-deny floor, a closed/auditable refusal vocabulary, and a result-
quarantine wall. You own the policy, you can read the code, and you can run it offline with no
model. If you're inside a single managed product and never leave it, its built-in guard may be
enough; reach for `fak` when you want a portable, reviewable boundary across providers and
clients.

### `fak` vs custom middleware

`fak` is the boundary you'd otherwise hand-roll — but with properties hand-rolled middleware
usually skips: **deny-as-value** (a refusal is a successful `200` carrying a verdict, never an
exception your client must catch), a **closed refusal vocabulary** so every deny cites a
provable code instead of free text, **result quarantine**, an audit log that records tool
*names*, verdicts, and timings but **never** request bodies / arguments / result content, and
a policy loader that is fail-loud and round-trip-stable. It's deterministic and test-backed,
so the boundary itself doesn't become the thing you debug at 2 a.m.

→ [api-reference.md → A refusal is not an error](api-reference.md#a-refusal-is-not-an-error) ·
[`fak/POLICY.md`](../../POLICY.md)

---

## Operational questions

### How do I debug a denied call?

Check the call against the policy with **no server at all**:

```bash
fak preflight --policy policy.json --tool git_push --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor
```

The `reason` tells you *why*: `DEFAULT_DENY` (never allow-listed → add it to `allow` /
`allow_prefix` if it's legitimate), `POLICY_BLOCK` (an explicit `deny` entry), `SELF_MODIFY`
(a write into a `self_modify_globs` path), `SECRET_EXFIL`, and so on — all from the
[closed refusal vocabulary](../../POLICY.md). In the running gateway, the same decision is
in the access log (`event: gateway_operation`, with `tool`, `verdict`, `reason`,
`disposition`, and a `trace_id`) and in the per-response `fak.adjudications` array.

→ [observability.md](observability.md) · [policy-guide.md](policy-guide.md)

### Can I change policy at runtime?

Yes, when `fak serve` was started with `--policy FILE`: edit the file on disk and POST to the
reload route.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/policy/reload -d '{}'
```

Reload is **replace, not merge** — the new manifest *is* the whole floor, so start from
`fak policy --dump` and `fak policy --check` it before reloading so you never widen the floor
by accident. (`SIGHUP` and signed manifests are roadmap; the HTTP reload route is what's
shipped today.)

→ [server-config.md](server-config.md) · [`fak/POLICY.md` → Roadmap](../../POLICY.md)

### How do I monitor `fak` behavior?

Three correlated, on-by-default surfaces, tied together by a `trace_id`:

| Surface | Route | Use it for |
|---|---|---|
| **Metrics** | `GET /metrics` | Prometheus dashboards, alerts, SLOs |
| **Live snapshot** | `GET /debug/vars` | "what is this process doing right now" |
| **Access log** | stdout / log sink | per-request audit, incident forensics |

The `kernel` block in `/debug/vars` (`submits`, `denies`, `transforms`, `quarantines`,
`admitted`, `vdso_hit_ratio`) is the running tally of what the gate has been doing. Crucially,
**none of these log request bodies, tool arguments, or result content** — only tool names,
verdicts, and timings — so you can ship the log to a SIEM without creating a new leak path.
Gate `/metrics` and `/debug/vars` behind auth or an internal interface in production.

→ [observability.md](observability.md)

### What happens when `fak` crashes?

- **No silent bypass.** A client pointed only at `fak` gets a connection error when the
  gateway is down — calls **fail**, they do not silently run unadjudicated. (If your *own*
  client has a fallback straight to the upstream, that's your bypass to remove.)
- **The floor reloads from disk.** The policy is read from the reviewed manifest at startup
  (and on reload), so a restart re-reads the same floor — there's no mutable security config
  that a crash could lose into a more permissive state.
- **Live quarantine state is in-memory.** The taint ledger backing quarantine is
  process-local, so live taint marks reset on restart; use `fak recall` to persist a finished
  session as a durable core dump if you need that state to survive.
- **`fak` does not supervise itself.** Run it under a process supervisor (systemd, a container
  restart policy, etc.) for automatic restart.

→ [deployment-guide.md](deployment-guide.md) · [security.md](security.md)

### How do I require authentication?

Auth is **off by default** (loopback-friendly). For anything reachable by another host, set
`--require-key-env VAR`, which requires a bearer token on every route **except** `/healthz`:

```bash
export FAK_TOKEN="$(openssl rand -hex 32)"
fak serve --addr 0.0.0.0:8080 --base-url … --model … \
  --policy floor.json --require-key-env FAK_TOKEN
```

The token is read from an **environment variable**, never a flag (so it never lands in shell
history or the process arg list). `fak` accepts it under either `Authorization: Bearer …`
(OpenAI / fak-native clients) or `x-api-key: …` (Anthropic / Claude clients). Auth also
covers `/metrics`, `/debug/vars`, and `/v1/fak/*`.

→ [security.md §3](security.md) · [server-config.md](server-config.md)

### Every tool call is denied — what did I do wrong?

Almost always: **no `--policy` loaded**, so the kernel default-denies everything. Author a
floor (`fak policy --dump > policy.json`, edit, `fak policy --check policy.json`) and pass
`--policy policy.json`. Other common gateway gotchas:

| Symptom | Cause / fix |
|---|---|
| `404` on `/v1/v1/messages` | You included `/v1` in an **Anthropic** base URL — point Anthropic SDKs at the *origin* (`http://127.0.0.1:8080`); OpenAI clients *do* include `/v1`. |
| `401 Unauthorized` | `--require-key-env` is set — send `Authorization: Bearer …` or `x-api-key: …` (a bare `Authorization` value with no `Bearer ` prefix is rejected). |
| `502` from `/v1/chat/completions` | Upstream model error, or the model announced tool calls but none parsed (fail-closed). Fix `--base-url` first. |
| Model ignores tools entirely | Use a tool-calling model — base completion models don't emit `tool_calls`. |
| `/v1/fak/syscall` returns empty | The fak-native key is `arguments`, **not** `args` — unknown keys are dropped. |

→ [migration-guide.md → Troubleshooting](migration-guide.md#troubleshooting) ·
[server-troubleshooting.md](server-troubleshooting.md)

---

## Limitations

`fak` is built to survive a skeptic reading the code, so the honest scope is stated plainly.

### What `fak` cannot protect against

- ❌ **Arguments of an allow-listed tool.** The floor bounds *which tools* run, by tool
  *name* — it does **not** bound the *values* an allow-listed tool is called with. An
  allow-listed `send_email` with attacker-chosen recipients is *not* stopped by the floor.
  Keep exfil-/irreversible-shaped tools **off** the allow-list and let `DEFAULT_DENY` hold
  them. (Argument-level value predicates are a [roadmap item](../../POLICY.md), not
  shipped; `redact_fields` and `self_modify_globs` are best-effort key/substring hygiene, not
  a cryptographic guarantee.)
- ❌ **The blast radius of an admitted call.** `fak` decides *whether* a call runs, not how
  safely it runs — it is not a TLS terminator, a WAF, a rate limiter, or an OS sandbox. Pair
  it with those (see [security.md §6](security.md)).
- ❌ **Request volume.** `fak` adjudicates correctness, not throughput; enforce rate limits and
  quotas at your proxy.

### <a id="the-detector-is-evadable-by-design"></a>The detector is evadable by design

The detector that *flags* poisoned results is **≈100% evadable by design** — never treat a
"clean" detector verdict as proof a result is safe. The protection is the quarantine
**policy**, not the detector. Treat a detector hit as a helpful bonus, never the floor.

### Model hallucination risks

`fak` constrains what the model can *do*, not whether the model is *right*. A model can still
hallucinate a plausible-but-wrong answer, propose a call to a tool that doesn't exist (the
kernel will `DEFAULT_DENY` an unknown tool), or emit a malformed call (grammar-repaired to
canonical arguments where possible, fail-closed otherwise). The kernel bounds the *effect*;
it does not make the model smarter.

### Third-party tool dependencies

Because **your client executes the admitted calls**, the safety of what actually happens still
depends on your tools and their runtime. An admitted call into a buggy or compromised
third-party tool can still do damage inside that tool's own permissions — which is exactly why
the executor needs its own sandbox and why irreversible operations should stay off the
allow-list.

### Known edge cases and wire gotchas

- **The in-kernel engine is a reference, not a serving engine** — front a real engine for
  production chat ([`fak/CLAIMS.md`](../../CLAIMS.md)).
- **Streaming is buffered then re-emitted** — correct on the wire, but not token-by-token
  passthrough before adjudication.
- **The response extension key is `fak`** (with `adjudications` / `result_admissions`). Some
  older integration pages show `_fak` / `admissions`; verify against
  [api-reference.md](api-reference.md#the-fak-response-extension) if your client parses it.
- **Anthropic base URLs take the origin, not `.../v1`**; OpenAI base URLs include `/v1`.
- **`fak`-native syscalls use `arguments`, not `args`** — unknown keys are silently dropped.

---

## Where to go next

| If you want to… | Read |
|---|---|
| Run `fak` for the first time (real output at every step) | [tutorial.md](tutorial.md) |
| Install the binary (Docker / prebuilt / source) | [`INSTALL.md`](../../INSTALL.md) · [`fak/GETTING-STARTED.md`](../../GETTING-STARTED.md) |
| Get a gateway running fast | [server-quickstart.md](server-quickstart.md) |
| Look up every flag and env var | [server-config.md](server-config.md) |
| Look up every endpoint and field | [api-reference.md](api-reference.md) |
| Fix a startup / port / model-load problem | [server-troubleshooting.md](server-troubleshooting.md) |
| Build a capability floor (worked examples) | [policy-guide.md](policy-guide.md) · [`fak/POLICY.md`](../../POLICY.md) |
| Harden a network-facing deployment | [security.md](security.md) · [deployment-guide.md](deployment-guide.md) |
| Wire up metrics, logs, and traces | [observability.md](observability.md) |
| Move an existing stack (LangChain / AutoGen / llama.cpp / OpenAI) over | [migration-guide.md](migration-guide.md) |
| Understand agent ↔ kernel integration | [agent-integration-architecture.md](agent-integration-architecture.md) |
| Understand the system design | [`fak/ARCHITECTURE.md`](../../ARCHITECTURE.md) · [Policy in the kernel](../explainers/policy-in-the-kernel.md) |
| See what's `[SHIPPED]` vs `[SIMULATED]` vs `[STUB]` | [`fak/CLAIMS.md`](../../CLAIMS.md) |
| Compare infrastructure efficiency vs alternatives | [`docs/fak-vs-alternatives-comparison.md`](../../docs/fak-vs-alternatives-comparison.md) |
| See the rest of the docs backlog | [documentation-roadmap.md](documentation-roadmap.md) |
