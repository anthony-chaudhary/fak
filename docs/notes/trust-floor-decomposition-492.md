---
title: "fak trust-floor wiring (#492): code-grounded status"
description: "A status decomposition of fak's result-side exfil floor and trust-floor child issues, each backed by file:line evidence read against HEAD."
---

# Trust-floor wiring — code-grounded status decomposition (epic #492)

_Status snapshot taken against `HEAD = 1c0c8e1` on 2026-06-21. Every status below is
backed by a read/grep of the tree at that commit — file:line citations are real and
were verified by hand, not copied from the issue. `go build ./...` is green at this
commit (the doc changes no code)._

## TL;DR — the epic's premise has largely been overtaken by `HEAD`

Epic #492 was opened against the structural finding in
`FEATURE-SPACE-MAP-fak-2026-06-17.md`: that the result-side exfil floor was
`Decide`-vs-`Syscall` *inert on the served path*, that `k.Decide` *emitted no
events*, and that several seams were *shipped-but-dormant*. **The public release
squash (Wave A, `v0.5.0`, 2026-06-17) and the work since closed most of that gap.**
Re-verifying the three named anchors against `HEAD`:

| Issue's structural claim | Status at `HEAD` | Evidence |
|---|---|---|
| `k.admitResult` reached only by `Reap`/`Syscall`, never the served path | **closed** — `AdmitResult` is now an exported dual of `Decide` and the proxy calls it | `internal/kernel/kernel.go:184` (exported `AdmitResult`), `internal/gateway/gateway.go:571` (`admitInboundResults`), `internal/gateway/http.go:233` |
| `k.Decide` (kernel.go:94-96) emits **no events** | **false at HEAD** — `Decide` emits `EvDecide` + `EvDeny` | `internal/kernel/kernel.go:94-101` |
| gateway proxy runs `k.Decide` **only** (gateway.go:152) | **false at HEAD** — proxy also folds the result-side stack | `internal/gateway/http.go:233`, `internal/gateway/messages.go:166` |

Note: `FEATURE-SPACE-MAP-fak-2026-06-17.md`, `NEXT-STEPS-fak-2026-06-17.md`, and
`TICKETS.md` are **not present** in the tree or in `git log --all` (squashed out of
the public release). `CLAIMS.md` survives at repo root and is the claims ledger
referenced below. Where a child's status depends on something I could not witness
without running a live server, it is marked **needs-runtime-witness**.

### Roll-up

| # | Item | Status | Tick? |
|---|---|---|---|
| #7  | Arm result-side on serving path + `TraceID` | **SHIPPED** | ✅ |
| #8  | IFC config seam (Authorize/SafeSinks/RegisterSource) | **SHIPPED** | ✅ |
| #9  | Argument-level value predicates | **SHIPPED** | ✅ |
| #10 | Durable decision journal + live event stream | **PARTIAL** | ☐ |
| #11 | Default dev-agent floor + CICD pillars | **SHIPPED** | ✅ |
| #12 | Lifecycle hot-reload + bounded + fail-open | **IN PROGRESS (#493)** | ☐ |
| #13 | Throughput/cost governor (emit `RATE_LIMITED`) | **SHIPPED** (env-config residual) | ✅ |
| #14 | In-kernel model as a `RegisterEngine` backend | **SHIPPED** | ✅ |
| #15 | Per-call engine routing + `Ref.Scope` residency gate | **PARTIAL** | ☐ |

---

## #7 — Arm the result-side stack on the serving path + thread `TraceID` · **SHIPPED**

**Evidence.**
- `k.AdmitResult` is now exported as the dual of `Decide`, with a doc comment that
  spells out its purpose ("the gateway's served path calls it … so the exfil floor
  is no longer inert on the proxy/adjudicate topology"): `internal/kernel/kernel.go:184-193`
  (delegates to the unchanged `admitResult` fold at `kernel.go:195-235`).
- The gateway funnels every client-produced result through it:
  `internal/gateway/gateway.go:545` (`admitOp` → `s.k.AdmitResult`), and the auto
  proxy arms it on the inbound `role="tool"` turn: `internal/gateway/gateway.go:555-610`
  (`admitInboundResults`, whose comment is tagged `(#7)`).
- Both proxy wires call it before forwarding to the upstream model:
  `internal/gateway/http.go:233` (OpenAI `/v1/chat/completions`) and
  `internal/gateway/messages.go:166` (Anthropic `/v1/messages`).
- `TraceID` is threaded end-to-end: `internal/gateway/gateway.go:487-493` (`buildCall`
  stamps it on the `ToolCall`) and `gateway.go:495-503` (`traceFor` mints a fresh
  non-empty id when the wire omits one, so distinct sessions never collapse onto the
  empty-string trace). The sink-gate reads the same `TraceID`-keyed taint high-water
  mark on the calls the model then proposes.
- `k.Decide` now emits `EvDecide`/`EvDeny` (`internal/kernel/kernel.go:94-101`), so the
  served `adjudicate` path is observable.
- Tests pin it: `internal/kernel/kernel_test.go:232` (`TestDirectDecideEmitsDecisionAndDeny`),
  `internal/gateway/proxy_exfil_floor_test.go`, `internal/gateway/anthropic_exfil_floor_test.go`,
  `internal/gateway/admit_test.go`.

**Residual.** None for the stated scope. Optional hardening: the explicit `fak_admit`
verb and the auto proxy share `admitOp`, but a host that runs `fak_adjudicate`-only
(no result hand-back) still gets no result-side coverage by construction — that is the
client's choice, not a gap.

**Finish touch-list.** Nothing required to close. (If desired: document the
`fak_admit` round-trip contract in `docs/mcp-tool-result.md`.)

---

## #8 — Ship the IFC config seam (Authorize / SafeSinks / RegisterSource as manifest fields) · **SHIPPED**

**Evidence.**
- The policy `Manifest` carries the three IFC config fields:
  `internal/policy/policy.go:70-72` (`SafeSinks []string`, `Authorize []AuthorizeRule`,
  `Sources map[string]string`), with `AuthorizeRule` at `policy.go:79-84`.
- `ToRuntime` resolves them into a `Runtime` (`policy.go:227-244`), and
  `ApplySources` installs host-authored source classes via
  `provenance.RegisterSource` (`policy.go:383-387`).
- The host actually wires them into the live IFC engine at boot:
  `cmd/fak/main.go:908` (`policy.ApplySources(rt)`) and `cmd/fak/main.go:909`
  (`ifc.ConfigureDefaultPolicy(ifcPolicy(rt))`), where `ifcPolicy` builds an
  `ifc.Policy` with `SafeSinks` and an `Authorize` closure from the manifest
  (`cmd/fak/main.go:912-943`, sink-class mapping at `main.go:944-954`).
- `DisallowUnknownFields` (`policy.go:161-169`) means a typo'd IFC key fails loudly.

**Residual.** The seam is complete for the three named fields. The `Authorize`
closure is exact-match (one tool → one sink class, `main.go:925-928`); a host wanting
glob/predicate sink authorization would need a richer rule shape.

**Finish touch-list.** Nothing required for the stated scope. Optional: extend
`AuthorizeRule` with a matcher (mirroring `ArgRule`) if pattern-based sink release is
wanted; add an `examples/ifc-policy.json` fixture.

---

## #9 — Argument-level value predicates in the policy manifest · **SHIPPED**

**Evidence.**
- Manifest schema: `internal/policy/policy.go:73-76` (`ArgRules []ArgRule`, comment
  tagged `(issue #9)`) and the `ArgRule` matchers `allow_glob` / `deny_regex` /
  `max_bytes` at `policy.go:86-106`.
- Compilation to the hot-path form: `policy.go:465-518` (`compileArgRules` →
  `adjudicator.ArgPredicate`, "exactly one matcher" fail-loud at `policy.go:487-489`).
- Enforcement on the decide path: the predicate type and its placement are at
  `internal/adjudicator/decide.go:53-62` and `decide.go:96-113`; `Adjudicate`
  evaluates the per-tool predicates at `decide.go:215` → `decide.go:267`
  (`evalArgPredicates`), with the evaluator + bounded-disclosure deny at
  `decide.go:649-682`. Predicates are indexed by tool (`decide.go:155-165`,
  `indexArgPredicates`) so they are O(matching-tool), not O(all-rules).
- Restrict-only invariant is tested: `internal/adjudicator/adjudicator_test.go:225`
  (`TestArgPredicatesAreRestrictOnly`), `adjudicator_argindex_test.go:30`
  (`TestArgPredicatesIndexedByTool`).

**Residual.** Three matchers ship (`allow_glob`, `deny_regex`, `max_bytes`). Numeric
range / enum / JSON-pointer-nested matchers are not present; a value predicate only
reads top-level decoded args (`evalArgPredicates` decodes once at `decide.go:653`).

**Finish touch-list.** Nothing required for the stated scope. Optional: add nested-key
(`a.b.c`) addressing and a numeric range matcher to `ArgRule` + `compileArgRules` +
`evalArgPredicates`.

---

## #10 — Durable decision/audit journal + live event stream · **PARTIAL**

**Shipped half.**
- A full durable, hash-chained, append-only **decision journal**:
  `internal/journal/journal.go` — `Row` schema (`journal.go:52-66`), per-row
  `chainHash` tamper-evidence (`journal.go:315-322`), `Verify` re-reads + recomputes
  the chain (`journal.go:444-516`), crash-safe head recovery (`journal.go:405-438`).
- It is a registered `abi.Emitter`, opt-in via `FAK_AUDIT_JOURNAL`
  (`journal.go:528-544`), recording `EvDecide`/`EvDeny`/`EvQuarantine`/`EvVDSOHit`
  (`journal.go:329-359`). Because `k.Decide` now emits (`kernel.go:94-101`), served
  adjudications are captured, not just `Syscall`.
- A live **in-process** stream + bounded tail exist: `journal.go:174-206`
  (`Subscribe`, `Recent`) and `journal.go:522-526` (`Active`).

**Residual (the unshipped half).** The journal's own doc comment promises "the
gateway's `/v1/fak/events` stream" (`journal.go:23-25`), but **no such route is
wired**: the HTTP mux registers no `/v1/fak/events` (`internal/gateway/http.go:32-49`)
and the MCP dispatch has no events verb (`internal/gateway/mcp.go:234-275`). The
gateway does not import `internal/journal` at all. So for a *remote/server* consumer
(the AUD/RES/MTS/MID columns this item targets) the live stream is unreachable over
the wire; only an in-process embedder can `Subscribe`.

**Finish touch-list.**
1. `internal/gateway/http.go` — add `mux.HandleFunc("/v1/fak/events", s.handleFakEvents)`
   (SSE or long-poll) reading `journal.Active().Recent(n)` + `Subscribe()`; 404 when
   `journal.Active()==nil` (the comment at `journal.go:525` already anticipates this).
2. `internal/gateway/mcp.go` — add a `fak_events` verb mirroring `fak_changes`
   (`mcp.go:263-267`) for the MCP wire.
3. `internal/gateway/gateway.go` (Config) — optional `EventsEnabled`/auth gate so the
   audit stream is not exposed on a no-auth loopback by default.
4. Tests: a served-then-`/v1/fak/events` round-trip asserting a DECIDE row appears.

---

## #11 — Real default dev-agent floor + wire the dormant CICD pillars · **SHIPPED**

**Evidence.**
- A real, deployable dev-agent floor exists distinct from the permissive bench
  policy: `internal/adjudicator/decide.go:769-814` (`DevAgentPolicy`) — it **denies**
  the shared-history git mutations (`git_push`/`git_merge`/`git_tag` →
  `decide.go:796-801`), bounds writes away from the spine via `SelfModifyGlobs`
  (`decide.go:802-811`), and allows a single high-level `ship_release`
  (`decide.go:793`).
- The CICD pillar — the ship gate — is wired as a real adjudicator:
  `internal/shipgate/adjudicate.go:51-65` (`ShipAdjudicator.Adjudicate` returns
  `VerdictRequireWitness` for ship-shaped tools) registered at rank 40
  (`adjudicate.go:70-76`) and enabled in the defconfig (`internal/registrations/registrations.go:52-54`).
  An unwitnessed ship is fail-closed to `UNWITNESSED` by the kernel's witness fold
  (`kernel.go:152-182`, `kernel.go:271-285`).
- The floor round-trips through the manifest loader and is mirrored on disk:
  `examples/dev-agent-policy.json` (asserted by
  `internal/adjudicator/devagent_manifest_test.go:17-27`). Behavior tests:
  `internal/adjudicator/devagent_test.go` (deny-spine-self-modify, allow `ship_release`
  at the floor), `internal/shipgate/ship_default_path_test.go`.
- Recorded as shipped in `docs/releases/v0.5.0.md:23-24` (#11).

**Residual.** The `ship_release` claim string is host-supplied in `Meta["witness"]`
(`adjudicate.go:55-58`); the *catalogue* of CICD ship verbs is the fixed
`shipTools` set (`adjudicate.go:28-34`). A deployment with a differently-named ship
action must add it there (no manifest seam for ship-tool names yet).

**Finish touch-list.** Nothing required for the stated scope. Optional: expose the
`shipTools` set as a manifest field so an adopter can name its ship verb without a
code change; document the witness-claim grammar (`ancestor:<ref>`, `clean:.`) in the
policy guide.

---

## #12 — Lifecycle: hot-reload + bounded state + fail-open posture · **IN PROGRESS (issue #493)**

**Do not duplicate — this is being worked concurrently as #493.** The most recent
commit on `HEAD` is `1c0c8e1 test(lifecycle): witness policy hot-reload + bounded
ledger + live trace reset (#493)`.

**On-disk evidence the seams exist.**
- Hot-reload + live trace reset are injected into the gateway:
  `internal/gateway/gateway.go:115-119` (`ReloadPolicy`, `ResetTrace` config),
  `gateway.go:135-159` (the `PolicyReloadFunc` / `TraceResetFunc` types + wire
  request/response), and routed at `internal/gateway/http.go:43-44`
  (`/v1/fak/policy/reload`, `/v1/fak/trace/reset`).
- The fail-open-vs-fail-closed posture choice is explicitly flagged as #12's
  territory in the journal's boot path: `internal/journal/journal.go:536-537`
  ("An auditor who requires fail-closed wires that as a separate posture (issue #12)").

**Classification.** In progress; defer to #493 for the residual + touch-list. This
doc intentionally does not enumerate finish steps to avoid colliding with that work.

---

## #13 — Throughput/cost governor (emit `RATE_LIMITED`) · **SHIPPED** (config-seam residual)

**Evidence.**
- A real enforcer exists and emits the closed-vocabulary reason:
  `internal/ratelimit/ratelimit.go:217-224` (`denyVerdict` sets
  `Reason: abi.ReasonRateLimited`), over a per-key token-bucket/quota
  (`ratelimit.go:138-156`). `RATE_LIMITED` is in the reason vocab
  (`internal/abi/reasons.go:19,38`) and maps to the `WAIT` disposition
  (`internal/kernel/kernel.go:433`).
- It is **registered and live by default** at rank 8:
  `internal/ratelimit/ratelimit.go:258` (`var Default = New()`), `ratelimit.go:260-267`
  (`init` → `abi.RegisterAdjudicator(8, Default)`), enabled in the defconfig
  (`internal/registrations/registrations.go:24`). It stays inert (Defers) until a cap
  is configured.
- Issue #13's acceptance is pinned by a named test:
  `internal/ratelimit/ratelimit_test.go:198-243`
  (`TestRateLimitedDenySurfacesWaitDisposition` — "the issue-#13 acceptance witness":
  the over-cap call denies `RATE_LIMITED` and the kernel's `DenyResult` carries the
  `WAIT` disposition). Proof doc: `docs/proofs/ratelimit.md`.

**Residual.** Configuration is **env-only** today —
`FAK_RATELIMIT_MAX_CALLS` / `FAK_RATELIMIT_MAX_COST` / `FAK_RATELIMIT_KEY`
(`ratelimit.go:26-28`, `ratelimit.go:271-285`). There is **no policy-manifest seam**:
`internal/policy/policy.go`'s `Manifest` has no rate-limit fields, so a reviewer can't
diff the throughput floor the way they can the capability floor.

**Finish touch-list.**
1. `internal/policy/policy.go` — add a `RateLimits` manifest block (per-key max
   calls / max cost / key dimension) and resolve it in `ToRuntime`.
2. `internal/ratelimit/ratelimit.go` — accept the resolved caps from the host
   (a `Configure`-style setter) in addition to the env path.
3. `cmd/fak/main.go` — apply the manifest rate-limit block at boot next to
   `ApplySources` (`main.go:908`).

---

## #14 — Wire the in-kernel model as a `RegisterEngine` backend · **SHIPPED**

**Evidence.**
- The in-kernel forward pass is exposed through the engine registration seam:
  `internal/modelengine/modelengine.go:43` (`const EngineID = "inkernel"`) and
  `modelengine.go:229` (`abi.RegisterEngine(EngineID, Default)`), enabled in the
  defconfig (`internal/registrations/registrations.go:67-72`).
- It is the **default** engine, not an opt-in: `cmd/fak/main.go:178` and `main.go:1098`
  default `--engine` to `inkernel`; the gateway defaults `EngineID` to `inkernel`
  (`internal/gateway/gateway.go:200-202`) and the engine seam lists it among the local
  engines (`internal/engine/engine.go:260`).
- The architest spine pins the single legitimate registrant of the id
  (`internal/architest/architest_test.go:1597-1604`). Recorded as shipped in
  `docs/releases/v0.5.0.md:25-26` (#14).

**Residual.** None for the stated scope. (The decode itself runs a synthetic
checkpoint unless `FAK_MODEL_DIR`/`--gguf` names a real export — that is the model-load
path, orthogonal to the engine-registration seam this item is about.)

**Finish touch-list.** Nothing required to close.

---

## #15 — Per-call engine routing + a `Ref.Scope` residency gate · **PARTIAL**

**Shipped half — per-call engine routing.**
- `abi.ToolCall` carries an optional per-call route: `internal/abi/types.go:161`
  (`Engine string // optional per-call engine route; empty => kernel default`).
- The kernel honors it on dispatch: `internal/kernel/kernel.go:365-370` (`routeFor`
  returns `c.Engine` when set, else the kernel default) consumed in `Reap` at
  `kernel.go:344-349`. So a single kernel can fan calls across registered engines per
  call, preserving the process-wide binding when unset.

**Residual (the unshipped half) — `Ref.Scope` residency gate.** The scope field and
its closed isolation lattice exist — `internal/abi/types.go:71` (`Scope ShareScope`,
default `ScopeAgent`) and `types.go:91-97` (`ScopeAgent`/`ScopeFleet`/`ScopeTenant`,
with the documented invariant "a result is never shared more widely than its scope")
— **but no non-test code enforces it.** A tree-wide grep for `ScopeFleet`/`ScopeTenant`/
`Ref.Scope`/residency turns up only the type definition, the ABI enum test
(`internal/abi/abi_test.go:25`), and unrelated KV-residency telemetry; there is no
ResultAdmitter or adjudicator keyed on `Ref.Scope`. The HYB use case this item targets
needs a gate that refuses a cross-residency share (e.g. a `ScopeAgent` result reused on
another agent's fleet-scoped request).

**Finish touch-list.**
1. New leaf (e.g. `internal/residency`) — a `ResultAdmitter` (or adjudicator) that
   compares a result's `Ref.Scope` against the requesting call's scope and emits a
   refusal (a new or existing closed reason) on a widen-attempt.
2. Plumb the requesting scope onto the call/trace so the gate has both operands.
3. Register it in `internal/registrations/registrations.go` and add a residency
   routing test alongside `internal/kernel/kernel_scope_test.go` (which today covers
   only `CallScope` *tool* routing, not `Ref.Scope` residency).

---

## Method / honesty notes

- Every file:line above was opened or grepped at `HEAD = 1c0c8e1`; none is transcribed
  from the issue. `go build ./...` returns 0 at this commit.
- I could fully ground all nine children from on-disk evidence. The two **partials**
  (#10, #15) are partial because a seam is genuinely unwired in the tree, not because I
  couldn't witness them. No child required a **needs-runtime-witness** fallback — but
  note that *runtime* confirmation that the served `/v1/fak/...` paths behave as the
  code reads (e.g. an actual quarantine reaching an upstream model) would require
  booting `fak serve`, which is out of scope for a static decomposition.
- `FEATURE-SPACE-MAP-fak-2026-06-17.md` (the epic's primary ref) is absent from the
  tree and history; this decomposition stands on the live code instead.
