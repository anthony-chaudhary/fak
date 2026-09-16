---
title: "Native harness default-on security features"
description: "What the fak native harness gives you by default — the capability floor, result quarantine, JIT secret paging, provenance, and witness gates — and the harnesskit contract a builder inherits rather than re-implements."
---

# Native harness default-on security features

The fak native harness is not a blank agent loop you bolt policy onto. It ships a
security substrate that is **on by default** at the call boundary: a default-deny
capability floor, a write-time result gate that pages poison out of context, a
just-in-time credential path that keeps secrets out of the model context and off
the wire, kernel-authored provenance, and effect-verify witness gates. This page
collects the features a builder inherits for free and names the `pkg/harnesskit`
contracts that expose the same posture to external products.

> **Read the honest scope.** As with [docs/fak/security.md](../fak/security.md) and
> the [threat model FAQ](../faq/security-threat-model.md): the load-bearing
> guarantee is **structural** (a refused tool was never wired up; a quarantined
> result never reaches attention). The detector that flags suspicious content is
> **deliberately evadable** and is a bonus, never the floor. Do not deploy the
> detector as your protection.

---

## 1. The default posture in one table

| Feature | Default | Where it lives | What it buys you |
|---|---|---|---|
| **Capability floor** | default-deny allow-list | `internal/adjudicator/default_policy.go` `DefaultPolicy` / `DevAgentPolicy` | an irreversible tool never allow-listed is refused regardless of context |
| **Result quarantine** (write-time gate) | on (rank 5, `FAK_NORMGATE!=off`) | `internal/normgate/normgate.go` | a poisoned/injection result is paged out to a stub before it can enter context |
| **Secret handling** | warn-first redact (opt-in hard seal) | `normgate.admitSecret` → `internal/canon` `RedactSecrets` | the credential span is masked in place; the rest of the output stays in context |
| **JIT secret paging** (harnesskit) | declared per tool, resolved at the execution boundary | `pkg/harnesskit/tool.go` `AuthBinding` / `SecretStore` | secrets never enter model context or serialized tool parameters |
| **Provenance / taint** | source-stamped by the kernel | `internal/provenance` (`Trusted`), `abi.Ref.Taint` | the model cannot self-declare its own content trusted |
| **Info-flow control (IFC)** | tainted→sink flows sink-gated (rank 30) | `internal/ifc` | a flow from an untrusted source to a sensitive sink is gated at adjudication |
| **Witness gates** | `require-witness` rung fails closed | `internal/witness`, `internal/shipgate`, `internal/witnessprocess` | a claim must be corroborated by evidence the agent did not author |
| **Plan CFI** | plan control-flow integrity adjudicator | `internal/plancfi` | a plan that drifts from its approved shape raises `RequireApproval` |
| **Durable-memory gate** | expire-by-default promotion | `internal/ctxmmu` `GateEphemeral`, `disposition.go` | a situational fact cannot silently become a standing belief |
| **KV/context containment** | quarantined taint refused for attention | `internal/ctxmmu`, `internal/contextq` | a quarantined span is mechanically denied to attention |

Each row has a witness test in its package; the consolidated claim is
`docs/claims/security-substrate-the-kernel-stops-believing-the-model.md`.

---

## 2. The secret page **in and out** (the question this page exists to answer)

The "secret page in and out" is two related mechanisms on one content-addressed
page model.

### 2.1 Out: a secret-shaped result is held out of context

At the write-time admission gate, when the canonical scanner
(`internal/canon`, a de-obfuscating normalize-and-rescan) finds a secret, the
default policy is **warn-first redact** — not a blind seal:

1. If the fail-closed posture is active (opt-in, for unattended/untrusted
   contexts) → **SEAL** the whole result: `normgate.quarantineOut` pages the bytes
   to the shared CAS and replaces the payload with a
   `{"_quarantined":true, "reason":"SECRET_EXFIL", ...}` stub. The bytes never
   reach attention.
2. If the secret is **obfuscated** and no raw-locatable span exists
   (`canon.RawSecretComplete == false`) → also **SEAL**: the in-place redactor
   cannot reach it, so the permissive path must not admit it.
3. Otherwise (the default) → **REDACT**: `canon.RedactSecrets` masks the
   credential span in place and the rest of the legitimate output stays in
   context (a `Transform` carrying `ReasonSecretRedacted`). There is no
   paged-out stub to re-read, which is what livelocked a real session under the
   older seal-everything policy.

The same seam handles general PII (`admitPII`) and injection (provenance-aware:
trusted-local and low-confidence injection are *retrievable transforms*, never
the loud seal).

### 2.2 In: a held page is gated on the way back

`normgate.PageIn` is the gated read of the held quarantine map. It enforces two
gates, mirroring `internal/recall`'s read-time re-screen:

1. a witness `Clear(id)` must have run — an uncleared (or unknown/evicted) id is
   refused **fail-closed**;
2. the retrieved bytes are **re-screened** through `canon.Scan` — a SECRET
   re-screen hit refuses release even after a clearance. *Clearance does not
   launder a credential back into context.*

The held ledger is FIFO-bounded (`DefaultMaxHeld = 8192`, override
`FAK_NORMGATE_MAX_HELD`) so a poison-heavy stream cannot grow it without bound;
held bytes are pinned in the CAS so the bound cannot reclaim them before the
gated page-in resolves them.

### 2.3 JIT credential paging (the builder-facing twin)

External builders do not hand-roll the "in and out" for tool credentials.
`pkg/harnesskit` declares it: `AuthBinding` / `AuthRequirement` name an
`AuthType` (`fleet_secret`, `oauth2`), a `SecretKey`/`SecretRefs`, and a
`ScrubSecretsFromResults` policy. At the execution boundary the runtime:

1. resolves the secret through the host `SecretStore`/`OAuthTokenProvider`
   (deny `AUTH_SECRET_MISSING` / `OAUTH_TOKEN_EXPIRED` if it cannot) — **before**
   the handler runs;
2. scrubs secret references, Bearer tokens, and sensitive headers from the
   result on the way out (`ScrubResult` → `scrubSensitiveBytes`).

The public contract states the invariant literally:

> `"secrets are paged just-in-time and never enter model context or serialized
> tool parameters"` — `PublicToolContract().Security`

That is the harnesskit form of the same principle as §2.1/§2.2: the secret lives
at the execution boundary, never in the context the model attends to.

---

## 3. What the builder inherits via `pkg/harnesskit`

`harnesskit` is deliberately **contract-only** — importing it does not expose the
engine, policy, gateway, or adjudicator internals (`harnesskit.go` package doc).
The security posture is expressed as declarations the host enforces:

- **Registration is reachability, never authority.** "The host supplies only
  `Services`; every `Invoke` remains subject to the effective session/tenant
  capability floor." A builder cannot widen its own authority by declaring
  `Requires []Capability`.
- **`ToolScope`** declares path scopes, network allowance, mutability
  (`read_only` / `mutating` / `destructive`), rate limits, and required
  capabilities — the vocabulary the host intersects with policy.
- **`ToolCondition` / `Condition`** express dynamic prerequisites (phase, role,
  prior-success) as named gates, not prose the model can talk past.
- **`AuthBinding`** is the JIT secret-paging declaration from §2.3.
- **Transport config stays out of the portable spec** so secrets do not enter the
  serialized product description (`Transport` comment).

The normative contracts are `PublicContract()` / `ContractJSON()`
(`docs/harness-kit-contract.md`) and `PublicToolContract()`
(`fak.harness.tools/v1`).

---

## 4. How a product turns these on (and off)

- **Default-on, no action required:** normgate runs at rank 5 in front of ctxmmu
  (one blank import); the default floor is the registered `adjudicator.Default`.
- **Tighten:** supply a reviewed `--policy` (never the bare default in prod);
  keep irreversible tools off the allow-list; add `arg_rules` /
  `self_modify_globs`; opt into the fail-closed secret posture for unattended
  contexts.
- **Escape hatches (blunt, reversible, per-process):**
  - `FAK_NORMGATE=off` makes `Admit` a no-op `Defer` (also disables the
    injection screen — a dev escape hatch, not a prod posture).
  - `FAK_NORMGATE_MAX_HELD` bounds the held ledger.
- **Never:** treat a clean detector verdict as proof of safety; treat the
  screener as load-bearing.

---

## 5. Honest scope (what is *not* default-protected)

- The **detector** is evadable by design — the quarantine *policy* and the
  capability lock are the protection.
- `fak` does **not** bound the resolved *effect* of an allow-listed coarse
  tool's arguments unless you wrote an `arg_rule`; keep exfil-shaped and
  destructive tools off the allow-list.
- `fak` **decides** whether a call runs; it does **not** sandbox the admitted
  call's side effects — pair it with an OS/sandbox layer.
- `redact_fields` / `self_modify_globs` / `RedactSecrets` are best-effort
  key/substring hygiene, not a cryptographic guarantee.
- KV/kvmmu eviction of a quarantined span is proven on a synthetic model and is
  not yet wired into the live HTTP loop; the context-side page-out is shipped.

Per-capability status is tracked in `CLAIMS.md` under
*Security substrate (the kernel stops believing the model)*.

---

## See also

- [Native harness orchestration & management](native-harness-orchestration-and-management.md) — the supervisor/orchestrator paradigm this security substrate rides on.
- [`docs/harness-kit-contract.md`](../harness-kit-contract.md) — the public builder contract.
- [`docs/fak/security.md`](../fak/security.md) — the operator hardening guide (floor, auth, network).
- [Security and the threat model FAQ](../faq/security-threat-model.md) — the Q&A form.
- `examples/session-reload/` — a runnable proof that a seal survives a process boundary and a tampered image fails closed.
- `examples/wire-quarantine-demo/` — the secret-shaped-result → quarantine → page-out demo.
