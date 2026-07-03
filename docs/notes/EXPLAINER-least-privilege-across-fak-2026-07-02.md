---
title: "Least privilege across fak: one principle, many enforcement axes (2026-07-02)"
description: "Maps the classical least-privilege principle onto fak's mechanisms — the policy floor, arg rules, IFC taint attenuation, L3 ShareScope, lanes/leases, witness authority — and names the inversion that makes it load-bearing for model subjects."
---

# Least privilege across fak — one principle, many enforcement axes

> **Thesis.** Saltzer & Schroeder (1975) state least privilege as: *every program and
> every user should operate using the smallest set of privileges necessary to complete
> the job.* Nearly every fak mechanism is this one principle re-derived for a new kind
> of subject — a model that can be *persuaded*. The re-derivation changes the design
> rules: grants must be structural (below the model layer), privileges should shrink
> as the trace touches untrusted data, and the subject's own word can never widen its
> grant. This note maps the principle onto the concrete mechanisms, names the design
> rules that fall out, and lists where the mapping is still `not yet`.

Companions: [`SECURITY-capability-floor-2026-06-18.md`](SECURITY-capability-floor-2026-06-18.md)
(the tool-axis visual), `EXPLAINER-trust-floor-two-lenses-2026-06-17.md` (the ring-3
inversion), [`POLICY.md`](../../POLICY.md) (the manifest schema this note cites).

## The subject inversion — why least privilege is load-bearing here

Classical least privilege assumes a *fixed* subject: a program whose behavior is
determined by its code, granted minimal privilege as defense-in-depth against bugs and
compromise. Misuse is the exception path.

An LLM agent inverts this. The subject's behavior is a function of its **context**, and
part of that context is attacker-writable (tool results, fetched pages, issue text). So
the honest planning assumption is: **any privilege the agent holds will eventually be
exercised under adversarial prompting** — injection converts "may" into "will". Three
consequences, all visible in fak's design:

1. **The grant decision moves below the model.** A privilege enforced by the model's
   judgment ("please don't call `refund_payment`") is not a privilege boundary at all —
   it is a suggestion to the attacker. fak adjudicates every call *before dispatch*
   from evidence the model did not author ([`internal/adjudicator`](../../internal/adjudicator/)).
2. **Default-deny is mandatory, not hygiene.** `POLICY.md`'s fail-closed posture — an
   empty manifest denies everything — is the only posture under which "I forgot to
   deny X" is safe.
3. **Keep irreversible capability off the grant entirely.** `POLICY.md`'s honest-scope
   section says it outright: argument-level defense of an allow-listed tool is
   best-effort, so the load-bearing move is to *not allow-list* exfil-shaped tools and
   let `DEFAULT_DENY` hold them. Least privilege beats detection.

## The map — one principle, per axis

| Axis | Classical analog | fak mechanism | Where |
|---|---|---|---|
| Tool surface | seccomp / syscall filter | default-deny allow-list manifest; per-agent manifests against one binary | [`POLICY.md`](../../POLICY.md) |
| Argument values | argument-aware sandbox | `arg_rules` RE2 deny-by-value (`Bash` allowed, `rm -rf` denied) | [`examples/dogfood-claude-policy.json`](../../examples/dogfood-claude-policy.json) |
| Read-only families | RO file mounts | `allow_prefix` (`read_`, `get_`, `search_`, …) grants a *shape* of tool, not a name list | `POLICY.md` schema |
| Self-modification | W^X, ring separation | `self_modify_globs` (both write paths: arg targets *and* shell command strings); `CORE_SELF_MODIFY` needs an independent witness | `POLICY.md` · AGENTS.md guard table |
| Information flow | taint tracking / no-exfil-after-read | IFC per-trace taint **high-water mark** gates egress sinks; provenance-keyed so a paraphrase cannot launder it | [`internal/ifc/ifc.go`](../../internal/ifc/ifc.go) |
| Shared cached data | MMU page protection + capabilities | L3 `ShareScope`: only `ScopeFleet` crosses a tenant boundary; `ScopeAgent`/`ScopeTenant` refused (`L3_CROSS_TENANT_SCOPE_DENIED`) | [`internal/gateway/l3share.go`](../../internal/gateway/l3share.go) |
| Filesystem extent | chroot / jail | `OUT_OF_TREE_WRITE`: writes resolving outside the workspace refused at the hook layer | [`docs/repo-guard.md`](../repo-guard.md) |
| Concurrency (space × time) | advisory locks with owners | lanes + leases: write authority over exactly one lane's globs, exclusive/shared mode, TTL'd; overlap refused (`COLLISION_RISK`) | [`dos.toml`](../../dos.toml) `[lanes]` · `dos arbitrate` |
| Staging | — | commit-by-path, never `git add -A`; `fak commit` asserts committed set == requested set (`PATHSPEC_RACE`) | AGENTS.md hard rules |
| Quantity | rlimits / quotas | `rate_limit` `max_calls` / `max_cost` — privilege is also *how many*, not only *which* | `POLICY.md` schema |
| Secret exposure | env scrubbing | `redact_fields` strips secret-shaped arg values before dispatch | `POLICY.md` schema |
| Assertion authority | attestation | the witness ladder: a self-report carries **zero** authority; only non-agent-authored evidence (diff, registry, CI) raises a claim to shipped | `dos verify` · AGENTS.md proof-by-default |
| Escalation | one-shot sudo | `FLEET_ALLOW_FRESH_DELETE=1` / `FLEET_ALLOW_LEAK=1` override **once**, per event; core-lock maintenance needs `--core-lock-maintenance-witness` resolved by an independent witness | AGENTS.md guard table |

Four of these deserve expansion — they are where fak *extends* the classical principle
rather than merely instantiating it.

## 1. Privilege attenuates monotonically with exposure (IFC)

The manifest is a *static* floor; the IFC ledger makes privilege **dynamic and
monotone-shrinking**. Each trace carries a taint high-water mark
(`internal/ifc/ifc.go`); once untrusted input enters the trace, the mark rises and
egress-shaped sinks become impossible for that trace — the privilege is *removed*, not
watched. The mark never lowers mid-trace; only an operator-approved session boundary
(`POST /v1/fak/trace/reset`) clears it. Keying on **provenance** rather than content is
what makes the attenuation robust: the model summarizing the tainted bytes in its own
words does not launder the taint.

Classical least privilege rarely does this — a Unix process keeps its privileges no
matter what it reads. The agent setting demands it, because *reading is how the subject
gets compromised*.

## 2. Grants are explicit allowlists, never rank comparisons (L3 ShareScope)

[`l3share.go`](../../internal/gateway/l3share.go) contains a small design lesson worth
generalizing. The scope enum is **not monotone** for the cross-tenant question
(`ScopeTenant`'s numeric value exceeds `ScopeFleet`'s), so the crossing check is an
explicit `s == ScopeFleet` allowlist — a `>=` rank compare would have been a silent
privilege-ordering bug, and any *future* scope fails closed automatically.

The rule: **least privilege wants capability checks, not level comparisons.** An
ordered "clearance" model invites the assumption that higher rank ⊇ lower rank, which
is exactly how privilege creeps. Every fak grant surface follows this — the policy
`allow` list, the lane trees, the share scopes — membership, not order.

## 3. The kernel applies least privilege to itself (narrow-only fast paths)

[`internal/adjudicator/rungprofile.go`](../../internal/adjudicator/rungprofile.go): a
`RungProfile` lets the read path elide refusal rungs that are provably inert for reads
(a latency optimization), but `SetPolicy` **clamps any attempt to elide a mandatory
write-class rung** — the profile may only *narrow* the floor, never widen it. The zero
profile reproduces the full rung sequence byte-for-byte.

This is least privilege applied to fak's own configuration surface: even the
performance knob is structurally incapable of granting anything. The same shape appears
in the `pythongate` ratchet (the grandfathered baseline only shrinks) and the additive-only
frozen ABI. **Monotone-narrowing invariants are how a system keeps least privilege
under its own evolution**, not just under its subjects' requests.

## 4. Least privilege over *claims*, not just actions

The classical principle governs what a subject may **do**. fak extends it to what a
subject may **assert**: an agent's self-report has the least possible authority — none.
"Shipped" is raised only by evidence the agent did not author (`dos verify`: registry
row, stamp-grammar grep over real commits; `dos commit-audit`: the diff git recorded).
`RUN_STATUS_CLAIMED_FIELD` even bans a `claimed` field from run digests *structurally*,
so a peer cannot accidentally consume self-reported status.

Escalation follows the same grammar: it is an **event, not a state**. A
`FLEET_ALLOW_*` override authorizes one commit; `CORE_SELF_MODIFY` clearance requires a
witness *other than the requester*. Compare `sudo` timestamps — a standing elevated
state — with what fak does: per-event grants that decay immediately.

## Where the mapping is `not yet` (honest scope + next checkable steps)

- **Argument-level grants are still coarse.** An allow-listed `send_email` with
  attacker-chosen recipients leans on detection, not structure (`POLICY.md` honest
  scope). The roadmap's structured value predicates (path-resolution, ranges,
  allow-by-arg) are the least-privilege deepening of this axis. *Next step:* pick one
  high-risk allow-listed tool in the dogfood manifest and prototype a structured
  predicate for it.
- **Delegation attenuation is not yet a stated invariant.** Least privilege on
  dispatch means a sub-worker's floor should be **⊆** its dispatcher's floor,
  intersected with its task lane. `RungProfile` proves the narrow-only clamp pattern
  exists in-kernel; I did not find an equivalent subset assertion on the dispatch path
  (grep for a policy-subset check came up empty). *Next step:* an architest-style floor —
  a test that a dispatched worker's effective manifest cannot contain a grant its
  parent lacks.
- **Grants are session-scoped; tasks are shorter than sessions.** Lanes already model
  privilege with a TTL; the tool floor does not — a manifest holds for the whole run
  even when only one phase needs the write-shaped grants. *Next step:* a phase-scoped
  manifest overlay (narrow-only, per the §3 rule) that drops write grants outside the
  implementing phase.
- **`admit_and_log` is the deliberate relief valve** — and it is shaped correctly:
  it admits only *read-shaped* `DEFAULT_DENY` calls, still logs `would_deny`, and
  leaves write-shaped calls fail-closed. Worth preserving as the template for any
  future posture: relaxation must be class-scoped, logged, and never touch the
  irreversible set.

The one-line summary: **fak is least privilege made structural for a persuadable
subject — grants below the model, attenuating with exposure, membership not rank,
narrow-only under evolution, and with zero authority granted to the subject's own
word.**
