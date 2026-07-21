# RESEARCH — Org Policy Plane: prior art + threat model (2026-07-20)

> Research note for **epic #5315 — Org Policy Plane (centrally-administered `fak`
> fleets)**, delivering the R1 child **#5316**. Parent epic #5315 extends
> **#5170 (Policy Amendment Classes)** by adding a sixth amendment channel,
> `central`. This note surveys prior art for centrally-administered
> endpoint/agent policy, gives a BORROW/AVOID table, enumerates the threat model,
> and makes concrete mechanism recommendations for the signing (#5317),
> precedence (#5318), envelope (#5320), and enrollment (#5323) tickets. **No code
> changes.**

**Cross-links:** epic #5315 (this note is its R1), amendment-class model #5170,
signing spike #5317, precedence spike #5318, work tickets #5319–#5325.
Anchor code: [`internal/policy/amendment.go`](../../internal/policy/amendment.go),
[`POLICY.md`](../../POLICY.md).

Ticket shorthand used below (from #5315's decomposition):

| Tag | Issue | What it is |
|---|---|---|
| R1 | #5316 | this note (prior art + threat model) |
| R2 | #5317 | signing / freshness / key-distribution spike |
| R3 | #5318 | precedence + entitlement-semantics spike |
| W1 | #5319 | `central` amendment channel in the registry |
| W2 | #5320 | `fak-org-policy/v1` signed envelope + verifier |
| W3 | #5321 | remote pull + last-good cache + offline refuse-to-widen |
| W4 | #5322 | precedence fold + `fak org status` provenance |
| W5 | #5323 | `fak enroll` trust anchor + device identity |
| W6 | #5324 | central usage-entitlement grants (enable-more) |
| W7 | #5325 | redacted org audit-telemetry sink |

---

## 0. The fak-native seam this must extend (not replace)

`fak`'s floor today is authored **per checkout**: a compiled-in
`DefaultPolicy`/`DevAgentPolicy` baseline, an on-disk `fak-policy/v1` manifest
(`POLICY.md`), and `.fak/guard/{allow,deny}.json` overlays. #5170's
`internal/policy/amendment.go` already answers, for every policy knob, **who may
amend it, in which direction, through which channel** — the closed vocabulary is:

- **Classes:** `FROZEN` (no channel weakens it), `RATCHET` (any channel tightens,
  none widens), `GATED_WIDEN` (a gated non-agent channel may loosen; always
  journaled), `SELF_AMENDABLE` (empty today).
- **Channels:** `compiled-in`, `operator-overlay`, `live-reload`,
  `operator-escalation`.

The org plane is **one new channel — `central` — not a new floor and not a new
class.** It sits, in authority, **below the compiled-in FROZEN floor and above
the operator overlay**. Its two moves both already exist in the class algebra:
`central` may **RATCHET** the whole fleet (honored for free by the existing
most-restrictive fold) and may **GATED_WIDEN per enrolled device/user/group** up
to — never past — the FROZEN cap. Everything below is chosen to keep that
property: *the org channel is powerful enough to enable more usage, but it is
structurally incapable of moving a FROZEN knob, and a stale/forged/replayed org
manifest can only ever make the floor stricter.*

This is the discipline the prior-art survey is measured against: we borrow
mechanisms only where they preserve refuse-to-widen and the FROZEN cap.

---

## 1. Prior-art survey

### 1a. Endpoint fleet management — MDM / Intune / Jamf

Config-profile distribution is the closest operational analogue: an IT admin
authors a profile once and a fleet of enrolled endpoints reads, obeys, and
reports against it.

- **Enrollment / trust anchor.** Apple MDM and Intune both establish a device
  identity at enrollment: the device gets a certificate (SCEP/PKCS), the MDM
  server is pinned, and thereafter targeted commands/profiles are addressed to
  that device identity. This is exactly `fak enroll` (W5): pin the org root key +
  mint a device id, TOFU-at-enroll.
- **Targeting / groups.** Profiles are scoped to device/user/group ("smart
  groups" in Jamf, Azure AD groups in Intune). This is `target` selector in the
  W2 envelope and the per-device entitlement grant in W6.
- **Signed profiles.** `.mobileconfig` profiles can be CMS/PKCS#7-signed so the
  device verifies issuer before applying. Directly informs W2.
- **Revocation & compliance reporting.** MDM can revoke a profile and receives
  compliance/inventory reports — the IT-visibility half (W7).

**AVOID:** the MDM *push* transport (APNs / a persistent server-initiated
channel) and its heavyweight PKI/CA lifecycle. `fak` ships one static
stdlib-only binary; a pull-on-startup + periodic-refresh loop with an optional
push over the *existing* `POST /v1/fak/policy/reload` seam (W3) is the right
weight. Also avoid MDM's implicit *fail-toward-managed-default* on profile
removal — `fak` must fail toward the compiled-in floor, not toward whatever the
last managed state was.

### 1b. Signed policy/config bundles — TUF, Sigstore, OPA/Rego bundle API

- **TUF (The Update Framework).** Purpose-built for exactly our worst case:
  *an attacker who controls the distribution endpoint should not be able to serve
  a stale or forged bundle.* TUF's load-bearing ideas for us: **explicit
  signed metadata separate from the payload**, **monotonic version numbers +
  expiration timestamps to defeat rollback/freeze attacks**, and **root-key
  pinning with a signed rotation path.** We take the *ideas* (version counter,
  expiry, signed rotation) without the full four-role (root/targets/snapshot/
  timestamp) delegation tree — overkill for one org signing one manifest.
- **Sigstore / COSE / JWS / x509 chains.** Full PKI transparency-log signing.
  Powerful but pulls in a dependency surface (Fulcio/Rekor, x509 chain
  validation) that violates the zero-dependency, stdlib-only constraint. **AVOID**
  the infrastructure; **BORROW** only the detached-signature-over-canonicalized-
  bytes shape, which `crypto/ed25519` gives us directly.
- **OPA / Rego bundle API.** The operational shape we want to mirror: a service
  serves a *signed bundle*, the agent polls (`If-None-Match`/ETag), verifies the
  bundle signature against a configured key, and **on any verification failure
  keeps the last-good bundle rather than going open.** OPA's `.signatures.json`
  (bundle files hashed, the manifest signed) and its *persist last downloaded
  bundle to disk for restart/offline* behavior map straight onto W3's last-good
  cache. **BORROW** the poll-verify-or-keep-last-good loop and the on-disk
  persistence; that is the W3 design almost verbatim.

### 1c. Device identity — SPIFFE/SPIRE, mTLS device certs, hardware attestation

- **SPIFFE/SPIRE SVIDs.** A workload gets a cryptographic identity (SVID, an
  x509 or JWT doc) issued after node+workload attestation, and services
  authenticate each other with it. The load-bearing idea for us is **attested
  identity as the target selector**: a central grant is addressed to a device
  identity the device can *prove*, not a self-asserted string. This is what makes
  "device spoofing another device's tier" (threat T7) hard.
- **mTLS device certs.** Simpler instance of the same: the device presents a cert
  to the org endpoint; the endpoint scopes what it returns to that cert. Good fit
  for W7's audit sink and W3's pull if we want the *endpoint* to authenticate the
  *device*.
- **Hardware attestation (TPM, Secure Enclave, remote attestation).** The
  strongest device-binding: the identity key is sealed in hardware and unspoofable
  even by a compromised host. **AVOID as a requirement** — it is platform-specific
  and breaks the "one static binary, runs anywhere, un-enrolled == today"
  property. **BORROW as an optional hardening tier** for W5: a device keypair
  generated at enroll and stored in the OS keystore where available, with a plain
  on-disk fallback; the *protocol* treats it as opaque so a later TPM-backed key
  drops in without a schema change.

### 1d. The fak-native column — `internal/policy/amendment.go` (#5170)

`fak` already has the piece the others bolt on after the fact: **a machine-checked
who-may-amend-what registry.** MDM/OPA/TUF all distribute policy but none of them
carries an intrinsic, per-knob, direction-typed model of *which knobs a
distribution channel is even allowed to move.* `amendment.go` does — and a
reflection conformance test fails the build if a knob lands unclassified. So the
`central` channel does not need a new enforcement mechanism: it needs a
**channel constant** (`ChannelCentral`, W1) added to the closed set, and each
`PolicyKnob` decides whether `central` is authorized. The existing
most-restrictive fold then honors a `central` RATCHET for free, and the FROZEN
knobs (egress SSRF floor, reversibility gate, structural danger arg-rules,
write-class self-modify rungs, `AdvisoryEligible` clamp) remain unmovable by
construction — no channel, `central` included, may list a FROZEN knob.

---

## 2. BORROW / AVOID table

| Target | BORROW (adopt) | AVOID (do not adopt) | Lands in |
|---|---|---|---|
| **MDM/Intune/Jamf** | Enroll-time device identity + server pin; group/device/user targeting; signed profile before apply; compliance reporting back to IT | Server-push transport (APNs), heavyweight CA/PKI lifecycle, fail-toward-last-managed-state on removal | W5, W6, W7 |
| **TUF** | Monotonic `version` counter + `expires` to defeat rollback/freeze; signed metadata separate from payload; signed key-rotation path | Full four-role delegation tree; separate snapshot/timestamp roles | W2, R2 |
| **Sigstore / COSE / JWS / x509** | Detached-signature-over-canonicalized-bytes shape | Transparency-log infra (Fulcio/Rekor), x509 chain validation, any non-stdlib crypto dependency | W2 |
| **OPA/Rego bundle API** | Poll → verify → keep-last-good loop; on-disk persistence of last verified bundle; refuse-open on verify failure | Rego evaluation engine; server-driven bundle discovery beyond a single URL | W3 |
| **SPIFFE/SPIRE** | Attested, provable device identity as the *target selector* for grants | Full SPIRE server/agent node-attestation deployment; SVID rotation infra | W5, W6 |
| **mTLS device certs** | Device authenticates to org endpoint; endpoint scopes response to that identity | Requiring a corporate CA to exist before `fak` works | W3, W7 |
| **Hardware attestation (TPM)** | Optional keystore-backed device key with opaque-key protocol so TPM drops in later | Making hardware attestation a *requirement* (platform lock-in, breaks un-enrolled==today) | W5 (optional tier) |
| **fak `amendment.go` (#5170)** | Everything — `central` is a new channel constant on the existing closed set; the fold + FROZEN cap are reused unchanged | Inventing a parallel precedence model or a new floor outside the amendment registry | W1, W4 |

---

## 3. Threat model

Each threat names the concrete mitigation and the ticket(s) that must carry it.
The invariant behind every row: **a bad org input can only make the floor
stricter; it can never widen past the compiled-in FROZEN floor, and it can never
fail open.**

| # | Threat | Attack | Mitigation (mechanism) | Carried by |
|---|---|---|---|---|
| **T1** | **Rollback / replay of an old, more-permissive manifest** | Adversary (or a stale mirror) serves a previously-valid envelope whose floor was looser | Monotonic `version` counter in the envelope; local persists **highest-seen** version and **refuses any lower** (TUF anti-rollback). `not_before`/`expires` window bounds freeze attacks. A rejected envelope leaves the *current* posture, never widens. | **W2** (verifier enforces monotonicity + window), **R2** specifies the counter + persistence semantics |
| **T2** | **Org root-key compromise** | Attacker steals the org signing key and issues a widening manifest | Blast radius is still capped by the **compiled-in FROZEN floor** — a forged central grant can never move a FROZEN knob or exceed the FROZEN ceiling on a GATED_WIDEN knob. Plus a **signed key-rotation path** (new root signed by the old, pinned at enroll) so a compromised key can be retired; and short `expires` windows to bound exposure. | **W1** (FROZEN knobs never list `central`), **R2**/**W2** (rotation + short expiry), **W4** (fold caps every grant at the FROZEN ceiling) |
| **T3** | **Offline / air-gap** | The org endpoint is unreachable for an extended period | **Last-good cache** serves within a `max_staleness` bound; **past the bound → refuse-to-widen**: drop every `central` GATED_WIDEN grant and fall back to the compiled-in floor. **Never fail-open**, never fail-to-last-managed. Un-enrolled `fak` is unaffected (org plane inert). | **W3** (cache + staleness bound + fallback), **R2** specifies the bound semantics |
| **T4** | **Insider widening past the compiled-in floor** | A malicious/careless admin authors a central manifest that tries to unlock a FROZEN knob or exceed a FROZEN ceiling | The precedence fold **clamps every central grant at the compiled-in FROZEN cap**; FROZEN knobs are structurally not in `central`'s authorized channel set, so the attempt is a no-op with a journaled clamp note. A red-team amendment-matrix case proves `central` cannot move a FROZEN knob. | **W1** (channel authorization), **W4**/**W6** (clamp-at-cap + journal), **R3** (precedence truth table) |
| **T5** | **The wrapped agent reaching the central channel** | The sandboxed agent tries to fetch/forge/edit the org manifest to grant itself capability | `central` is **out-of-band, above the operator overlay** — not an agent-writable surface (agent-self stays closed, SELF_AMENDABLE is empty). The org config/enrollment/cache files are injected into `SelfModifyGlobs` (the same `protectGuardPolicyConfig` discipline that already protects the guard config), so a self-edit is a `SELF_MODIFY` refusal. | **W1** (`central` not agent-reachable), **W5** (enrollment/cache files self-protected), **W3** (cache file in SelfModifyGlobs) |
| **T6** | **Malicious / MitM policy endpoint** | An attacker intercepts the pull and serves a forged or tampered manifest | **Signature verification against the enroll-pinned org root key** — transport is untrusted; authenticity comes from the pin, not TLS. A bad signature / wrong issuer / tampered body **fails closed** to the last-good (T3 path), never applies. Optional mTLS to the endpoint hardens exfil of the audit stream but is not the trust root. | **W2** (verify against pinned key, fail-closed), **W5** (the pin), **R2** (canonicalization so the signed bytes are unambiguous) |
| **T7** | **Device spoofing another device's usage tier** | A low-tier device claims another device's identity to receive a higher entitlement grant | Grants are addressed to an **attested/provable device identity** (SPIFFE-style), established at enroll and bound to a device keypair — not a self-asserted string. The verifier checks the grant `target` against the *proven* device id; a mismatch drops the grant (refuse-to-widen). Optional keystore/TPM-backed key raises the bar. | **W5** (provable device identity), **W6** (grant target matched to attested id), **R2** (identity in the pin) |

---

## 4. Concrete recommendations

Not a menu — one recommendation per ticket, chosen to preserve
refuse-to-widen, the FROZEN cap, and the stdlib-only single-binary property.

### R2 / #5317 — signing, freshness, key distribution → **Ed25519 + monotonic version + TOFU-at-enroll pin + max-staleness bound**

- **Signature:** **Ed25519 detached signature over the canonicalized envelope**,
  `crypto/ed25519` (stdlib). Reject COSE/JWS/x509 chains and Sigstore — they add
  a dependency surface that breaks the single-static-binary constraint for no gain
  over a single org signing a single manifest. Canonicalize the envelope by
  signing over the JSON body with a fixed, sorted-key serialization (or a defined
  byte-exact `signed` sub-object) so the signed bytes are unambiguous.
- **Freshness / anti-rollback:** a **monotonic `version` counter** the local
  persists (highest-seen) and refuses to go below, **plus** `not_before`/`expires`.
  (TUF's anti-rollback + freeze-attack defense, minus the delegation tree.)
- **Key distribution:** **TOFU-at-enroll** — pin the org root public key at
  `fak enroll` time; support a **rotation envelope signed by the outgoing key**.
- **Offline:** last-good cache + a **`max_staleness` bound**; past it,
  refuse-to-widen to the compiled-in floor.

### R3 / #5318 — precedence + entitlement semantics → **strict authority lattice, central-widen is a ceiling, operator overlay is a further floor**

- **Authority lattice (most → least):** **compiled-in FROZEN floor → `central`
  org overlay → operator overlay → agent-self (closed).**
- **Per class:** `central` RATCHET is always honored via the existing
  most-restrictive fold; `central` GATED_WIDEN raises a ceiling **capped at the
  FROZEN cap**; FROZEN is untouchable; SELF_AMENDABLE n/a.
- **Device-operator rule:** a local operator may only **tighten below** a central
  grant, **never widen past** it. So a central widen is a *ceiling* and an
  operator overlay is a *further floor* under it. Answers #5318's two questions
  explicitly: *can central raise a cap the operator lowered?* — **no** (the
  operator's tighten wins under most-restrictive); *can the operator widen past a
  central grant?* — **no** (operator is strictly below central). This reuses
  #5170 Track D's precedence — **one precedence model, not two.**

### W2 / #5320 — envelope → **TUF-shaped `fak-org-policy/v1` wrapping a `fak-policy/v1` body, Ed25519-verified, fail-closed on every check**

Adopt the OPA-bundle *shape* (signed wrapper around the existing manifest) with
TUF's freshness fields. Envelope field list (implementable verbatim by W2, aligned
with #5317's DoD):

```
{ issuer, alg, sig, version, not_before, expires, min_version, target, body }
```

where `body` is a `fak-policy/v1` manifest (POLICY.md schema — unchanged). Each of
{bad signature, expired, `not_before` in future, rolled-back `version`, wrong
issuer, `min_version` too high} must **fail closed with a distinct
closed-vocabulary reason** (reuse `POLICY_BLOCK`/`TRUST_VIOLATION`/`MALFORMED`
from `internal/abi/reasons.go` — do not invent a new code family), matching the
existing fail-loud loader discipline in POLICY.md.

### W5 / #5323 — enroll → **MDM-style enroll-time pin + SPIFFE-style provable device identity, opt-in, keystore-backed where available**

- Adopt the **MDM enrollment model**: `fak enroll --org <url>` pins the org root
  public key (TOFU) and mints a **device identity**, persisted to local config;
  `--status` / `--revoke` for lifecycle; a second enroll to a different org is
  **refused unless `--revoke`-d first** (no silent re-pin).
- Adopt the **SPIFFE idea of a provable identity** as the grant target: generate a
  device keypair at enroll, store it in the OS keystore where available with a
  plain on-disk fallback, and keep the key **opaque to the protocol** so a
  TPM/Secure-Enclave-backed key drops in later without a schema change.
- **Un-enrolled `fak` stays byte-for-byte today's behavior** — the org plane is
  entirely inert without an enrollment; `--org-policy-url` with no enrollment is a
  no-op / explicit error, never a silent trust bootstrap.

---

## 5. Conclusion

Centrally-administered endpoint policy is a solved-ish problem next door, and the
mechanisms compose cleanly onto `fak`'s existing amendment-class model without a
new floor or a second precedence system: **borrow MDM's enroll-time device
identity + targeting, TUF's monotonic-version/expiry anti-rollback, the OPA
poll-verify-keep-last-good loop, and SPIFFE's provable device identity; verify a
TUF-shaped `fak-org-policy/v1` envelope with stdlib Ed25519 against an
enroll-pinned root key; and let the org plane live entirely as the new `central`
amendment channel (#5170) whose GATED_WIDEN grants are ceilings capped by the
compiled-in FROZEN floor.** The single load-bearing invariant across all seven
threats and all seven work tickets: **a stale, forged, replayed, or MitM'd org
manifest can only ever make the floor stricter — it refuses to widen and falls
back to the compiled-in floor, never fails open.**
