# RESEARCH — org-policy signing, freshness, and key-distribution model

- **Date:** 2026-07-20
- **Issue:** #5317 (research spike) · **Epic:** #5315 (Org Policy Plane) · **Parent:** #5170 (Policy Amendment Classes)
- **Lane:** research (decision spike) · **Class:** research
- **Downstream implementers:** W2 #5320 (envelope + verifier), W3 #5321 (remote pull + cache + offline fallback)
- **Status:** RECOMMENDED — pick this scheme before W2 implements it.

## TL;DR — the one recommended scheme

**Ed25519 detached signature (`crypto/ed25519`) over a length-prefixed canonical
concatenation of the envelope's signed fields, carrying a monotonic `version`
counter plus `not_before`/`expires`; the org ROOT public key is pinned TOFU at
`fak enroll`, rotated only through a root-signed key-rotation envelope; a local
`fak` persists the highest-seen `version` and refuses any lower, caches the
last-good manifest, and past a compiled-in max-staleness bound refuses-to-widen
back to the compiled-in FROZEN floor — never fail-open.**

Everything below is stdlib-only, so it preserves fak's one-static-binary, zero
external dependency invariant (`go.mod`: *"Zero external dependencies (standard
library only) — there is no go.sum."*; AGENTS.md: the Go module is the whole
repository root).

## Why this shape (grounding in the existing floor)

The envelope wraps — it does not replace — the existing `fak-policy/v1` body
(`POLICY.md`, `internal/policy/policy.go`, `const Version = "fak-policy/v1"`).
The inner body is exactly today's manifest schema (`policy.Manifest`,
`ParseManifest` with `DisallowUnknownFields`, closed deny vocabulary in
`internal/abi/reasons.go`). The `central` channel sits, per epic #5315's
precedence lattice, **above** the operator overlay and **below** the compiled-in
FROZEN floor:

> compiled-in FROZEN floor → central org overlay → operator overlay → agent-self (closed)

So the verifier's job is narrow: prove the bytes are (a) authentic (org-issued),
(b) fresh (not a rolled-back older/more-permissive manifest), and (c) usable
offline without failing open — then hand the inner `fak-policy/v1` body to the
already-existing `policy.ParseManifest` / `ToRuntime` path unchanged.

## (a) Recommendation + REJECTED-ALTERNATIVES table

### Signature algorithm — recommend Ed25519 detached signature

| Option | stdlib-native? | Verdict | Why |
|---|---|---|---|
| **Ed25519 detached sig (`crypto/ed25519`)** | **Yes** — `crypto/ed25519`, in the standard library since Go 1.13, no `go.sum` entry | **RECOMMENDED** | 32-byte public key, 64-byte signature, deterministic, no parameter/curve negotiation, no parser attack surface. `ed25519.Verify(pub, msg, sig)` is a single total function. Matches the "one static binary, stdlib-only" invariant exactly. |
| x509 certificate chain (`crypto/x509`) | Partly (`crypto/x509` is stdlib) | REJECTED | Pulls in chain-building, name constraints, EKU, revocation (CRL/OCSP), validity-window and path-length semantics — a large parser/policy surface fak does not need for a single org→device trust edge. TOFU root-pinning gives the same "one trust anchor" property with none of the PKI machinery. |
| COSE / JWS | **No** | REJECTED | Both require a third-party library (no stdlib COSE; JWS needs a JOSE dep) — a direct violation of the zero-dependency invariant. JWS additionally invites the alg-confusion / `alg:none` class of bugs. The compact-serialization framing buys nothing over a fixed field list we control. |
| RSA-PSS / ECDSA-P256 (`crypto/rsa`, `crypto/ecdsa`) | Yes | REJECTED (fallback-only) | Both are stdlib and would work, but ECDSA needs a nonce/RNG at *sign* time (deterministic RFC-6979 is not in stdlib) and RSA keys/sigs are an order of magnitude larger. Ed25519 is deterministic, smaller, and simpler; no reason to prefer either. |

### Canonicalization — recommend length-prefixed field concatenation (NOT re-serialized JSON)

| Option | Verdict | Why |
|---|---|---|
| **Length-prefixed concat of the signed fields in a FIXED order** (`domain-tag ‖ len(f)‖f` for each field) | **RECOMMENDED** | Signer and verifier agree a fixed field order; each field is emitted as an 8-byte big-endian length followed by its bytes, prefixed by a constant domain-separation tag (`"fak-org-policy/v1\n"`). Unambiguous (no field-boundary confusion), trivial in stdlib (`encoding/binary`), and independent of any JSON whitespace/key-order/number-format quirk. |
| Sign the raw received JSON bytes as-transmitted | REJECTED | Requires the transport to preserve exact bytes and the local to *not* re-parse-and-re-emit before verify; fragile across proxies and pretty-printers, and couples the signature to an encoder we do not control. |
| Canonical-JSON (RFC 8785 JCS) re-serialization | REJECTED | No JCS canonicalizer in the Go stdlib; hand-rolling one (sorted keys, number canonicalization, UTF-8 escaping rules) is exactly the parser surface we are trying to avoid. The concat scheme sidesteps JSON canonicalization entirely. |

Canonical rule for W2 (#5320), stated precisely:

```
signed_bytes =
    "fak-org-policy/v1\n"                     // constant domain-separation tag
  ‖ u64be(len(issuer))   ‖ issuer
  ‖ u64be(len(alg))      ‖ alg              // "ed25519"
  ‖ u64be(len(target))   ‖ target           // canonical selector string
  ‖ u64be(8)             ‖ u64be(version)    // monotonic counter, fixed width
  ‖ u64be(8)             ‖ i64be(not_before) // unix seconds, fixed width
  ‖ u64be(8)             ‖ i64be(expires)    // unix seconds, fixed width
  ‖ u64be(8)             ‖ u64be(min_version)
  ‖ u64be(len(body))     ‖ body              // the raw fak-policy/v1 body bytes
```

`sig` and `issuer`-key-id material are NOT part of `signed_bytes` (a signature
never signs itself). `body` is signed as the exact received bytes and is handed
verbatim to `policy.ParseManifest` only *after* signature+freshness verify pass.
`u64be` = 8-byte big-endian unsigned; `i64be` = 8-byte big-endian two's-complement.

## (b) Exact stdlib primitives (no third-party imports)

| Purpose | stdlib primitive |
|---|---|
| Signature verify | `crypto/ed25519` — `ed25519.Verify(pub ed25519.PublicKey, msg, sig []byte) bool`; `ed25519.PublicKey` is a `[32]byte`, `sig` is 64 bytes |
| Key/sig transport encoding | `encoding/base64` (`base64.StdEncoding`) or `encoding/hex` for the pinned-key file and the `sig`/`issuer_key` envelope fields |
| Envelope decode | `encoding/json` with `json.NewDecoder(r).DisallowUnknownFields()` — the same fail-loud discipline `policy.ParseManifest` already uses |
| Canonical byte assembly | `encoding/binary` (`binary.BigEndian.PutUint64`) + `bytes.Buffer` |
| Freshness clock | `time` — `time.Now().Unix()` compared against `not_before`/`expires`; max-staleness as a `time.Duration` |
| Highest-seen version + last-good cache persistence | `os` + `encoding/json` to a file under the enroll state dir (atomic write: temp file + `os.Rename`) |
| Constant-time compare (key-id / digest equality) | `crypto/subtle` (`subtle.ConstantTimeCompare`) |

Note: Ed25519's `Verify` is already constant-time and total; `crypto/subtle` is
only for comparing pinned key-ids / digests, not for the signature check itself.

## (c) Envelope field list — implementable verbatim by W2 (#5320)

`fak-org-policy/v1` envelope. All fields REQUIRED unless marked optional. Unknown
fields REJECTED at decode (`DisallowUnknownFields`), matching the inner manifest's
fail-loud contract.

| Field | JSON key | Type | Meaning / constraint |
|---|---|---|---|
| Schema tag | `version_tag` | string | Must be `fak-org-policy/v1` (envelope schema tag; a different MAJOR is refused, a newer v1 minor forward-accepted — mirrors `policy.validateVersion`). Distinct from the numeric `version` counter below. |
| Issuer | `issuer` | string | Org identity that signed this envelope (e.g. `org:acme`). Bound into `signed_bytes`; must equal the issuer recorded at enroll. |
| Algorithm | `alg` | string | Must be the exact literal `ed25519`. Any other value → refuse (no negotiation, closes the alg-confusion hole). Bound into `signed_bytes`. |
| Signature | `sig` | string (base64) | 64-byte Ed25519 detached signature over `signed_bytes`. NOT part of `signed_bytes`. |
| Issuer key id | `issuer_key` | string (base64/hex, optional) | Identifies WHICH pinned root/rotation key verifies this envelope; lets the local pick the right pinned key during rotation overlap. If absent, the current pinned root is used. |
| Version counter | `version` | uint64 | Monotonic anti-rollback counter. The local persists the highest value ever accepted and refuses any envelope whose `version` is **strictly less** (see anti-rollback semantics). Bound into `signed_bytes`. |
| Not before | `not_before` | int64 (unix seconds) | Envelope is invalid before this instant. Bound into `signed_bytes`. |
| Expires | `expires` | int64 (unix seconds) | Envelope is invalid at/after this instant. Bound into `signed_bytes`. Must satisfy `not_before < expires`. |
| Minimum binary version | `min_version` | uint64 | The lowest `fak` build the org will accept for this manifest; a binary older than `min_version` refuses-to-widen and stays on its compiled-in floor (org can force a fleet upgrade before a widening lands). Bound into `signed_bytes`. |
| Target selector | `target` | string | Canonical device/user/group selector this grant applies to (e.g. `device:*`, `group:eng`, `user:jane@acme`). Bound into `signed_bytes`. A local whose enrolled identity does not match `target` ignores the widening (still may honor a fleet-wide RATCHET). |
| Body | `body` | object (raw `fak-policy/v1`) | The inner manifest — exactly today's `policy.Manifest` schema (`POLICY.md`). Signed as its exact received bytes; handed verbatim to `policy.ParseManifest` → `ToRuntime` only AFTER verify passes. Its own inner `version` string field stays `fak-policy/v1`. |

Key-rotation is the SAME envelope with a `body` that carries the new root public
key(s) instead of (or alongside) a policy manifest — see rotation semantics below —
so there is exactly one signed object type to implement and verify.

## (d) Anti-rollback + offline-max-staleness semantics (W2/W3 test assertions)

Stated as checkable predicates so #5320 / #5321 can turn each line into an assertion.

### Verify order (fail-closed at every step)

An envelope is ACCEPTED-AND-MAY-WIDEN iff ALL of the following hold, checked in order;
the first failure → `refuse-to-widen` (keep last-good if still fresh, else fall to the
compiled-in FROZEN floor). Never fail-open at any step.

1. `version_tag == "fak-org-policy/v1"` (or a forward-accepted `fak-org-policy/v1.x`).
2. `alg == "ed25519"` exactly.
3. `issuer` equals the issuer pinned at enroll.
4. `ed25519.Verify(pinnedKey(issuer_key), signed_bytes, sig) == true`, where
   `signed_bytes` is reconstructed by the canonical rule in (a).
5. `not_before <= now < expires` (clock from `time.Now().Unix()`).
6. `fak_build_version >= min_version`.
7. `version >= persisted_highest_seen_version` (anti-rollback; see below).
8. enrolled identity matches `target` (else: envelope is not for this device — ignore its widening, but a fleet-wide RATCHET body still folds via the most-restrictive fold).

### Anti-rollback (monotonic version)

- The local persists `highest_seen_version` = the max `version` of any envelope it has
  **ever accepted** (step 7 passed AND signature valid). Persisted atomically (temp +
  `os.Rename`) in the enroll state dir.
- **REFUSE** any envelope with `version < highest_seen_version` — even if its signature,
  `issuer`, and time window are all valid. This is the rollback defense: a replayed older,
  more-permissive manifest is rejected. Assertion: given accepted v5, an otherwise-valid v4
  envelope → refuse-to-widen, posture unchanged.
- `version == highest_seen_version` is **idempotent-accept** (same manifest re-fetched):
  it re-establishes last-good freshness (updates the cache timestamp) but changes no knob.
- `highest_seen_version` only ever increases; it is never reset by an incoming envelope
  (only a root-signed rotation/enroll-reset may re-baseline it, and only upward).
- A failed signature/time/target check does **not** advance `highest_seen_version`
  (only a fully accepted envelope advances it), so a bad envelope cannot poison the counter.

### Offline / max-staleness (last-good cache, refuse-to-widen)

- On every successful accept, cache `{envelope, received_at}` as **last-good** (atomic write).
- `MaxStaleness` is a **compiled-in** duration (a FROZEN knob — the org manifest cannot
  extend its own tolerated staleness; that would let a captured old manifest widen forever).
- On startup / refresh when the endpoint is unreachable:
  - If `now - last_good.received_at <= MaxStaleness` **AND** `now < last_good.expires`:
    honor the cached central overlay (still fresh enough offline).
  - Else (past the staleness bound OR the cached envelope has expired):
    **refuse-to-widen** — discard the central overlay's *widening* effect and fall back to the
    compiled-in FROZEN floor. Any RATCHET (tightening) the local last saw MAY be retained
    (tightening never violates the floor), but no widened ceiling survives past the bound.
- The FROZEN floor is the hard cap in all cases: `expires` past, staleness exceeded, bad
  signature, wrong `target`, or un-enrolled → the resolved posture is **at most** the
  compiled-in floor. There is no code path where a missing/stale/bad central manifest yields
  a MORE permissive posture than the compiled-in floor. Assertion: with the endpoint down and
  `now - received_at > MaxStaleness`, a tool the central overlay had unlocked is DENIED again
  (back to `DEFAULT_DENY` / floor).
- Un-enrolled `fak` never consults any of this — byte-for-byte today's behavior (epic invariant).

### Key distribution + rotation (TOFU root pin)

- `fak enroll --org <url>` fetches and **pins** the org ROOT Ed25519 public key
  (trust-on-first-use) plus a device identity, persisted in the enroll state dir. All
  subsequent envelopes verify against this pinned key (or a rotation key it introduces).
- **Rotation path (root-signed):** a key-rotation is delivered as the SAME
  `fak-org-policy/v1` envelope, signed by the CURRENT pinned root, whose `body` announces the
  NEW root public key (and an overlap window). The local verifies the rotation envelope under
  the old pinned key, then adds/promotes the new key to the pinned set. New key rides in under
  the same anti-rollback `version` monotonicity, so a replayed *old* rotation cannot demote the
  key set. This gives forward key mobility without ever trusting an unauthenticated key.
- No out-of-band CA, no CRL/OCSP: the single pinned root (rotated only by itself) is the whole
  trust anchor — the property x509 chains would have given us, minus the PKI surface.

## Open follow-ons (not blocking the recommendation)

- Multi-root / threshold signing (M-of-N org keys) is out of scope for W2; the `issuer_key`
  field and the pinned-key *set* leave room for it without an envelope schema change.
- The exact on-disk layout of the enroll state dir (paths, file perms) is W5's (#5315 W5)
  concern; this note fixes only the fields that must be persisted (`highest_seen_version`,
  pinned key set, last-good cache + `received_at`).
