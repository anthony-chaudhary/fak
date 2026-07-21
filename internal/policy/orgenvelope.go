// orgenvelope.go implements the `fak-org-policy/v1` signed envelope and its
// offline verifier (W2 of epic #5315, issue #5320) — the primitive a locally
// enrolled `fak` uses to prove a pulled ORG manifest is (a) authentic (issued by
// the org root key), (b) fresh (inside its not_before/expires window and not a
// rolled-back older version), and (c) applicable to the running binary
// (min_version) BEFORE that manifest may change the local floor.
//
// The envelope wraps a `fak-policy/v1` Manifest body — it does NOT fork the
// manifest parse path: the body is decoded and validated through the exact same
// ParseManifest + ToRuntime discipline the local loader uses, so an org-pushed
// floor is held to the same closed refusal vocabulary and fail-loud unknown-field
// rules as an on-disk one.
//
// Every verification failure returns a distinct closed-vocabulary refusal from
// internal/abi (never free text), grouped by the nature of the failure:
//   - MALFORMED       — structurally broken envelope or body (bad JSON, unknown
//                        field, wrong alg, un-decodable signature, invalid inner
//                        manifest, inverted window).
//   - OVERSIZE        — the envelope exceeds the byte budget.
//   - TRUST_VIOLATION — authenticity failure (no root key, wrong signature, wrong
//                        issuer): the org cannot be proven to have issued it.
//   - UNWITNESSED     — freshness failure (before not_before, past expires, or a
//                        rolled-back version): it cannot be witnessed as current.
//   - POLICY_BLOCK    — the envelope's own min_version rule blocks this binary
//                        (the running binary is too old to apply the policy).
//
// The verifier is a PURE function of its inputs: the caller passes `now`, the
// highest version it has ever accepted (anti-rollback), and the running binary
// version, rather than the verifier reading the clock or build identity itself.
// This keeps it deterministic and table-testable. Stdlib-only: crypto/ed25519.
package policy

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OrgEnvelopeVersion is the envelope schema tag.
const OrgEnvelopeVersion = "fak-org-policy/v1"

// AlgEd25519 is the only signature algorithm this verifier speaks. A single
// static binary with no crypto dependencies favors stdlib crypto/ed25519 over an
// x509 chain or COSE/JWS (issue #5317's R2 decision).
const AlgEd25519 = "ed25519"

// DefaultMaxOrgEnvelopeBytes bounds the transmitted envelope so a hostile or
// corrupt endpoint cannot force unbounded parse work. Past it the envelope is
// refused OVERSIZE before it is decoded.
const DefaultMaxOrgEnvelopeBytes = 1 << 20 // 1 MiB

// OrgEnvelope is the `fak-org-policy/v1` signed wrapper around a `fak-policy/v1`
// manifest Body. Field semantics:
//   - Issuer: the org identity that signed this envelope (checked against a
//     caller-supplied expected issuer when set).
//   - Alg: the signature algorithm; must be AlgEd25519.
//   - Sig: base64 (std) Ed25519 detached signature over the CANONICAL bytes of
//     the envelope with Sig excluded (see canonicalBytes).
//   - Version: a monotonic anti-rollback counter. Local `fak` persists the
//     highest version it has accepted and refuses any lower one — so a replayed
//     older, more-permissive manifest cannot widen the floor.
//   - NotBefore / Expires: the validity window as Unix seconds. Outside it the
//     envelope refuses to widen (UNWITNESSED), never fails open.
//   - MinVersion: the minimum running `fak` version that may APPLY this envelope
//     (dotted numeric, e.g. "0.9.0"). A binary older than it is blocked.
//   - Target: an opaque device/user/group selector the central plane targets
//     grants with. It is carried and signed but not interpreted here.
//   - Body: the wrapped `fak-policy/v1` manifest, validated through the local
//     manifest parse path (never a fork).
type OrgEnvelope struct {
	Issuer     string          `json:"issuer"`
	Alg        string          `json:"alg"`
	Sig        string          `json:"sig"`
	Version    uint64          `json:"version"`
	NotBefore  int64           `json:"not_before"`
	Expires    int64           `json:"expires"`
	MinVersion string          `json:"min_version,omitempty"`
	Target     string          `json:"target,omitempty"`
	Body       json.RawMessage `json:"body"`
}

// canonicalEnvelope is the signature preimage layout: the envelope MINUS the
// signature, with a fixed field order (Go marshals struct fields in declaration
// order) and a Body that has been compacted to its canonical whitespace-free
// form. Signer and verifier both build this identical structure, so the signed
// bytes are reproducible on both sides without a bespoke canonical-JSON library.
type canonicalEnvelope struct {
	Issuer     string          `json:"issuer"`
	Alg        string          `json:"alg"`
	Version    uint64          `json:"version"`
	NotBefore  int64           `json:"not_before"`
	Expires    int64           `json:"expires"`
	MinVersion string          `json:"min_version"`
	Target     string          `json:"target"`
	Body       json.RawMessage `json:"body"`
}

// canonicalBytes returns the deterministic signature preimage for e: the
// envelope with Sig excluded and Body compacted. Because struct fields marshal
// in declaration order and the compacted Body is byte-identical for identical
// input bytes, both SignEnvelope and VerifyEnvelope derive the same preimage.
func canonicalBytes(e OrgEnvelope) ([]byte, error) {
	var cb bytes.Buffer
	if len(bytes.TrimSpace(e.Body)) == 0 {
		cb.WriteString("null")
	} else if err := json.Compact(&cb, e.Body); err != nil {
		return nil, err
	}
	return json.Marshal(canonicalEnvelope{
		Issuer:     e.Issuer,
		Alg:        e.Alg,
		Version:    e.Version,
		NotBefore:  e.NotBefore,
		Expires:    e.Expires,
		MinVersion: e.MinVersion,
		Target:     e.Target,
		Body:       json.RawMessage(cb.Bytes()),
	})
}

// SignEnvelope stamps e with Alg=ed25519 and an Ed25519 signature over its
// canonical bytes, returning the signed copy. It is the counterpart to
// VerifyEnvelope (both share canonicalBytes), used by a signer and by tests to
// produce a well-formed envelope.
func SignEnvelope(e OrgEnvelope, priv ed25519.PrivateKey) (OrgEnvelope, error) {
	if len(priv) != ed25519.PrivateKeySize {
		return OrgEnvelope{}, fmt.Errorf("org-policy: signing key must be %d bytes, got %d", ed25519.PrivateKeySize, len(priv))
	}
	e.Alg = AlgEd25519
	pre, err := canonicalBytes(e)
	if err != nil {
		return OrgEnvelope{}, err
	}
	e.Sig = base64.StdEncoding.EncodeToString(ed25519.Sign(priv, pre))
	return e, nil
}

// Marshal renders the signed envelope as transmittable JSON.
func (e OrgEnvelope) Marshal() ([]byte, error) { return json.Marshal(e) }

// EnvelopeError is a verification failure carrying a closed-vocabulary refusal
// Reason plus a short machine Detail token that distinguishes causes sharing a
// Reason (e.g. "expired" vs "rollback", both UNWITNESSED). It never fails open:
// the presence of an EnvelopeError means the envelope must NOT change the floor.
type EnvelopeError struct {
	Reason abi.ReasonCode
	Detail string
	err    error
}

func (e *EnvelopeError) Error() string {
	msg := fmt.Sprintf("org-policy envelope: %s (%s)", abi.ReasonName(e.Reason), e.Detail)
	if e.err != nil {
		return msg + ": " + e.err.Error()
	}
	return msg
}

func (e *EnvelopeError) Unwrap() error { return e.err }

func fail(reason abi.ReasonCode, detail string, err error) *EnvelopeError {
	return &EnvelopeError{Reason: reason, Detail: detail, err: err}
}

// VerifyOptions carries the caller-supplied trust anchor and the ambient facts
// the verifier is deliberately NOT allowed to read for itself, so the check is a
// pure function of its inputs.
type VerifyOptions struct {
	// RootPublicKey is the org root Ed25519 public key pinned at enroll time.
	RootPublicKey ed25519.PublicKey
	// ExpectedIssuer, when non-empty, must equal the envelope's Issuer.
	ExpectedIssuer string
	// Now is the wall clock to check the not_before/expires window against.
	Now time.Time
	// HighestSeenVersion is the greatest envelope Version this local ever
	// accepted; a Version below it is a rollback and is refused.
	HighestSeenVersion uint64
	// RunningVersion is the running binary's version (dotted numeric). The
	// envelope's min_version must not exceed it.
	RunningVersion string
	// MaxBytes bounds the raw envelope; 0 uses DefaultMaxOrgEnvelopeBytes.
	MaxBytes int
}

// Verified is the successful verification result: the parsed envelope plus the
// inner manifest resolved through the standard loader, so a caller can apply the
// central floor with full provenance.
type Verified struct {
	Envelope OrgEnvelope
	Manifest Manifest
	Runtime  Runtime
}

// VerifyEnvelope decodes and verifies raw `fak-org-policy/v1` bytes against the
// caller-supplied trust anchor and ambient facts. It fails CLOSED on the first
// problem, returning an *EnvelopeError whose Reason is a closed-vocabulary code.
// The check order is: size, structure, algorithm, inner-manifest validity, trust
// anchor presence, signature authenticity, issuer, freshness window,
// anti-rollback, and finally min_version — authenticity is established before any
// signed claim (time window, version) is trusted.
func VerifyEnvelope(raw []byte, opts VerifyOptions) (Verified, error) {
	max := opts.MaxBytes
	if max <= 0 {
		max = DefaultMaxOrgEnvelopeBytes
	}
	if len(raw) > max {
		return Verified{}, fail(abi.ReasonOversize, "size",
			fmt.Errorf("envelope is %d bytes, budget is %d", len(raw), max))
	}

	// Structural decode with the same fail-loud unknown-field discipline the
	// manifest loader uses: a typo'd envelope key is a hard error, not a drop.
	var e OrgEnvelope
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return Verified{}, fail(abi.ReasonMalformed, "decode", err)
	}

	if e.Alg != AlgEd25519 {
		return Verified{}, fail(abi.ReasonMalformed, "alg",
			fmt.Errorf("unsupported alg %q (want %s)", e.Alg, AlgEd25519))
	}
	if e.NotBefore != 0 && e.Expires != 0 && e.Expires < e.NotBefore {
		return Verified{}, fail(abi.ReasonMalformed, "window",
			fmt.Errorf("expires %d precedes not_before %d", e.Expires, e.NotBefore))
	}

	// Validate the inner body through the EXACT local manifest parse path — no
	// fork. ParseManifest rejects unknown fields; ToRuntime validates the closed
	// deny vocabulary, postures, patterns, and every runtime rule.
	m, err := ParseManifest(e.Body)
	if err != nil {
		return Verified{}, fail(abi.ReasonMalformed, "body", err)
	}
	rt, err := m.ToRuntime()
	if err != nil {
		return Verified{}, fail(abi.ReasonMalformed, "body", err)
	}

	// Trust anchor must be present and well-formed before authenticity can be
	// established at all.
	if len(opts.RootPublicKey) != ed25519.PublicKeySize {
		return Verified{}, fail(abi.ReasonTrustViolation, "no_root_key",
			fmt.Errorf("org root public key must be %d bytes, got %d", ed25519.PublicKeySize, len(opts.RootPublicKey)))
	}
	sig, err := base64.StdEncoding.DecodeString(e.Sig)
	if err != nil {
		return Verified{}, fail(abi.ReasonMalformed, "sig", err)
	}
	if len(sig) != ed25519.SignatureSize {
		return Verified{}, fail(abi.ReasonMalformed, "sig",
			fmt.Errorf("signature is %d bytes, want %d", len(sig), ed25519.SignatureSize))
	}
	pre, err := canonicalBytes(e)
	if err != nil {
		return Verified{}, fail(abi.ReasonMalformed, "canonical", err)
	}
	if !ed25519.Verify(opts.RootPublicKey, pre, sig) {
		return Verified{}, fail(abi.ReasonTrustViolation, "signature",
			fmt.Errorf("Ed25519 signature does not verify against the org root key"))
	}

	// Authenticity established — now the signed claims may be trusted.
	if iss := strings.TrimSpace(opts.ExpectedIssuer); iss != "" && e.Issuer != iss {
		return Verified{}, fail(abi.ReasonTrustViolation, "issuer",
			fmt.Errorf("envelope issuer %q is not the expected %q", e.Issuer, iss))
	}

	now := opts.Now.Unix()
	if e.NotBefore != 0 && now < e.NotBefore {
		return Verified{}, fail(abi.ReasonUnwitnessed, "not_before",
			fmt.Errorf("now %d is before not_before %d", now, e.NotBefore))
	}
	if e.Expires != 0 && now > e.Expires {
		return Verified{}, fail(abi.ReasonUnwitnessed, "expired",
			fmt.Errorf("now %d is past expires %d", now, e.Expires))
	}

	if e.Version < opts.HighestSeenVersion {
		return Verified{}, fail(abi.ReasonUnwitnessed, "rollback",
			fmt.Errorf("envelope version %d is below highest-seen %d", e.Version, opts.HighestSeenVersion))
	}

	if mv := strings.TrimSpace(e.MinVersion); mv != "" {
		cmp, ok := compareDotted(opts.RunningVersion, mv)
		if !ok || cmp < 0 {
			return Verified{}, fail(abi.ReasonPolicyBlock, "min_version",
				fmt.Errorf("running version %q does not satisfy min_version %q", opts.RunningVersion, mv))
		}
	}

	return Verified{Envelope: e, Manifest: m, Runtime: rt}, nil
}

// compareDotted compares two dotted-numeric versions ("0.9.0" vs "1"), returning
// -1/0/1 and ok=false if EITHER side is not a pure dotted-numeric string. A
// leading "v" is tolerated; missing trailing segments count as zero. It is
// intentionally strict (no pre-release parsing): an unparseable running version
// against a set min_version fails the gate closed, never open.
func compareDotted(a, b string) (int, bool) {
	pa, oka := parseDotted(a)
	pb, okb := parseDotted(b)
	if !oka || !okb {
		return 0, false
	}
	for i := 0; i < len(pa) || i < len(pb); i++ {
		var x, y uint64
		if i < len(pa) {
			x = pa[i]
		}
		if i < len(pb) {
			y = pb[i]
		}
		switch {
		case x < y:
			return -1, true
		case x > y:
			return 1, true
		}
	}
	return 0, true
}

func parseDotted(s string) ([]uint64, bool) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "v")
	if s == "" {
		return nil, false
	}
	parts := strings.Split(s, ".")
	out := make([]uint64, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.ParseUint(strings.TrimSpace(p), 10, 64)
		if err != nil {
			return nil, false
		}
		out = append(out, n)
	}
	return out, true
}
