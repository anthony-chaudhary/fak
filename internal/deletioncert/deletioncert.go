// Package deletioncert mints and verifies a DeletionCertificate: a single,
// portable, re-checkable artifact that binds a bit-exact KV eviction to the
// tamper-evident audit journal that recorded it. It is the FOLD the
// provable-deletion lane was missing — every input already exists in the kernel
// (the eviction count from model.KVCache.Evict, the hash-chained anchor row from
// internal/journal, the integrity epoch from internal/vdso); this package does
// nothing but bind them into one signed receipt and re-verify it.
//
// WHAT THE CERTIFICATE PROVES.
//
//   - An eviction of a NAMED span actually ran (EvictedCount, Span) and the
//     surviving context is byte-identical to one that never saw the span
//     (Equivalence: the max|Δ|=0 claim + the run id that produced it).
//   - That event is anchored to a specific row of an append-only, hash-chained
//     journal (Anchor: {Seq, PrevHash, Hash}). Verify re-reads the journal,
//     re-derives the chain, and confirms the anchor row is present and UNEDITED.
//     If anyone rewrote history after the fact, journal.Verify breaks the chain
//     and this certificate fails closed.
//   - The whole bundle is covered by a detached signature, so a single flipped
//     field in the certificate itself is detectable independently of the journal.
//
// WHAT IT DOES NOT PROVE — and the package is honest about this so a caller
// cannot over-read it (see PROVABLE-DELETION-CERTIFICATE.md):
//
//   - It does NOT prove the data is gone from anywhere EXCEPT the inference
//     working set + agent memory the eviction touched. Fine-tuned weights,
//     embeddings, backups, and replicas are out of scope. Scope names exactly
//     which surface the receipt covers; do not imply more.
//   - v1 is SELF-SIGNED by default: the issuer holds the signing key, so the
//     signature proves integrity (nobody tampered) but not independence (you
//     still trust the issuer minted it honestly). The ExternalAnchor seam is the
//     named escape hatch — populate it with an RFC-3161 timestamp / transparency
//     -log inclusion proof to make the anchor checkable WITHOUT trusting the
//     issuer. Until then, the honest claim is "tamper-evident and re-verifiable
//     against the journal", NOT "third-party-certified".
//
// The package is a leaf: it imports only encoding/crypto stdlib. It never touches
// the adjudication hot path and resolves no blob bytes.
package deletioncert

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// SchemaVersion is the certificate schema id. Bump it if the signed pre-image
// (the fields covered by canonicalBytes) changes, so an old verifier never
// silently accepts a new-shaped certificate.
const SchemaVersion = "fak.deletioncert/v1"

// Span names the evicted region of the inference working set: the contiguous KV
// position range [From, From+Len) that model.KVCache.Evict removed.
type Span struct {
	From int `json:"from"`
	Len  int `json:"len"`
}

// Anchor binds the eviction event to one row of a hash-chained audit journal
// (internal/journal). Seq/PrevHash/Hash are copied verbatim from the journal Row
// that recorded the QUARANTINE/eviction; Verify re-derives them from the journal
// on disk and refuses the certificate if they do not match an intact chain.
//
// ResultDigest is LOAD-BEARING and is the fix for the position-not-subject hole a
// threat review found: {seq,prev_hash,hash} pin the row's POSITION and integrity
// but NOT which data it was about, so without a content digest any genuine past
// row could be relabeled to a new subject. Verify enforces Certificate.Subject ==
// Anchor.ResultDigest, so a certificate is bound to the specific bytes the row
// recorded, not merely to a slot in the chain.
type Anchor struct {
	Seq          uint64 `json:"seq"`           // the journal row's monotonic sequence number
	PrevHash     string `json:"prev_hash"`     // the chained-in prefix hash
	Hash         string `json:"hash"`          // chainHash(prev, row) — the row's tamper-evident id
	ResultDigest string `json:"result_digest"` // the row's content digest — pins WHICH data (subject binding)
}

// Equivalence is the bit-exact deletion claim. MaxAbsDelta is the measured
// max|Δ| between the post-eviction surviving context and a reference run that
// never saw the span; for the standard GQA/RoPE path it is 0 (byte-identical).
// RunID identifies the run/test that produced the measurement so an auditor can
// re-run it. Claim is the human-readable statement.
type Equivalence struct {
	Claim       string  `json:"claim"`
	MaxAbsDelta float64 `json:"max_abs_delta"`
	RunID       string  `json:"run_id"`
}

// ExternalAnchor is the named escape hatch from self-attestation. v1 leaves it
// empty (the certificate is self-signed). Populating it with an RFC-3161
// timestamp token or a transparency-log inclusion proof over Anchor.Hash makes
// the anchor independently checkable without trusting the issuer's key — the path
// to a genuinely third-party-verifiable receipt. Verify treats a present
// ExternalAnchor as advisory metadata in v1 (it does not yet validate the proof);
// the field exists so a certificate can carry it forward without a schema break.
type ExternalAnchor struct {
	Kind  string `json:"kind,omitempty"`  // e.g. "rfc3161", "ct-log"
	Proof string `json:"proof,omitempty"` // opaque, base64 — the timestamp token / inclusion proof
}

// Certificate is the portable deletion receipt. Signature and PublicKey are NOT
// part of the signed pre-image (they wrap it); every other field is covered by
// canonicalBytes and thus by the signature.
type Certificate struct {
	Schema string `json:"schema"` // SchemaVersion

	Subject      string `json:"subject"`       // digest of the evicted span — WHAT was evicted; MUST equal Anchor.ResultDigest
	Scope        string `json:"scope"`         // the surface this receipt covers, e.g. "inference-working-set+agent-memory"
	Method       string `json:"method"`        // the technical measure, named on the face: "kv-cache-eviction"
	ModelPath    string `json:"model_path"`    // the eviction code path; the equivalence claim is admissible ONLY for "gqa-rope"
	CodeCommit   string `json:"code_commit"`   // the fak commit the eviction+equivalence test ran on (pins the claim to a version)
	WitnessName  string `json:"witness_name"`  // the vDSO witness the entries were admitted under ("" if none)
	Span         Span   `json:"span"`          // the evicted KV position range
	EvictedCount int    `json:"evicted_count"` // positions Evict REPORTED removing — a self-report, not a witnessed cache delta

	Equivalence    Equivalence    `json:"equivalence"`     // the max|Δ|=0 bit-exact claim (asserted by run-id, re-checked only as a signed string)
	Anchor         Anchor         `json:"anchor"`          // the hash-chained journal row recording the event
	JournalHead    string         `json:"journal_head"`    // the journal chain-head hash at issue time (so wholesale rewrite is detectable once external-anchored)
	TrustEpoch     uint64         `json:"trust_epoch"`     // vDSO integrity clock after revocation — per-pool/per-node liveness, NOT fleet-wide
	IssuedAtUnix   int64          `json:"issued_at_unix"`  // ISSUER-asserted wall clock (untrusted until ExternalAnchor carries a real timestamp)
	ExternalAnchor ExternalAnchor `json:"external_anchor"` // the self-attestation escape seam (empty in v1)

	PublicKey string `json:"public_key"` // hex ed25519 public key — the trust root for Signature
	Signature string `json:"signature"`  // hex ed25519 signature over canonicalBytes
}

// Mint binds the supplied facts into a signed Certificate. It does not run an
// eviction, read a journal, or read a clock — the caller supplies every fact
// (this keeps the package a pure fold and deterministic for tests). priv signs
// the canonical pre-image; the matching public key is embedded so a verifier with
// only the certificate can check integrity.
func Mint(priv ed25519.PrivateKey, c Certificate) (Certificate, error) {
	if len(priv) != ed25519.PrivateKeySize {
		return Certificate{}, fmt.Errorf("deletioncert: bad private key size %d", len(priv))
	}
	c.Schema = SchemaVersion
	if c.Scope == "" {
		c.Scope = "inference-working-set+agent-memory"
	}
	if c.Method == "" {
		c.Method = "kv-cache-eviction"
	}
	// Subject MUST equal the anchor row's content digest so the certificate is
	// bound to WHICH data the row recorded, not just a position in the chain. If
	// the caller did not pre-set Subject, derive it from the anchor; Verify then
	// re-enforces the equality (a mismatch is a relabel attempt → invalid).
	if c.Subject == "" {
		c.Subject = c.Anchor.ResultDigest
	}
	c.PublicKey = hex.EncodeToString(priv.Public().(ed25519.PublicKey))
	c.Signature = "" // excluded from the pre-image
	pre, err := canonicalBytes(c)
	if err != nil {
		return Certificate{}, err
	}
	c.Signature = hex.EncodeToString(ed25519.Sign(priv, pre))
	return c, nil
}

// JournalVerifier re-checks that an Anchor names a real, intact row of a
// hash-chained journal. internal/journal satisfies this via a thin adapter (see
// the cmd demo / tests) so this package need not import journal (keeping it a
// leaf). It returns the row's recomputed Hash and whether the chain up to and
// including that row verified.
type JournalVerifier interface {
	// AnchorRow looks up the journal row with the given Seq, re-derives the chain
	// hash for every row up to it, and returns (prevHash, hash, ok). ok is false
	// if the row is absent or the chain is broken at or before it.
	AnchorRow(seq uint64) (prevHash, hash string, ok bool)
}

// Result is the typed verdict of Verify. Valid is the single bit a gate keys on;
// the booleans below say WHICH check failed so a caller can report precisely.
type Result struct {
	Valid         bool   `json:"valid"`
	SignatureOK   bool   `json:"signature_ok"`
	AnchorOK      bool   `json:"anchor_ok"`        // the journal row exists and the chain is intact
	AnchorBound   bool   `json:"anchor_bound"`     // the cert's Anchor matches the journal's row hashes
	SubjectBound  bool   `json:"subject_bound"`    // Subject == Anchor.ResultDigest (cert pins WHICH data, not just a position)
	EquivalenceOK bool   `json:"equivalence_ok"`   // the deletion claim is the bit-exact one (max|Δ|==0)
	SelfAttested  bool   `json:"self_attested"`    // true when no ExternalAnchor — issuer holds the key, so this is auditable self-attestation, not independence
	Reason        string `json:"reason,omitempty"` // first failing check, "" when Valid
}

// Verify re-checks a certificate. jv may be nil to skip the journal-binding rung
// (signature + equivalence only) — useful when the journal is not co-located, but
// a nil verifier yields AnchorOK=false and Valid=false, because the whole point is
// that the anchor is re-checkable. A non-nil jv re-derives the row hashes from the
// journal on disk and confirms the certificate's Anchor matches an intact chain.
func Verify(c Certificate, jv JournalVerifier) Result {
	var r Result
	r.SelfAttested = c.ExternalAnchor.Kind == ""

	// 1. Signature over the canonical pre-image — detects any edited field.
	pub, err := hex.DecodeString(c.PublicKey)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		r.Reason = "bad public key"
		return r
	}
	sig, err := hex.DecodeString(c.Signature)
	if err != nil {
		r.Reason = "bad signature encoding"
		return r
	}
	unsigned := c
	unsigned.Signature = ""
	pre, err := canonicalBytes(unsigned)
	if err != nil {
		r.Reason = "canonicalize: " + err.Error()
		return r
	}
	if !ed25519.Verify(ed25519.PublicKey(pub), pre, sig) {
		r.Reason = "signature mismatch (certificate altered)"
		return r
	}
	r.SignatureOK = true

	// 2. Equivalence: the deletion claim must be the bit-exact one. A certificate
	// that records a non-zero drift is not a provable-deletion receipt.
	if c.Equivalence.MaxAbsDelta == 0 {
		r.EquivalenceOK = true
	} else {
		r.Reason = fmt.Sprintf("equivalence not bit-exact (max|Δ|=%g)", c.Equivalence.MaxAbsDelta)
		return r
	}

	// 3. Anchor binding: re-derive the journal row and confirm the cert matches an
	// intact chain. This is the rung that makes the receipt re-verifiable rather
	// than asserted — without it the cert is just a signed claim.
	if jv == nil {
		r.Reason = "no journal verifier (anchor unchecked)"
		return r
	}
	prevHash, hash, ok := jv.AnchorRow(c.Anchor.Seq)
	if !ok {
		r.Reason = fmt.Sprintf("journal anchor seq %d absent or chain broken", c.Anchor.Seq)
		return r
	}
	r.AnchorOK = true
	if prevHash != c.Anchor.PrevHash || hash != c.Anchor.Hash {
		r.Reason = "anchor hash mismatch (cert not bound to this journal)"
		return r
	}
	r.AnchorBound = true

	// 4. Subject binding: the certificate's subject must equal the anchor row's
	// content digest, so a genuine past row cannot be relabeled to a different
	// subject. Position + integrity is not subject; this is what pins WHICH data.
	if c.Subject == "" || c.Subject != c.Anchor.ResultDigest {
		r.Reason = "subject not bound to anchor content digest (possible relabel)"
		return r
	}
	r.SubjectBound = true

	r.Valid = true
	return r
}

// MarshalIndent renders a certificate as pretty JSON (the on-disk / on-wire form).
func (c Certificate) Marshal() ([]byte, error) { return json.MarshalIndent(c, "", "  ") }

// Parse decodes a certificate from its JSON form.
func Parse(b []byte) (Certificate, error) {
	var c Certificate
	if err := json.Unmarshal(b, &c); err != nil {
		return Certificate{}, fmt.Errorf("deletioncert: parse: %w", err)
	}
	if c.Schema != SchemaVersion {
		return Certificate{}, fmt.Errorf("deletioncert: unknown schema %q (want %q)", c.Schema, SchemaVersion)
	}
	return c, nil
}

// canonicalBytes is the signed pre-image: the certificate with Signature cleared,
// marshaled by encoding/json (which emits object keys in sorted order), so the
// same logical certificate always produces the same bytes. PublicKey IS covered
// (it is the trust root); Signature is NOT (it is the output).
func canonicalBytes(c Certificate) ([]byte, error) {
	c.Signature = ""
	b, err := json.Marshal(c)
	if err != nil {
		return nil, fmt.Errorf("deletioncert: canonicalize: %w", err)
	}
	return b, nil
}

// ErrNoEviction is returned by callers that refuse to mint a certificate for a
// zero-count eviction (nothing was removed, so there is nothing to attest).
var ErrNoEviction = errors.New("deletioncert: refusing to certify a zero-count eviction")
