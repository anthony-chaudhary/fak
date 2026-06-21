package cachemeta

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// KVManifest is the self-describing, importable record for a precomputed KV
// artifact — the "Can I Buy Your KV Cache?" / thaw / market-KV shape. It binds the
// identity axes a resident span MUST match before reuse (source digest, model,
// tokenizer, adapter, precision, position convention, producer) plus a producer-
// stated integrity checksum over the KV bytes and a detached signature over the
// binding digest. §2.4's gap: no signed/importable KV manifest existed in-tree.
//
// cachemeta never sees the KV bytes; IntegrityChecksum is producer-stated. What a
// signature covers is ManifestBindingDigest — the metadata binding, recomputable
// here — so a resident-claim checker can prove the binding was not tampered with
// without paging in terabytes of KV.
type KVManifest struct {
	SourceDigest       string // digest of the source text the KV was prefilled from
	SpanDigest         string // identity of the KV span
	Tokens             int64  // span length in positions
	ModelID            string
	TokenizerID        string
	AdapterID          string       // LoRA / adapter id ("" = base)
	Precision          string       // "fp16" | "int8" | "fp8" | ...
	PositionConvention PositionMode // how positions were encoded
	Producer           string
	ProducerKeyID      string
	IntegrityChecksum  string // producer-stated checksum over the KV bytes
	Signature          ManifestSignature
}

// ManifestSignature is a detached signature over ManifestBindingDigest. The
// algorithm is named (not mandated) so tier-1 stays crypto-scheme-free; the
// integrator performs the actual verification and reports SignatureVerified on the
// resident claim.
type ManifestSignature struct {
	Algorithm string // "ed25519" | "hmac-sha256" | "none" | ""
	Value     string // hex-encoded signature
}

// ManifestBindingDigest is the deterministic sha256 over every binding axis of a
// KVManifest. It is what a signature MUST cover, so a checker can detect tampering
// of any field a resident span is allowed to reuse against.
func ManifestBindingDigest(m KVManifest) string {
	h := sha256.New()
	writeField := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{0})
	}
	writeField(m.SourceDigest)
	writeField(m.SpanDigest)
	writeField(m.ModelID)
	writeField(m.TokenizerID)
	writeField(m.AdapterID)
	writeField(m.Precision)
	writeField(string(m.PositionConvention))
	writeField(m.Producer)
	writeField(m.ProducerKeyID)
	writeField(m.IntegrityChecksum)
	_, _ = h.Write([]byte(strconv.FormatInt(m.Tokens, 10)))
	return hex.EncodeToString(h.Sum(nil))
}

// ResidentClaim is what a resident KV span PURPORTS to be. The integrator resolves
// the resident span's declared axes and performs the signature verification, then
// hands the claim to CheckResidentClaim. Refusal rule 8: a KV artifact imported
// from digest alone is refused — model/tokenizer/position/producer AND a verified
// signature are all required.
type ResidentClaim struct {
	ModelID            string
	TokenizerID        string
	AdapterID          string
	Precision          string
	PositionConvention PositionMode
	Producer           string
	SpanDigest         string
	Tokens             int64
	IntegrityChecksum  string
	SignatureVerified  bool // integrator verified manifest.Signature over ManifestBindingDigest
}

// CheckResidentClaim decides whether a resident KV span may be reused as the
// performance material its manifest describes. It is FAULT (not silent recompute)
// when the binding disagrees or the signature is unverified, and HIT only when
// every binding axis matches AND the signature was verified. A hit is fleet-scoped,
// model-bound prefill material — never semantic proof (§2.4).
func CheckResidentClaim(claim ResidentClaim, manifest KVManifest) LookupVerdict {
	e := manifestEntry(manifest)
	if !claim.SignatureVerified || manifest.Signature.Algorithm == "" || manifest.Signature.Algorithm == "none" {
		return Fault(e, ReasonUnsignedArtifact)
	}
	if claim.ModelID != manifest.ModelID ||
		claim.TokenizerID != manifest.TokenizerID ||
		claim.AdapterID != manifest.AdapterID ||
		claim.Precision != manifest.Precision ||
		claim.PositionConvention != manifest.PositionConvention ||
		claim.Producer != manifest.Producer ||
		claim.SpanDigest != manifest.SpanDigest ||
		claim.Tokens != manifest.Tokens ||
		claim.IntegrityChecksum != manifest.IntegrityChecksum {
		return Fault(e, ReasonManifestMismatch)
	}
	return Hit(e)
}

// FromKVManifest lowers a (verified) manifest into a cachemeta entry on the
// kv_artifact plane. Callers should prefer CheckResidentClaim for the gating
// verdict; this helper is the entry shape a sink observes once a claim passes.
func FromKVManifest(m KVManifest, opts ...Option) Entry {
	e := manifestEntry(m)
	apply(&e, opts)
	return e
}

func manifestEntry(m KVManifest) Entry {
	producer := m.Producer
	if producer == "" {
		producer = "kv-artifact"
	}
	digest := m.SpanDigest
	if digest == "" {
		digest = ManifestBindingDigest(m)
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaKVSpan,
			Length:    m.Tokens,
			Unit:      UnitPositions,
		},
		Plane: PlaneKVArtifact,
		Derivation: Derivation{
			Producer:     producer,
			ModelID:      m.ModelID,
			TokenizerID:  m.TokenizerID,
			PositionMode: m.PositionConvention,
		},
		Validity: Validity{
			TrustEpoch: 0,
		},
		Security: Security{
			// Imported KV is reusable prefill material, not a trust verdict over
			// meaning: it is admitted for KV-prefix reuse but never bypasses the
			// adjudication of a subsequent tool call.
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: AdmissionAllow,
			AdmittedBy:       producer,
			Reason:           "performance_material_not_proof",
		},
		Residency: Residency{Tier: TierDRAM, Owner: producer},
		Coherence: Coherence{InvalidationMode: InvalidationPolicy},
		Labels: map[string]string{
			"adapter":            m.AdapterID,
			"precision":          m.Precision,
			"source_digest":      m.SourceDigest,
			"integrity_checksum": m.IntegrityChecksum,
			"binding_digest":     ManifestBindingDigest(m),
			"sig_algorithm":      m.Signature.Algorithm,
		},
	}
	return e
}
