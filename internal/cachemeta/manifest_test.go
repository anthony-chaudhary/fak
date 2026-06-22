package cachemeta

import "testing"

func sampleManifest() KVManifest {
	return KVManifest{
		SourceDigest:       "src-1",
		SpanDigest:         "span-1",
		Tokens:             4096,
		ModelID:            "m",
		TokenizerID:        "tok",
		AdapterID:          "base",
		Precision:          "fp16",
		PositionConvention: PositionPrefixAligned,
		Producer:           "publisher",
		ProducerKeyID:      "key-1",
		IntegrityChecksum:  "cksum-1",
		Signature:          ManifestSignature{Algorithm: "ed25519", Value: "deadbeef"},
		AccessPolicy:       "fleet-internal",
	}
}

func TestManifestBindingDigestIsDeterministicOverBindingAxes(t *testing.T) {
	m := sampleManifest()
	d1 := ManifestBindingDigest(m)
	d2 := ManifestBindingDigest(m)
	if d1 != d2 {
		t.Fatalf("binding digest not deterministic")
	}
	// Tampering with ANY binding axis must change the digest a signature covers.
	tampered := m
	tampered.ModelID = "other"
	if ManifestBindingDigest(tampered) == d1 {
		t.Fatalf("model change must change binding digest")
	}
	tampered = m
	tampered.Precision = "int8"
	if ManifestBindingDigest(tampered) == d1 {
		t.Fatalf("precision change must change binding digest")
	}
	tampered = m
	tampered.PositionConvention = PositionRelocatable
	if ManifestBindingDigest(tampered) == d1 {
		t.Fatalf("position-convention change must change binding digest")
	}
}

// Refusal rule 8: a KV artifact imported from digest alone is refused — needs
// model/tokenizer/position/producer AND a verified signature.
func TestCheckResidentClaimRefusesUnsignedArtifact(t *testing.T) {
	m := sampleManifest()
	claim := ResidentClaim{
		ModelID: m.ModelID, TokenizerID: m.TokenizerID, AdapterID: m.AdapterID,
		Precision: m.Precision, PositionConvention: m.PositionConvention,
		Producer: m.Producer, SpanDigest: m.SpanDigest, IntegrityChecksum: m.IntegrityChecksum,
		SignatureVerified: false, // <-- not verified
	}
	v := CheckResidentClaim(claim, m)
	if v.Kind != LookupFault || v.Reason != ReasonUnsignedArtifact {
		t.Fatalf("unsigned artifact must be FAULT(unsigned_artifact), got %+v", v)
	}
}

func TestCheckResidentClaimRefusesBindingMismatch(t *testing.T) {
	m := sampleManifest()
	claim := ResidentClaim{
		ModelID: m.ModelID, TokenizerID: m.TokenizerID, AdapterID: m.AdapterID,
		Precision:          "int8", // <-- mismatch
		PositionConvention: m.PositionConvention,
		Producer:           m.Producer, SpanDigest: m.SpanDigest, IntegrityChecksum: m.IntegrityChecksum,
		SignatureVerified: true,
	}
	v := CheckResidentClaim(claim, m)
	if v.Kind != LookupFault || v.Reason != ReasonManifestMismatch {
		t.Fatalf("binding mismatch must be FAULT(manifest_mismatch), got %+v", v)
	}
}

func TestCheckResidentClaimRefusesLengthMismatch(t *testing.T) {
	m := sampleManifest()
	claim := ResidentClaim{
		ModelID: m.ModelID, TokenizerID: m.TokenizerID, AdapterID: m.AdapterID,
		Precision: m.Precision, PositionConvention: m.PositionConvention,
		Producer: m.Producer, SpanDigest: m.SpanDigest,
		Tokens:            m.Tokens - 1,
		IntegrityChecksum: m.IntegrityChecksum,
		SignatureVerified: true,
	}
	v := CheckResidentClaim(claim, m)
	if v.Kind != LookupFault || v.Reason != ReasonManifestMismatch {
		t.Fatalf("length mismatch must be FAULT(manifest_mismatch), got %+v", v)
	}
}

func TestCheckResidentClaimHitsOnVerifiedExactBinding(t *testing.T) {
	m := sampleManifest()
	claim := ResidentClaim{
		ModelID: m.ModelID, TokenizerID: m.TokenizerID, AdapterID: m.AdapterID,
		Precision: m.Precision, PositionConvention: m.PositionConvention,
		Producer: m.Producer, SpanDigest: m.SpanDigest, Tokens: m.Tokens,
		IntegrityChecksum: m.IntegrityChecksum,
		SignatureVerified: true,
	}
	v := CheckResidentClaim(claim, m)
	if v.Kind != LookupHit || !v.CanServe() {
		t.Fatalf("verified exact binding should HIT, got %+v", v)
	}
	// A hit is performance material, not semantic proof.
	if v.Entry.Security.Reason != "performance_material_not_proof" {
		t.Fatalf("KV-artifact hit should be marked performance material, got %q", v.Entry.Security.Reason)
	}
	if v.Entry.Plane != PlaneKVArtifact {
		t.Fatalf("KV artifact must be on kv_artifact plane: %s", v.Entry.Plane)
	}
}

func TestFromKVManifestDescribesImportableArtifact(t *testing.T) {
	e := FromKVManifest(sampleManifest())
	if e.Labels["binding_digest"] != ManifestBindingDigest(sampleManifest()) {
		t.Fatalf("entry should carry the recomputable binding digest")
	}
	if e.Labels["sig_algorithm"] != "ed25519" {
		t.Fatalf("entry should name the signature algorithm")
	}
	// Provenance + access-control metadata must survive the lowering so a sink can
	// see who produced the KV and under what access policy it was admitted.
	if e.Labels["producer_key_id"] != "key-1" {
		t.Fatalf("entry should carry the producer key id, got %q", e.Labels["producer_key_id"])
	}
	if e.Labels["access_policy"] != "fleet-internal" {
		t.Fatalf("entry should carry the access policy, got %q", e.Labels["access_policy"])
	}
}

// Acceptance §2.4: provenance metadata (an attributable producer + its key id) is
// REQUIRED before any third-party KV is admissible — digest/identity alone is not.
func TestValidateManifestRequiresProvenance(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*KVManifest)
	}{
		{"no producer", func(m *KVManifest) { m.Producer = "" }},
		{"no producer key", func(m *KVManifest) { m.ProducerKeyID = "" }},
	} {
		m := sampleManifest()
		tc.mutate(&m)
		if reason, ok := ValidateManifest(m); ok || reason != ReasonMissingProvenance {
			t.Fatalf("%s: want refuse(missing_provenance), got reason=%q ok=%v", tc.name, reason, ok)
		}
		// The lowering checker must surface the same refusal, not a HIT.
		claim := claimFor(m)
		v := CheckResidentClaim(claim, m)
		if v.Kind != LookupFault || v.Reason != ReasonMissingProvenance {
			t.Fatalf("%s: CheckResidentClaim must FAULT(missing_provenance), got %+v", tc.name, v)
		}
	}
}

// Acceptance §2.4: access-control metadata is REQUIRED for any third-party KV; a
// manifest with no declared access policy is refused before admission.
func TestCheckResidentClaimRefusesMissingAccessControl(t *testing.T) {
	m := sampleManifest()
	m.AccessPolicy = ""
	if reason, ok := ValidateManifest(m); ok || reason != ReasonAccessControlReq {
		t.Fatalf("want refuse(access_control_required), got reason=%q ok=%v", reason, ok)
	}
	v := CheckResidentClaim(claimFor(m), m)
	if v.Kind != LookupFault || v.Reason != ReasonAccessControlReq {
		t.Fatalf("missing access policy must FAULT(access_control_required), got %+v", v)
	}
}

// claimFor builds a fully-matching, signature-verified resident claim for a
// manifest so a test can isolate the metadata gate from the binding/signature gates.
func claimFor(m KVManifest) ResidentClaim {
	return ResidentClaim{
		ModelID: m.ModelID, TokenizerID: m.TokenizerID, AdapterID: m.AdapterID,
		Precision: m.Precision, PositionConvention: m.PositionConvention,
		Producer: m.Producer, SpanDigest: m.SpanDigest, Tokens: m.Tokens,
		IntegrityChecksum: m.IntegrityChecksum, SignatureVerified: true,
	}
}
