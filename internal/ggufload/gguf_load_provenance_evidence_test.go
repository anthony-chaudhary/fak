package ggufload

// gguf_load_provenance_evidence_test.go — the PRODUCER-TO-EVIDENCE seam witness
// for #4746 (root incident #4273).
//
// WHAT WAS STILL UNWITNESSED. #4746's acceptance reads "run/readiness evidence
// records the artifact digest, so publication claims are bound to the actual
// loader semantics". Three pieces landed separately and each is tested in
// isolation: internal/model owns the artifact and its Digest, this package's
// File.LoadProvenance produces one from a parsed header, and
// internal/provenance.RunManifest carries and shape-checks a load_provenance
// content address. Nothing tested them TOGETHER, and they are deliberately built
// so that neither end can check the other — provenance is stdlib-only and never
// imports the loader (it just carries the address), while model cannot see the
// evidence schema at all.
//
// That leaves a real, silent seam. provenance.Validate refuses anything that is
// not "sha256:<64 lowercase hex>"; model.LoadProvenance.Digest happens to emit
// exactly that. The two agree today by convention alone. If either side changed
// its rendering — an uppercase hex encoder, a "blake3:" prefix, a truncated
// address — every real load would start producing a digest the manifest refuses,
// and the failure would surface as "inconclusive manifest" at publication time
// rather than at the change that caused it. This file is the test that fails at
// the change instead.
//
// It also upgrades the acceptance criterion from synthetic to real. The
// provenance package's own test necessarily stamps a hand-written digest
// (loaderSemanticsA/B) because it may not import a producer; here the digest
// comes from an actual GGUF header parse, so the criterion is witnessed against
// the value a live load would supply.
//
// The intra-tier import is legal: architest puts ggufload and provenance both at
// tier 1, and the edge exists only in this test file — the loader binary still
// does not depend on the evidence schema.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/provenance"
)

// evidenceManifest is a complete, valid run manifest carrying digest as its
// load-provenance content address. Every other field is fixed, so two manifests
// built from different digests differ in loader semantics and in NOTHING else —
// which is the #4273 shape the comparison has to catch.
func evidenceManifest(digest string) provenance.RunManifest {
	return provenance.RunManifest{
		Case:           "decode-parity/qwen36-ssm-a@v1",
		Model:          "qwen3.6@sha256:" + "ab",
		Tokenizer:      "qwen-bpe@sha256:cd",
		LoadProvenance: digest,
		Backend:        "fak-engine",
		Seed:           7,
		CodeRev:        "github.com/anthony-chaudhary/fak@095101ee8",
		Baseline:       "nightly/qwen36-decode-parity",
		Tolerance:      "exact",
		BinaryRev:      "fak-0.9.0+095101ee8",
		Hardware:       "cpu-avx512",
		DecodeParams:   map[string]string{"temperature": "0"},
		CacheState:     "cold",
		Normalization:  "nfc+trim",
		Tier:           provenance.TierNightly,
		Cost:           "1.1s / 1xcpu / 512 tok",
	}
}

// TestLoadProvenanceDigestIsAcceptedByRunEvidence closes the producer-to-evidence
// seam: the digest a REAL header parse produces must satisfy the run manifest's
// content-address shape check. Neither package can assert this alone — provenance
// may not import the loader, and the loader does not know the evidence schema —
// so without this witness the two agree only by convention.
func TestLoadProvenanceDigestIsAcceptedByRunEvidence(t *testing.T) {
	f := readProvenanceFixture(t, "qwen3.6", 3)
	p, err := f.LoadProvenance(loadProvenanceScopeForTest())
	if err != nil {
		t.Fatalf("LoadProvenance: %v", err)
	}

	// The load really did record the #4273 transform, so the digest below
	// addresses an artifact with loader semantics in it rather than an empty one.
	if _, _, ok := p.TransformTensors(TransformInvertNegExpDecay); !ok {
		t.Fatalf("fixture recorded no %s transform; the digest would bind nothing",
			TransformInvertNegExpDecay)
	}

	m := evidenceManifest(p.Digest())
	if err := m.Validate(); err != nil {
		t.Fatalf("run evidence refused a real load-provenance digest %q: %v\n"+
			"the producer's Digest rendering and the manifest's content-address "+
			"check have drifted apart", p.Digest(), err)
	}
}

// TestRunEvidenceDivergesOnRealLoaderSemantics is #4746's acceptance criterion
// "run/readiness evidence records the artifact digest, so publication claims are
// bound to the actual loader semantics", witnessed with real digests.
//
// The two runs stand for the broken and the fixed #4273 load: they agree on every
// fact the manifest records — model, tokenizer, backend, hardware, seed, decode
// params — and differ only in what the loader DID to the bytes. Before the
// load_provenance field existed, that pair fingerprinted identically and Compare
// returned pass, certifying a run whose tensors were built under different
// semantics than its baseline.
func TestRunEvidenceDivergesOnRealLoaderSemantics(t *testing.T) {
	f := readProvenanceFixture(t, "qwen3.6", 3)
	scope := loadProvenanceScopeForTest()

	good, err := f.LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance (baseline): %v", err)
	}
	// A different loader revision is the cheapest real change of loader
	// SEMANTICS: same checkpoint bytes, different code deciding what they mean.
	bumped := scope
	bumped.LoaderRev = "fak-loader@deadbeef"
	moved, err := f.LoadProvenance(bumped)
	if err != nil {
		t.Fatalf("LoadProvenance (bumped loader rev): %v", err)
	}
	if good.Digest() == moved.Digest() {
		t.Fatalf("precondition failed: two loader revisions produced one digest %q", good.Digest())
	}

	baseline := evidenceManifest(good.Digest())
	candidate := evidenceManifest(moved.Digest())

	if baseline.Equivalent(candidate) {
		t.Fatal("two runs under different loader semantics must not be equivalent")
	}
	art := provenance.Compare(baseline, candidate)
	if art.Pass() {
		t.Fatalf("a real loader-semantics difference must not pass, got verdict %q", art.Verdict)
	}
	if art.Divergence == nil {
		t.Fatal("a failing compare must localize its divergence")
	}
	if art.Divergence.Field != "load_provenance" {
		t.Fatalf("divergence must localize to load_provenance, got %q", art.Divergence.Field)
	}

	// Negative control: the SAME load re-parsed must still pass, so the field
	// discriminates loader semantics rather than failing every comparison put
	// through it. This is the determinism property doing evidence work — a digest
	// that wobbled per parse would make every run diverge from its own baseline.
	g := readProvenanceFixture(t, "qwen3.6", 3)
	reparsed, err := g.LoadProvenance(scope)
	if err != nil {
		t.Fatalf("LoadProvenance (re-parsed): %v", err)
	}
	if art := provenance.Compare(baseline, evidenceManifest(reparsed.Digest())); !art.Pass() {
		t.Fatalf("an identical re-parsed load must still pass, got %q (%s)", art.Verdict, art.Reason)
	}
}
