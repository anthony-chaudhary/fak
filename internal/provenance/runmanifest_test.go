package provenance

import (
	"bytes"
	"strings"
	"testing"
)

// loaderSemanticsA / loaderSemanticsB are two well-formed load-provenance
// digests standing for two loads of the SAME model bytes under different loader
// semantics — the #4273 shape, where every other recorded fact agrees.
var (
	loaderSemanticsA = "sha256:" + strings.Repeat("a1", 32)
	loaderSemanticsB = "sha256:" + strings.Repeat("b2", 32)
)

// goodManifest is a fully-populated, valid baseline quality-run manifest — the
// starting point every acceptance test perturbs by exactly one field.
func goodManifest() RunManifest {
	return RunManifest{
		Case:      "decode-parity/greedy-eos@v3",
		Model:     "glm-4.6@sha256:1111",
		Tokenizer: "glm-bpe@sha256:2222",
		Backend:   "fak-engine",
		// The loader-semantics content address (#4746). Synthetic here on
		// purpose: this package must stay stdlib-only, so it carries the digest
		// model.LoadProvenance.Digest emits without importing the loader.
		LoadProvenance: loaderSemanticsA,
		Seed:           42,
		CodeRev:        "github.com/anthony-chaudhary/fak@abc1234",
		Baseline:       "nightly-2026-06-25/decode-parity",
		Tolerance:      "exact",
		BinaryRev:      "fak-0.9.0+abc1234",
		Hardware:       "cpu-avx512",
		DecodeParams: map[string]string{
			"temperature": "0",
			"top_p":       "1",
			"max_tokens":  "256",
		},
		CacheState:    "cold",
		Normalization: "nfc+trim",
		Env: map[string]string{
			"OMP_NUM_THREADS": "8",
			"FAK_API_KEY":     "sk-secret-value",
		},
		Tier:    TierNightly,
		Cost:    "3.2s / 1xcpu / 4.1k tok",
		Secrets: []string{"OMP_NUM_THREADS"}, // pretend this one is sensitive too
	}
}

// TestEquivalentRunsNormalizeIdentically — acceptance criterion 1: two runs that
// differ only in insignificant whitespace and map order share a fingerprint, and
// changing exactly one decode flag makes them diverge visibly.
func TestEquivalentRunsNormalizeIdentically(t *testing.T) {
	a := goodManifest()

	// b is a.equivalent: whitespace noise + a different map iteration order (Go
	// maps are unordered anyway, so we rebuild the maps to prove order-independence).
	b := goodManifest()
	b.Model = "  glm-4.6@sha256:1111  "
	b.Normalization = "nfc+trim "
	b.DecodeParams = map[string]string{"max_tokens": "256", "temperature": "0", "top_p": "1"}

	if !a.Equivalent(b) {
		t.Fatalf("equivalent runs must normalize identically:\n a=%s\n b=%s", a.Fingerprint(), b.Fingerprint())
	}

	// Change exactly one flag: it must break equivalence AND be the visible diff.
	c := goodManifest()
	c.DecodeParams["temperature"] = "0.7"
	if a.Equivalent(c) {
		t.Fatal("a changed decode flag must change the fingerprint")
	}
	d := a.FirstDivergence(c)
	if d == nil || d.Field != "decode_params.temperature" || d.Baseline != "0" || d.Candidate != "0.7" {
		t.Fatalf("one changed flag must be visible as the first divergence, got %+v", d)
	}
}

// TestPlantedDefectFailsThenFixPasses is THE WITNESS (#4514): a captured proof that
// fails against a planted representative defect and passes after the fix, in a
// clean, independently replayed environment (every verdict is re-derived from the
// JSON bundle alone, not from the in-memory manifest).
func TestPlantedDefectFailsThenFixPasses(t *testing.T) {
	baseline := goodManifest()

	// A clean, matching candidate run PASSES — and the pass survives an independent
	// replay from its serialized bundle.
	clean := goodManifest()
	if art := replayed(t, Compare(baseline, clean)); !art.Pass() {
		t.Fatalf("a matching run must pass, got verdict=%q reason=%q", art.Verdict, art.Reason)
	}

	// Plant a representative defect: a regression silently flips the engine's decode
	// temperature (a classic engine-caused quality drift the middle ladder exists to
	// localize). Compare must FAIL and name the first actionable divergence.
	defective := goodManifest()
	defective.DecodeParams["temperature"] = "0.8"
	art := replayed(t, Compare(baseline, defective))
	if art.Pass() {
		t.Fatal("a planted decode-temperature defect must NOT pass")
	}
	if art.Verdict != verdictFail {
		t.Fatalf("verdict must be fail, got %q", art.Verdict)
	}
	if art.Divergence == nil || art.Divergence.Field != "decode_params.temperature" {
		t.Fatalf("failure must localize the first actionable divergence, got %+v", art.Divergence)
	}

	// Fix the defect (realign the flag): the same comparison now PASSES, again via an
	// independent replay. Fail-before / pass-after, both re-adjudicated from bytes.
	fixed := goodManifest()
	if art := replayed(t, Compare(baseline, fixed)); !art.Pass() {
		t.Fatalf("after the fix the run must pass, got verdict=%q reason=%q", art.Verdict, art.Reason)
	}
}

// replayed serializes an artifact and rebuilds it in a fresh value — standing in
// for a clean, independent environment that holds only the bundle bytes.
func replayed(t *testing.T, art ReplayArtifact) ReplayArtifact {
	t.Helper()
	out, err := ReplayFrom(art.JSON())
	if err != nil {
		t.Fatalf("replay from bundle failed: %v", err)
	}
	if out.Verdict != art.Verdict || out.Fingerprint != art.Fingerprint {
		t.Fatalf("replay must reproduce the verdict/fingerprint: %q/%s vs %q/%s",
			out.Verdict, out.Fingerprint, art.Verdict, art.Fingerprint)
	}
	return out
}

// TestMissingEvidenceNeverPass — acceptance criterion 3 fail-closed rule: an
// incomplete manifest is inconclusive, and inconclusive is never a pass.
func TestMissingEvidenceNeverPass(t *testing.T) {
	baseline := goodManifest()

	incomplete := goodManifest()
	incomplete.Tokenizer = "" // evidence dropped
	if err := incomplete.Validate(); err == nil {
		t.Fatal("a manifest missing the tokenizer must not validate")
	}
	art := Compare(baseline, incomplete)
	if art.Pass() {
		t.Fatal("inconclusive (missing) evidence must never pass")
	}
	if art.Verdict != verdictInconclusive || !strings.Contains(art.Reason, "tokenizer") {
		t.Fatalf("inconclusive verdict must name the missing field, got %q / %q", art.Verdict, art.Reason)
	}

	// An inconclusive BASELINE also blocks a pass (fail-closed on both sides).
	if Compare(RunManifest{}, goodManifest()).Pass() {
		t.Fatal("an empty baseline must never yield a pass")
	}
}

// TestRequiredProvenanceFields — acceptance criterion 2: each case must record
// model, tokenizer, engine/backend, seed-or-oracle, code revision, and
// tolerance/baseline provenance; dropping any one is inconclusive and named.
func TestRequiredProvenanceFields(t *testing.T) {
	drops := []struct {
		name string
		mut  func(*RunManifest)
		want string
	}{
		{"model", func(m *RunManifest) { m.Model = "" }, "model"},
		{"load_provenance", func(m *RunManifest) { m.LoadProvenance = "" }, "load_provenance"},
		{"tokenizer", func(m *RunManifest) { m.Tokenizer = "" }, "tokenizer"},
		{"backend", func(m *RunManifest) { m.Backend = "" }, "backend"},
		{"code_rev", func(m *RunManifest) { m.CodeRev = "" }, "code_rev"},
		{"baseline", func(m *RunManifest) { m.Baseline = "" }, "baseline"},
		{"tolerance", func(m *RunManifest) { m.Tolerance = "" }, "tolerance"},
		{"seed-and-oracle", func(m *RunManifest) { m.Seed = 0; m.Oracle = "" }, "seed|oracle"},
	}
	for _, d := range drops {
		m := goodManifest()
		d.mut(&m)
		err := m.Validate()
		if err == nil {
			t.Fatalf("%s: dropping the field must be inconclusive", d.name)
		}
		if !strings.Contains(err.Error(), d.want) {
			t.Fatalf("%s: error must name %q, got %v", d.name, d.want, err)
		}
	}

	// A deterministic ORACLE substitutes for a seed — that combination validates.
	m := goodManifest()
	m.Seed = 0
	m.Oracle = "golden/decode-parity-greedy"
	if err := m.Validate(); err != nil {
		t.Fatalf("a deterministic oracle must substitute for a seed: %v", err)
	}
}

// TestLoaderSemanticsDivergeUnderIdenticalRecordedFacts is the #4746 acceptance
// criterion "run evidence records the artifact digest, so publication claims are
// bound to the actual loader semantics" — stated as the #4273 failure it exists
// to catch.
//
// The two runs here are the broken load and the fixed load: SAME model bytes,
// tokenizer, backend, hardware, quant, seed, and decode params — agreement on
// every fact the manifest recorded before this field existed. They differ only
// in what the loader DID to those bytes. Without load_provenance the pair is
// indistinguishable and Compare returns pass, certifying a run whose tensors
// were built under different semantics than its baseline.
func TestLoaderSemanticsDivergeUnderIdenticalRecordedFacts(t *testing.T) {
	baseline := goodManifest()
	candidate := goodManifest()
	candidate.LoadProvenance = loaderSemanticsB

	if baseline.Fingerprint() == candidate.Fingerprint() {
		t.Fatalf("two loads under different loader semantics must not fingerprint identically")
	}

	art := Compare(baseline, candidate)
	if art.Pass() {
		t.Fatalf("a loader-semantics difference must not pass, got verdict %q", art.Verdict)
	}
	if art.Divergence == nil {
		t.Fatalf("a failing compare must localize its divergence")
	}
	if art.Divergence.Field != "load_provenance" {
		t.Fatalf("divergence must localize to load_provenance, got %q", art.Divergence.Field)
	}
	if art.Divergence.Baseline != loaderSemanticsA || art.Divergence.Candidate != loaderSemanticsB {
		t.Fatalf("divergence must publish both loader digests, got %q vs %q",
			art.Divergence.Baseline, art.Divergence.Candidate)
	}

	// The loader delta must be read BEFORE any downstream flag: a difference in
	// loader semantics invalidates the comparison of everything below it, so an
	// operator must not be sent chasing a decode param first.
	candidate.DecodeParams = map[string]string{"temperature": "0.9"}
	if d := Compare(baseline, candidate).Divergence; d == nil || d.Field != "load_provenance" {
		t.Fatalf("load_provenance must outrank a downstream decode param, got %+v", d)
	}

	// Negative control: agreeing on loader semantics still passes, so the field
	// discriminates rather than failing everything put through it.
	same := goodManifest()
	if art := Compare(baseline, same); !art.Pass() {
		t.Fatalf("identical loader semantics must still pass, got %q (%s)", art.Verdict, art.Reason)
	}
}

// TestMalformedLoadProvenanceIsInconclusive pins the shape check. A
// present-but-unshaped digest is worse than an absent one: it satisfies the
// presence check while addressing no artifact, so the manifest LOOKS bound to
// its loader semantics and proves nothing.
func TestMalformedLoadProvenanceIsInconclusive(t *testing.T) {
	malformed := []struct{ name, val string }{
		{"no algorithm prefix", strings.Repeat("a1", 32)},
		{"wrong algorithm", "sha1:" + strings.Repeat("a1", 32)},
		{"prefix only", "sha256:"},
		{"too short", "sha256:" + strings.Repeat("a1", 16)},
		{"too long", "sha256:" + strings.Repeat("a1", 48)},
		{"uppercase hex", "sha256:" + strings.Repeat("A1", 32)},
		{"non-hex", "sha256:" + strings.Repeat("zz", 32)},
		{"a model id, not a digest", "qwen3.6@sha256:1111"},
	}
	for _, bad := range malformed {
		m := goodManifest()
		m.LoadProvenance = bad.val
		err := m.Validate()
		if err == nil {
			t.Fatalf("%s: %q must not validate as a content address", bad.name, bad.val)
		}
		if !strings.Contains(err.Error(), "load_provenance") {
			t.Fatalf("%s: error must name the field, got %v", bad.name, err)
		}
		// Fail-closed all the way through: an unshaped digest can never reach a
		// pass verdict from either side of a comparison.
		if art := Compare(m, goodManifest()); art.Pass() {
			t.Fatalf("%s: an unshaped baseline digest must never pass", bad.name)
		}
		if art := Compare(goodManifest(), m); art.Pass() {
			t.Fatalf("%s: an unshaped candidate digest must never pass", bad.name)
		}
	}
}

// TestTierAndCostRequired — acceptance criterion 4: a case must be assigned an
// explicit PR/nightly/release tier and document its runtime/resource cost.
func TestTierAndCostRequired(t *testing.T) {
	noTier := goodManifest()
	noTier.Tier = TierUnset
	if err := noTier.Validate(); err == nil || !strings.Contains(err.Error(), "tier") {
		t.Fatalf("an unassigned tier must be inconclusive, got %v", err)
	}

	noCost := goodManifest()
	noCost.Cost = ""
	if err := noCost.Validate(); err == nil || !strings.Contains(err.Error(), "cost") {
		t.Fatalf("a missing cost must be inconclusive, got %v", err)
	}

	for _, tr := range []Tier{TierPR, TierNightly, TierRelease} {
		if !tr.Valid() {
			t.Fatalf("%v must be a valid explicit tier", tr)
		}
	}
	if TierUnset.Valid() {
		t.Fatal("the zero-value tier must not be valid")
	}
}

// TestReplayArtifactScrubbedAndRoundTrips — acceptance criterion 3 scrub rule plus
// replay-completeness: the emitted bundle carries no secret values, yet replays to
// the exact same fingerprint, so an independent environment reproduces the run.
func TestReplayArtifactScrubbedAndRoundTrips(t *testing.T) {
	baseline := goodManifest()
	candidate := goodManifest()
	candidate.DecodeParams["temperature"] = "0.9" // force a failure so an artifact is emitted

	art := Compare(baseline, candidate)
	blob := art.JSON()

	// Secret VALUES never appear in the bundle: the api-key value (secret-shaped key)
	// and the explicitly-listed OMP thread value are both redacted.
	if bytes.Contains(blob, []byte("sk-secret-value")) {
		t.Fatal("a secret-shaped value must be scrubbed from the replay artifact")
	}
	if art.Manifest.Env["FAK_API_KEY"] != redacted {
		t.Fatalf("secret-shaped env value must be redacted, got %q", art.Manifest.Env["FAK_API_KEY"])
	}
	if art.Manifest.Env["OMP_NUM_THREADS"] != redacted {
		t.Fatalf("explicitly-listed secret must be redacted, got %q", art.Manifest.Env["OMP_NUM_THREADS"])
	}
	// A non-secret value is preserved (scrubbing is targeted, not blanket).
	if art.Manifest.DecodeParams["max_tokens"] != "256" {
		t.Fatalf("a non-secret value must be preserved, got %q", art.Manifest.DecodeParams["max_tokens"])
	}

	// Replay-complete: rebuilding from the bundle reproduces the candidate's
	// fingerprint exactly (scrubbing is idempotent and secrets are outside identity).
	back, err := ReplayFrom(blob)
	if err != nil {
		t.Fatalf("ReplayFrom: %v", err)
	}
	if back.Fingerprint != candidate.Fingerprint() {
		t.Fatalf("replay must reproduce the fingerprint: %s vs %s", back.Fingerprint, candidate.Fingerprint())
	}
	if got := back.Manifest.Fingerprint(); got != back.Fingerprint {
		t.Fatalf("the replayed manifest must re-derive its own recorded fingerprint: %s vs %s", got, back.Fingerprint)
	}
}
