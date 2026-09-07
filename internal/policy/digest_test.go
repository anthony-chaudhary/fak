package policy

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestInitialPolicyGenerationAndDigest proves an initial policy has Generation 1
// and a valid SHA-256 hex string of length 64.
func TestInitialPolicyGenerationAndDigest(t *testing.T) {
	manifest := []byte(`{"version":"fak-policy/v1","allow":["read_file","list_dir"]}`)

	// Construction via NewPolicy
	p, err := NewPolicy(manifest)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}
	if got := p.Generation(); got != 1 {
		t.Fatalf("initial Generation() = %d, want 1", got)
	}
	digest := p.ContentDigest()
	if len(digest) != 64 {
		t.Fatalf("ContentDigest() length = %d, want 64", len(digest))
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil {
		t.Fatalf("ContentDigest() is not valid hex: %v", err)
	}
	if len(decoded) != 32 {
		t.Fatalf("decoded digest length = %d, want 32", len(decoded))
	}

	// Construction via New
	pNew, err := New(manifest)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := pNew.Generation(); got != 1 {
		t.Fatalf("New Generation() = %d, want 1", got)
	}
	if got := pNew.ContentDigest(); got != digest {
		t.Fatalf("New ContentDigest() = %s, want %s", got, digest)
	}

	// Construction via ParsePolicy
	pParsed, err := ParsePolicy(manifest)
	if err != nil {
		t.Fatalf("ParsePolicy: %v", err)
	}
	if got := pParsed.Generation(); got != 1 {
		t.Fatalf("ParsePolicy Generation() = %d, want 1", got)
	}
	if got := pParsed.ContentDigest(); got != digest {
		t.Fatalf("ParsePolicy ContentDigest() = %s, want %s", got, digest)
	}

	// Direct &Policy{} zero-value
	var pZero Policy
	if got := pZero.Generation(); got != 1 {
		t.Fatalf("zero Policy Generation() = %d, want 1", got)
	}
	zeroDigest := pZero.ContentDigest()
	if len(zeroDigest) != 64 {
		t.Fatalf("zero Policy ContentDigest() length = %d, want 64", len(zeroDigest))
	}
	if _, err := hex.DecodeString(zeroDigest); err != nil {
		t.Fatalf("zero Policy ContentDigest() not valid hex: %v", err)
	}

	// Construction via FromBytes
	pFromBytes := FromBytes(manifest)
	if got := pFromBytes.Generation(); got != 1 {
		t.Fatalf("FromBytes Generation() = %d, want 1", got)
	}
	if got := pFromBytes.ContentDigest(); got != digest {
		t.Fatalf("FromBytes ContentDigest() = %s, want %s", got, digest)
	}

	// Construction via LoadPolicy from file
	tmpDir := t.TempDir()
	policyPath := filepath.Join(tmpDir, "policy.json")
	if err := os.WriteFile(policyPath, manifest, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	pLoaded, err := LoadPolicy(policyPath)
	if err != nil {
		t.Fatalf("LoadPolicy: %v", err)
	}
	if got := pLoaded.Generation(); got != 1 {
		t.Fatalf("LoadPolicy Generation() = %d, want 1", got)
	}
	if got := pLoaded.ContentDigest(); got != digest {
		t.Fatalf("LoadPolicy ContentDigest() = %s, want %s", got, digest)
	}
}

// TestReloadNewContentIncrementsGenerationAndUpdatesDigest proves reloading with new
// content increments Generation from 1 to 2 and updates ContentDigest.
func TestReloadNewContentIncrementsGenerationAndUpdatesDigest(t *testing.T) {
	manifest1 := []byte(`{"version":"fak-policy/v1","allow":["read_file"]}`)
	p, err := NewPolicy(manifest1)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}

	if got := p.Generation(); got != 1 {
		t.Fatalf("initial Generation() = %d, want 1", got)
	}
	digest1 := p.ContentDigest()
	if len(digest1) != 64 {
		t.Fatalf("initial ContentDigest() length = %d, want 64", len(digest1))
	}

	manifest2 := []byte(`{"version":"fak-policy/v1","allow":["read_file","write_file"]}`)
	p.RecordReload(manifest2)

	if got := p.Generation(); got != 2 {
		t.Fatalf("after reload Generation() = %d, want 2", got)
	}
	digest2 := p.ContentDigest()
	if len(digest2) != 64 {
		t.Fatalf("after reload ContentDigest() length = %d, want 64", len(digest2))
	}
	if digest2 == digest1 {
		t.Fatalf("after reload ContentDigest() did not change: %s == %s", digest2, digest1)
	}
	expectedDigest2 := ComputeContentDigest(manifest2)
	if digest2 != expectedDigest2 {
		t.Fatalf("after reload ContentDigest() = %s, want %s", digest2, expectedDigest2)
	}

	// Subsequent reload with another new content increments generation to 3
	manifest3 := []byte(`{"version":"fak-policy/v1","allow":["search_kb"]}`)
	p.RecordReload(manifest3)

	if got := p.Generation(); got != 3 {
		t.Fatalf("after second reload Generation() = %d, want 3", got)
	}
	digest3 := p.ContentDigest()
	if digest3 == digest2 || digest3 == digest1 {
		t.Fatalf("digest3 collided with earlier digests: %s", digest3)
	}
	expectedDigest3 := ComputeContentDigest(manifest3)
	if digest3 != expectedDigest3 {
		t.Fatalf("digest3 = %s, want %s", digest3, expectedDigest3)
	}
}

// TestReloadIdenticalContentPreservesOrIncrementsGeneration proves reloading with identical
// content preserves the ContentDigest and increments the generation counter monotonically.
func TestReloadIdenticalContentPreservesOrIncrementsGeneration(t *testing.T) {
	manifest := []byte(`{"version":"fak-policy/v1","allow":["read_file"]}`)
	p, err := NewPolicy(manifest)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}

	initialGen := p.Generation()
	initialDigest := p.ContentDigest()

	// Reload with identical content
	p.RecordReload(manifest)

	// ContentDigest must remain identical
	if got := p.ContentDigest(); got != initialDigest {
		t.Fatalf("ContentDigest changed after identical reload: got %s, want %s", got, initialDigest)
	}

	// Monotonic reload generation counter increments to 2
	genAfter := p.Generation()
	if genAfter != initialGen+1 {
		t.Fatalf("Generation() after identical reload = %d, want %d", genAfter, initialGen+1)
	}

	// Another identical reload increments to 3 while keeping digest unchanged
	p.RecordReload(manifest)
	if got := p.ContentDigest(); got != initialDigest {
		t.Fatalf("ContentDigest changed after second identical reload: got %s, want %s", got, initialDigest)
	}
	if got := p.Generation(); got != initialGen+2 {
		t.Fatalf("Generation() after second identical reload = %d, want %d", got, initialGen+2)
	}
}

// TestDeterministicDigestComputation proves SHA-256 digest computation is deterministic,
// collision-resistant across distinct contents, and produces valid 64-char hex strings.
func TestDeterministicDigestComputation(t *testing.T) {
	sample := []byte(`{"version":"fak-policy/v1","allow":["tool_a","tool_b"]}`)

	// Repeat 50 times; must always produce identical string
	first := ComputeContentDigest(sample)
	if len(first) != 64 {
		t.Fatalf("digest length = %d, want 64", len(first))
	}
	for i := 0; i < 50; i++ {
		repeat := ComputeContentDigest(sample)
		if repeat != first {
			t.Fatalf("iteration %d: non-deterministic digest got %s, want %s", i, repeat, first)
		}
	}

	// Empty bytes produce known SHA-256
	emptyDigest := ComputeContentDigest(nil)
	const wantEmpty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if emptyDigest != wantEmpty {
		t.Fatalf("empty digest = %s, want %s", emptyDigest, wantEmpty)
	}
	if got := ComputeContentDigest([]byte{}); got != wantEmpty {
		t.Fatalf("empty slice digest = %s, want %s", got, wantEmpty)
	}

	// Different content produces different digests
	diffSample := []byte(`{"version":"fak-policy/v1","allow":["tool_a","tool_c"]}`)
	diffDigest := ComputeContentDigest(diffSample)
	if diffDigest == first {
		t.Fatalf("unexpected collision between different contents: %s", diffDigest)
	}

	// ComputeRulesetDigest is deterministic over Manifest
	m := Manifest{
		Version: Version,
		Allow:   []string{"read_file"},
	}
	mDigest1 := ComputeRulesetDigest(m)
	mDigest2 := m.ContentDigest()
	if len(mDigest1) != 64 || mDigest1 != mDigest2 {
		t.Fatalf("manifest digest mismatch: %s vs %s", mDigest1, mDigest2)
	}
}

// TestPolicySummaryJSON proves PolicySummary captures ContentDigest and Generation,
// serializes with the exact JSON tags, and round-trips through encoding/json.
func TestPolicySummaryJSON(t *testing.T) {
	manifest := []byte(`{"version":"fak-policy/v1","allow":["read_file"]}`)
	p, err := NewPolicy(manifest)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}

	summary := p.Summary()
	if summary.Generation != 1 {
		t.Fatalf("summary.Generation = %d, want 1", summary.Generation)
	}
	if summary.ContentDigest != p.ContentDigest() {
		t.Fatalf("summary.ContentDigest = %s, want %s", summary.ContentDigest, p.ContentDigest())
	}

	// PolicySummary method alias
	if p.PolicySummary() != summary {
		t.Fatalf("PolicySummary() != Summary()")
	}

	// JSON serialization format
	rawJSON, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(rawJSON, &parsed); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if _, ok := parsed["content_digest"]; !ok {
		t.Fatalf("missing 'content_digest' JSON key: %s", string(rawJSON))
	}
	if _, ok := parsed["generation"]; !ok {
		t.Fatalf("missing 'generation' JSON key: %s", string(rawJSON))
	}

	// Round-trip back into PolicySummary
	var roundTripped PolicySummary
	if err := json.Unmarshal(rawJSON, &roundTripped); err != nil {
		t.Fatalf("json.Unmarshal into PolicySummary: %v", err)
	}
	if roundTripped != summary {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", roundTripped, summary)
	}
}

// TestConcurrentPolicyAccess tests thread-safety when reading ContentDigest and Generation
// while concurrent reloads occur.
func TestConcurrentPolicyAccess(t *testing.T) {
	manifest := []byte(`{"version":"fak-policy/v1","allow":["read_file"]}`)
	p, err := NewPolicy(manifest)
	if err != nil {
		t.Fatalf("NewPolicy: %v", err)
	}

	var wg sync.WaitGroup
	workers := 10
	iterations := 100

	// Reader workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				digest := p.ContentDigest()
				if len(digest) != 64 {
					t.Errorf("read digest invalid length: %d", len(digest))
				}
				gen := p.Generation()
				if gen < 1 {
					t.Errorf("read generation invalid: %d", gen)
				}
				s := p.Summary()
				if len(s.ContentDigest) != 64 || s.Generation < 1 {
					t.Errorf("read summary invalid: %+v", s)
				}
			}
		}()
	}

	// Writer worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < iterations; j++ {
			p.RecordReload([]byte(`{"version":"fak-policy/v1","allow":["reloaded_tool"]}`))
		}
	}()

	wg.Wait()

	if p.Generation() <= 1 {
		t.Fatalf("generation should have advanced during reloads: got %d", p.Generation())
	}
}
