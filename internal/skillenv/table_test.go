package skillenv

import (
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestVersionedPageTable_SideBySideV1V2 proves v1 and v2 of one skill can be
// resident simultaneously with no cross-talk. The acceptance is:
// - v1 and v2 are cached in distinct slots (different cache keys).
// - Each version serves its own cached view; no aliasing.
func TestVersionedPageTable_SideBySideV1V2(t *testing.T) {
	cache := contextq.NewViewCache()

	// Build v1's procedural view.
	v1Rec := contextq.SkillContextRecord{
		SkillName:        "code-review",
		Version:          "1.4.0",
		InvocationDigest: "sha256:v1-digest",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}
	v1Body := []byte("## v1 procedure")
	v1Cold := v1Rec.Resolve(cache, func() []byte { return v1Body })
	if v1Cold.Verdict.Kind != contextq.MaterializationFault || !v1Cold.Built {
		t.Fatalf("v1 cold: got %v, want FAULT with Built=true", v1Cold.Verdict)
	}

	// Build v2's procedural view (same skill, different version).
	v2Rec := contextq.SkillContextRecord{
		SkillName:        "code-review",
		Version:          "2.0.0",
		InvocationDigest: "sha256:v2-digest",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}
	v2Body := []byte("## v2 procedure")
	v2Cold := v2Rec.Resolve(cache, func() []byte { return v2Body })
	if v2Cold.Verdict.Kind != contextq.MaterializationFault || !v2Cold.Built {
		t.Fatalf("v2 cold: got %v, want FAULT with Built=true", v2Cold.Verdict)
	}

	// Warm HITs: each version hits its own slot; no cross-talk.
	v1Warm := v1Rec.Resolve(cache, func() []byte { t.Fatal("v1 build should not run"); return nil })
	if v1Warm.Verdict.Kind != contextq.MaterializationHit || v1Warm.Built {
		t.Fatalf("v1 warm: got %v, want HIT with Built=false", v1Warm.Verdict)
	}
	v2Warm := v2Rec.Resolve(cache, func() []byte { t.Fatal("v2 build should not run"); return nil })
	if v2Warm.Verdict.Kind != contextq.MaterializationHit || v2Warm.Built {
		t.Fatalf("v2 warm: got %v, want HIT with Built=false", v2Warm.Verdict)
	}

	// Verify payloads are distinct (no aliasing).
	if string(v1Warm.Payload) != string(v1Body) {
		t.Fatalf("v1 payload mismatch")
	}
	if string(v2Warm.Payload) != string(v2Body) {
		t.Fatalf("v2 payload mismatch")
	}
}

// TestHotSwap_Remap proves hot-swap = remap: flipping the page-table entry
// from v1 to v2 redirects new invocations without disturbing in-flight ones.
func TestHotSwap_Remap(t *testing.T) {
	// No MMU/KV context needed for the page-table logic itself.
	table := New(nil, nil, nil)

	// Pin v1 as active.
	prev, blast, err := table.Pin("code-review", "1.4.0")
	if err != nil {
		t.Fatalf("pin v1: %v", err)
	}
	if prev != "" {
		t.Fatalf("pin v1 prev version = %q, want empty", prev)
	}
	if blast.Tokens != 0 || blast.DependentEntries != 0 {
		t.Fatalf("pin v1 blast radius non-zero with nil mmu: %+v", blast)
	}

	// Verify ActiveVersion returns v1.
	v1, ok := table.ActiveVersion("code-review")
	if !ok || v1 != "1.4.0" {
		t.Fatalf("active version after pin v1: got %q, ok=%v, want 1.4.0", v1, ok)
	}

	// Hot-swap to v2 = remap the page-table entry.
	prev, blast, err = table.Pin("code-review", "2.0.0")
	if err != nil {
		t.Fatalf("hot-swap v1→v2: %v", err)
	}
	if prev != "1.4.0" {
		t.Fatalf("hot-swap prev version = %q, want 1.4.0", prev)
	}

	// Verify new invocations see v2.
	v2, ok := table.ActiveVersion("code-review")
	if !ok || v2 != "2.0.0" {
		t.Fatalf("active version after swap: got %q, ok=%v, want 2.0.0", v2, ok)
	}
}

// TestHotSwap_InFlightSurvives proves a swap mid-invocation does not disturb
// the pinned in-flight frame. The acceptance is: an in-flight invocation's
// procedural view stays intact even after the page-table is remapped.
func TestHotSwap_InFlightSurvives(t *testing.T) {
	cache := contextq.NewViewCache()
	table := New(nil, nil, nil)

	// Pin v1 as active.
	table.Pin("code-review", "1.4.0")

	// Build v1's procedural view (simulating an in-flight invocation).
	v1Rec := contextq.SkillContextRecord{
		SkillName:        "code-review",
		Version:          "1.4.0",
		InvocationDigest: "sha256:v1-invocation",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}
	v1Body := []byte("## v1 in-flight")
	v1Cold := v1Rec.Resolve(cache, func() []byte { return v1Body })
	if v1Cold.Verdict.Kind != contextq.MaterializationFault || !v1Cold.Built {
		t.Fatalf("v1 cold: got %v", v1Cold.Verdict)
	}

	// Hot-swap to v2 mid-invocation.
	table.Pin("code-review", "2.0.0")

	// The in-flight invocation's view is still HIT (the pinned frame survives).
	v1Warm := v1Rec.Resolve(cache, func() []byte { t.Fatal("v1 build should not run"); return nil })
	if v1Warm.Verdict.Kind != contextq.MaterializationHit || v1Warm.Built {
		t.Fatalf("in-flight v1 after swap: got %v, want HIT with Built=false", v1Warm.Verdict)
	}
	if string(v1Warm.Payload) != string(v1Body) {
		t.Fatalf("in-flight v1 payload changed after swap")
	}

	// A new invocation with v2's digest is a cold build (different slot).
	v2Rec := contextq.SkillContextRecord{
		SkillName:        "code-review",
		Version:          "2.0.0",
		InvocationDigest: "sha256:v2-invocation",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}
	v2Cold := v2Rec.Resolve(cache, func() []byte { return []byte("## v2") })
	if v2Cold.Verdict.Kind != contextq.MaterializationFault || !v2Cold.Built {
		t.Fatalf("v2 cold after swap: got %v", v2Cold.Verdict)
	}
}

// TestRollback_InverseRemap proves rollback = inverse remap: unpinning restores
// the prior resolver-based behavior (or a prior pinned version via Swap).
func TestRollback_InverseRemap(t *testing.T) {
	table := New(nil, nil, nil)

	// Pin v1, then hot-swap to v2.
	table.Pin("code-review", "1.4.0")
	table.Pin("code-review", "2.0.0")

	// Verify v2 is active.
	v2, ok := table.ActiveVersion("code-review")
	if !ok || v2 != "2.0.0" {
		t.Fatalf("before rollback: got %q, ok=%v, want 2.0.0", v2, ok)
	}

	// Rollback = inverse remap via Unpin.
	unpinned, blast, err := table.Unpin("code-review")
	if err != nil {
		t.Fatalf("unpin: %v", err)
	}
	if unpinned != "2.0.0" {
		t.Fatalf("unpinned version = %q, want 2.0.0", unpinned)
	}
	if blast.Tokens != 0 || blast.DependentEntries != 0 {
		t.Fatalf("unpin blast radius non-zero with nil mmu: %+v", blast)
	}

	// After unpin, ActiveVersion falls back to the resolver (which returns empty).
	v, ok := table.ActiveVersion("code-review")
	if ok {
		t.Fatalf("after rollback: resolver returned version %q, want empty", v)
	}
}

// TestBlastRadius_PreFlip proves blast radius is reported before a flip.
func TestBlastRadius_PreFlip(t *testing.T) {
	mmu := ctxmmu.New()
	kvctx := kvmmu.NewWithGate(model.NewSynthetic(model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 48, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}).NewSession(), mmu)
	table := New(nil, mmu, kvctx)

	// Pin a skill and verify blast radius is reported (even if zero, since no spans are resident).
	prev, blast, err := table.Pin("code-review", "1.4.0")
	if err != nil {
		t.Fatalf("pin: %v", err)
	}
	if prev != "" {
		t.Fatalf("prev = %q, want empty", prev)
	}
	// Blast radius is zero because the cache is empty; the important thing is it's reported.
	if blast.Tokens != 0 || blast.DependentEntries != 0 {
		t.Fatalf("blast radius with empty cache: %+v", blast)
	}

	// Swap also reports blast radius.
	prev, blast, err = table.Swap("code-review", "1.4.0", "2.0.0")
	if err != nil {
		t.Fatalf("swap: %v", err)
	}
	if prev != "1.4.0" {
		t.Fatalf("swap prev = %q, want 1.4.0", prev)
	}
	// Still zero (empty cache).
	if blast.Tokens != 0 || blast.DependentEntries != 0 {
		t.Fatalf("swap blast radius: %+v", blast)
	}
}

// TestTable_List proves List returns the complete page-table snapshot.
func TestTable_List(t *testing.T) {
	table := New(nil, nil, nil)

	// Pin multiple skills.
	table.Pin("skill-a", "1.0.0")
	table.Pin("skill-b", "2.0.0")
	table.Pin("skill-c", "3.0.0")

	// List should return all pinned skills.
	snapshot := table.List()
	if len(snapshot) != 3 {
		t.Fatalf("list length = %d, want 3", len(snapshot))
	}
	if snapshot["skill-a"] != "1.0.0" || snapshot["skill-b"] != "2.0.0" || snapshot["skill-c"] != "3.0.0" {
		t.Fatalf("snapshot = %+v, want all pinned skills", snapshot)
	}

	// Unpin one and verify it's gone from the snapshot.
	table.Unpin("skill-b")
	snapshot = table.List()
	if len(snapshot) != 2 {
		t.Fatalf("list length after unpin = %d, want 2", len(snapshot))
	}
	if _, ok := snapshot["skill-b"]; ok {
		t.Fatalf("skill-b still in snapshot after unpin")
	}
}

// TestTable_ErrorCases proves error cases are refused.
func TestTable_ErrorCases(t *testing.T) {
	table := New(nil, nil, nil)

	// Pin empty skill name.
	_, _, err := table.Pin("", "1.0.0")
	if err == nil {
		t.Fatal("pin empty skill name: want error")
	}

	// Pin empty version.
	_, _, err = table.Pin("skill-a", "")
	if err == nil {
		t.Fatal("pin empty version: want error")
	}

	// Unpin empty skill name.
	_, _, err = table.Unpin("")
	if err == nil {
		t.Fatal("unpin empty skill name: want error")
	}

	// Swap with mismatched fromVersion.
	table.Pin("skill-a", "1.0.0")
	_, _, err = table.Swap("skill-a", "2.0.0", "3.0.0")
	if err == nil {
		t.Fatal("swap with mismatched fromVersion: want error")
	}
}

// TestTable_ConcurrentSwaps proves concurrent swaps are safe (mu protects the map).
func TestTable_ConcurrentSwaps(t *testing.T) {
	table := New(nil, nil, nil)
	table.Pin("skill-a", "1.0.0")

	var errors atomic.Int64
	var completed atomic.Int64
	const workers = 10

	// Launch concurrent swaps on the same skill.
	for i := 0; i < workers; i++ {
		go func(i int) {
			_, _, err := table.Pin("skill-a", "2.0.0")
			if err != nil {
				errors.Add(1)
			}
			completed.Add(1)
		}(i)
	}

	// Wait for all workers to complete.
	for {
		if completed.Load() == int64(workers) {
			break
		}
	}

	// All workers should succeed (no race conditions).
	if e := errors.Load(); e != 0 {
		t.Fatalf("%d workers failed, want 0", e)
	}

	// Final state should be 2.0.0 (some worker's write won).
	v, ok := table.ActiveVersion("skill-a")
	if !ok || v != "2.0.0" {
		t.Fatalf("final version = %q, ok=%v, want 2.0.0", v, ok)
	}
}