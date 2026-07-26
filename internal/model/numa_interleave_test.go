package model

import "testing"

// These are the failure-first tests for the #4974 placement half. compute's
// decode_interleave_test.go already pins the VERDICT (does this host want interleave, and is
// it overrideable); what was unpinned is the model-side half — WHICH resident slabs the
// verdict gets applied to. That set is derived from the loaded model's shape, and getting it
// wrong is silent: an omitted store keeps its default first-touch node-0 placement while the
// label still reports "applied", so the ordinary decode path would under-reproduce the
// witnessed regime with nothing to show for it.

// regionSet indexes the collected regions by backing-array base pointer so a test can assert
// coverage and exactly-once placement without depending on map iteration order.
func regionSet(t *testing.T, m *Model) map[*byte]int {
	t.Helper()
	out := make(map[*byte]int)
	for _, r := range m.residentDecodeRegions() {
		if len(r) == 0 {
			t.Fatalf("residentDecodeRegions returned an empty region: mbind would be called on nothing")
		}
		out[&r[0]]++
	}
	return out
}

func q4kSlab(n int) *q4kTensor {
	return &q4kTensor{out: 1, in: qkK, nblk: 1, raw: make([]byte, n)}
}

func kquantSlab(n int) *kQuantTensor {
	return &kQuantTensor{out: 1, in: qkK, nblk: 1, kind: kindQ6K, raw: make([]byte, n)}
}

// TestResidentDecodeRegionsCoversEveryStreamedRawStore is the core #4974 guarantee: every
// resident raw store the CPU decode GEMV streams must be handed to the interleave apply. A
// q4_k_m artifact is a MIXTURE — the Q4_K majority lands in q4kw, and the Q5_K/Q6_K minority
// (commonly ffn_down / lm_head, plus mixed-quant routed experts) lands in kqw, whose bytes
// kQuantMatRows streams every decode step through the same q4kDecodeWorkers pool. Placing
// only q4kw leaves the kqw slabs pinned to the loader's node.
func TestResidentDecodeRegionsCoversEveryStreamedRawStore(t *testing.T) {
	q4k := q4kSlab(128)
	kq := kquantSlab(64)
	m := &Model{
		q4kw: map[string]*q4kTensor{"model.layers.0.mlp.gate_proj.weight": q4k},
		kqw:  map[string]*kQuantTensor{"model.layers.0.mlp.down_proj.weight": kq},
	}

	got := regionSet(t, m)
	if got[&q4k.raw[0]] != 1 {
		t.Errorf("resident Q4_K slab placed %d times, want exactly 1", got[&q4k.raw[0]])
	}
	if got[&kq.raw[0]] != 1 {
		t.Errorf("resident k-quant (Q6_K) slab placed %d times, want exactly 1: the q4_k_m "+
			"Q5_K/Q6_K minority is streamed by kQuantMatRows on the same decode path, so "+
			"leaving it out keeps those bytes on the loader's first-touch NUMA node", got[&kq.raw[0]])
	}
	if len(got) != 2 {
		t.Errorf("collected %d distinct regions, want 2 (one per resident raw store)", len(got))
	}
}

// TestResidentDecodeRegionsPlacesAliasedHeadOnce pins the dedup: a raw lm_head reachable
// through BOTH q4kw and the separately pinned q4khead is one allocation, and mbind'ing it
// twice would double-count regions in the witness label for no placement gain.
func TestResidentDecodeRegionsPlacesAliasedHeadOnce(t *testing.T) {
	head := q4kSlab(96)
	other := q4kSlab(32)
	m := &Model{
		q4kw:    map[string]*q4kTensor{"lm_head.weight": head, "model.layers.0.mlp.up_proj.weight": other},
		q4khead: head,
	}

	got := regionSet(t, m)
	if got[&head.raw[0]] != 1 {
		t.Errorf("aliased lm_head slab placed %d times, want exactly 1", got[&head.raw[0]])
	}
	if len(got) != 2 {
		t.Errorf("collected %d distinct regions, want 2 (aliased head must not double-count)", len(got))
	}
}

// TestResidentDecodeRegionsSkipsUnbackedTensors pins the freed/absent cases: a nil map entry
// and a tensor whose raw was dropped (FAK_Q4K_FREE_CPU single-residency after a device upload)
// must not reach mbind, which would fail on a zero-length region.
func TestResidentDecodeRegionsSkipsUnbackedTensors(t *testing.T) {
	live := q4kSlab(48)
	m := &Model{
		q4kw: map[string]*q4kTensor{
			"live":   live,
			"nil":    nil,
			"freed":  {out: 1, in: qkK, nblk: 1, raw: nil},
			"empty":  {out: 1, in: qkK, nblk: 1, raw: []byte{}},
			"nilraw": {out: 1, in: qkK, nblk: 1},
		},
		kqw: map[string]*kQuantTensor{
			"kq-nil":   nil,
			"kq-freed": {out: 1, in: qkK, nblk: 1, kind: kindQ6K, raw: nil},
		},
	}

	regions := m.residentDecodeRegions()
	if len(regions) != 1 {
		t.Fatalf("collected %d regions, want 1: only the live slab is backed", len(regions))
	}
	if &regions[0][0] != &live.raw[0] {
		t.Errorf("collected region is not the live slab")
	}
}

// TestResidentDecodeRegionsEmptyModel pins that a model with no resident raw store (the f32 /
// Q8 paths never populate q4kw or kqw) yields nothing to place rather than panicking.
func TestResidentDecodeRegionsEmptyModel(t *testing.T) {
	if regions := (&Model{}).residentDecodeRegions(); len(regions) != 0 {
		t.Errorf("empty model collected %d regions, want 0", len(regions))
	}
}

// TestNUMAInterleaveLabelUnrunBeforeApply pins the witness contract: a decode/bench RESULT
// line must be able to tell "placement was never attempted" apart from "placement was
// attempted and skipped", so an unrun model can never be read as a clean skip.
func TestNUMAInterleaveLabelUnrunBeforeApply(t *testing.T) {
	m := &Model{}
	if got := m.NUMAInterleaveLabel(); got != "interleave=unrun" {
		t.Errorf("NUMAInterleaveLabel() before apply = %q, want %q", got, "interleave=unrun")
	}

	// ApplyDecodeNUMAInterleave is a no-op off linux/amd64 and on a single-node host, but it
	// must always cache a verdict, and the cached label must be what a later read returns.
	applied := m.ApplyDecodeNUMAInterleave()
	if applied == "" {
		t.Fatal("ApplyDecodeNUMAInterleave returned an empty label; the placement decision must always be reportable")
	}
	if applied == "interleave=unrun" {
		t.Error("ApplyDecodeNUMAInterleave cached the unrun sentinel; a run must record a real verdict")
	}
	if got := m.NUMAInterleaveLabel(); got != applied {
		t.Errorf("NUMAInterleaveLabel() = %q after apply returned %q; the cached verdict must match", got, applied)
	}
}
