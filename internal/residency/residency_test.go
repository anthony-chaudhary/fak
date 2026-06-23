package residency

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// tinyCfg is a minimal valid model.Config; NewSynthetic on it is cheap. The weights are
// meaningless (synthetic) — what is faithful is that each id binds a distinct, real
// *model.Model handle, which is exactly what the residency binding + page-out hand-back
// witness need. (Mirrors modelengine.SyntheticConfig's shape.)
func tinyCfg() model.Config {
	return model.Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
}

func newModel(t *testing.T) *model.Model {
	t.Helper()
	return model.NewSynthetic(tinyCfg())
}

// TestBudgetNeverExceeded drives an over-budget admit sequence and asserts the
// polymodel invariant — used <= budget — after every Admit. The residency layer
// inherits it by delegating the budget test to polymodel.Pool; this witness proves the
// delegation is wired (not just claimed).
func TestBudgetNeverExceeded(t *testing.T) {
	r := New(500)
	for i, id := range []string{"a", "b", "c", "d", "e"} {
		if _, err := r.Admit(polymodel.ModelID(id), newModel(t), int64(100+i*40), "fam", "", false); err != nil {
			t.Fatalf("Admit %s: %v", id, err)
		}
		if r.Used() > r.Budget() {
			t.Fatalf("after %s: used %d > budget %d", id, r.Used(), r.Budget())
		}
	}
}

// TestLRUEvictsColdestUnpinned proves a hot model survives an over-budget admit while a
// cold one is paged out, and that the evicted *model.Model handle is handed back (the
// page-out signal polymodel.Pool alone cannot give — it returns only IDs).
func TestLRUEvictsColdestUnpinned(t *testing.T) {
	r := New(200)
	mA, mB := newModel(t), newModel(t)
	if _, err := r.Admit("A", mA, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("B", mB, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if !r.Touch("A") { // A hot → B is the coldest
		t.Fatal("Touch A: not resident")
	}
	evicted, err := r.Admit("C", newModel(t), 100, "fam", "", false)
	if err != nil {
		t.Fatalf("Admit C: %v", err)
	}
	if len(evicted) != 1 || evicted[0].ID != "B" {
		t.Fatalf("expected [B] evicted, got %v", evicted)
	}
	if evicted[0].Weights != mB {
		t.Fatal("evicted handle is not B's *model.Model — page-out hand-back lost the binding")
	}
	if _, ok := r.Get("B"); ok {
		t.Fatal("B still resident after eviction")
	}
	if _, ok := r.Get("A"); !ok {
		t.Fatal("A (hot) was evicted instead of B (cold)")
	}
}

// TestPinnedNeverEvicted proves a pinned model is exempt from LRU eviction, and that an
// admit which would require dropping a pinned resident fails CLOSED (ErrPinnedNoRoom)
// leaving the resident set unchanged.
func TestPinnedNeverEvicted(t *testing.T) {
	r := New(150)
	if _, err := r.Admit("P", newModel(t), 100, "fam", "", true); err != nil {
		t.Fatal(err)
	} // pinned; 100/150
	if _, err := r.Admit("Q", newModel(t), 50, "fam", "", false); err != nil {
		t.Fatal(err)
	} // 150/150
	// R(50) fits by evicting Q (the only unpinned resident); P stays.
	evicted, err := r.Admit("R", newModel(t), 50, "fam", "", false)
	if err != nil {
		t.Fatalf("R should fit by evicting Q: %v", err)
	}
	if len(evicted) != 1 || evicted[0].ID != "Q" {
		t.Fatalf("expected [Q] evicted, got %v", evicted)
	}
	if _, ok := r.Get("P"); !ok {
		t.Fatal("pinned P was evicted")
	}
	// Now P(100,pinned)+R(50). S(60) needs 60; only R(50,unpinned) is evictable, freeing
	// 50 < 60 → ErrPinnedNoRoom, set unchanged.
	_, err = r.Admit("S", newModel(t), 60, "fam", "", false)
	if !errors.Is(err, polymodel.ErrPinnedNoRoom) {
		t.Fatalf("expected ErrPinnedNoRoom, got %v", err)
	}
	if r.Len() != 2 || r.Used() != 150 {
		t.Fatalf("failed admit mutated state: len=%d used=%d", r.Len(), r.Used())
	}
}

// TestAdmitAllOrNothing proves an erroring admit leaves the resident set byte-for-byte
// unchanged (no half-eviction): a too-large model neither evicts nor admits.
func TestAdmitAllOrNothing(t *testing.T) {
	r := New(100)
	if _, err := r.Admit("A", newModel(t), 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	before := r.Used()
	_, err := r.Admit("BIG", newModel(t), 1000, "fam", "", false)
	if !errors.Is(err, polymodel.ErrTooLarge) {
		t.Fatalf("expected ErrTooLarge, got %v", err)
	}
	if r.Used() != before || r.Len() != 1 {
		t.Fatalf("failed admit mutated state: used=%d len=%d", r.Used(), r.Len())
	}
	if _, ok := r.Get("BIG"); ok {
		t.Fatal("BIG admitted after an erroring admit")
	}
}

// TestReAdmitIsTouch proves re-admitting a resident id is a recency update, not a new
// entry — and that it evicts nothing.
func TestReAdmitIsTouch(t *testing.T) {
	r := New(300)
	if _, err := r.Admit("A", newModel(t), 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("B", newModel(t), 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	evicted, err := r.Admit("A", newModel(t), 100, "fam", "", false)
	if err != nil || len(evicted) != 0 {
		t.Fatalf("re-admit A should be a no-op Touch, got err=%v evicted=%v", err, evicted)
	}
	if r.Len() != 2 {
		t.Fatalf("re-admit changed len: %d", r.Len())
	}
}

// TestEvictHandBack proves explicit Evict returns the bound weight handle and clears
// residency, and is a no-op (false) on a non-resident id.
func TestEvictHandBack(t *testing.T) {
	r := New(1000)
	m := newModel(t)
	if _, err := r.Admit("X", m, 100, "fam", "d1", false); err != nil {
		t.Fatal(err)
	}
	got, ok := r.Evict("X")
	if !ok || got != m {
		t.Fatal("Evict did not return the bound handle")
	}
	if _, ok := r.Get("X"); ok {
		t.Fatal("X still resident after Evict")
	}
	if _, ok := r.Evict("X"); ok {
		t.Fatal("Evict of a non-resident id returned true")
	}
}

// TestNilWeightsRejected proves Admit requires a real weight handle (the binding is the
// point of this layer — a nil handle has nothing to bind).
func TestNilWeightsRejected(t *testing.T) {
	r := New(1000)
	if _, err := r.Admit("N", nil, 10, "fam", "", false); !errors.Is(err, ErrNilWeights) {
		t.Fatalf("expected ErrNilWeights, got %v", err)
	}
}

// TestDescriptorRoundTrip proves the family / prefixDigest / pinned / weightBytes keys
// survive the descriptor→weights binding (they are what cross-model prefill share and
// ensemble speculation key on).
func TestDescriptorRoundTrip(t *testing.T) {
	r := New(1000)
	if _, err := r.Admit("M", newModel(t), 100, "qwen", "digest42", true); err != nil {
		t.Fatal(err)
	}
	d, ok := r.Descriptor("M")
	if !ok {
		t.Fatal("M not resident")
	}
	if d.Family != "qwen" || d.PrefixDigest != "digest42" || !d.Pinned || d.WeightBytes != 100 {
		t.Fatalf("descriptor lost fields: %+v", d)
	}
}

// TestConcurrentAdmit proves the Manager is safe under concurrent admitters and that the
// budget invariant survives the race (run with -race). The mutex makes each Admit atomic;
// used can never exceed budget regardless of interleaving.
func TestConcurrentAdmit(t *testing.T) {
	r := New(400)
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := polymodel.ModelID(string(rune('a' + i%10)))
			_, _ = r.Admit(id, newModel(t), 100, "fam", "", false)
			_ = r.Touch(id)
		}(i)
	}
	wg.Wait()
	if r.Used() > r.Budget() {
		t.Fatalf("budget exceeded under concurrency: used %d > budget %d", r.Used(), r.Budget())
	}
}
