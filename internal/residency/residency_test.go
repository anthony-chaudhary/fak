package residency

import (
	"errors"
	"strconv"
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

// TestSetBudgetShrinkPagesOutLRU shrinks the resident budget at runtime and asserts the
// coldest UNPINNED residents are paged out in LRU order with their weight handles handed
// back — the page-out signal polymodel.Pool.Resize alone cannot give (it returns only IDs).
func TestSetBudgetShrinkPagesOutLRU(t *testing.T) {
	r := New(300)
	mA, mB, mC := newModel(t), newModel(t), newModel(t)
	if _, err := r.Admit("A", mA, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("B", mB, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("C", mC, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	// Hotten B then C; A stays coldest. Shrinking 300→150 frees 150: evict A then B
	// (coldest-first), leaving C (100 <= 150).
	r.Touch("B")
	r.Touch("C")
	evicted, err := r.SetBudget(150)
	if err != nil {
		t.Fatalf("SetBudget: %v", err)
	}
	if len(evicted) != 2 || evicted[0].ID != "A" || evicted[1].ID != "B" {
		t.Fatalf("evicted = %v, want [A B] coldest-first", evicted)
	}
	if evicted[0].Weights != mA || evicted[1].Weights != mB {
		t.Fatal("paged-out handles lost their *model.Model binding")
	}
	if r.Budget() != 150 || r.Used() != 100 || r.Len() != 1 {
		t.Fatalf("after shrink: budget=%d used=%d len=%d, want 150/100/1", r.Budget(), r.Used(), r.Len())
	}
	if _, ok := r.Get("C"); !ok {
		t.Fatal("hot C was paged out")
	}
	if _, ok := r.Get("A"); ok {
		t.Fatal("A still bound after page-out")
	}
}

// TestSetBudgetGrowPagesOutNothing proves growing the budget evicts nothing and keeps every
// binding — the hot-add-headroom direction of the runtime knob.
func TestSetBudgetGrowPagesOutNothing(t *testing.T) {
	r := New(200)
	if _, err := r.Admit("A", newModel(t), 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	evicted, err := r.SetBudget(1000)
	if err != nil || len(evicted) != 0 {
		t.Fatalf("grow: err=%v evicted=%v, want none", err, evicted)
	}
	if r.Budget() != 1000 || r.Len() != 1 {
		t.Fatalf("after grow: budget=%d len=%d, want 1000/1", r.Budget(), r.Len())
	}
}

// TestSetBudgetPinnedOverflowRefused shrinks below the pinned footprint: no eviction can
// make room, so SetBudget refuses with ErrPinnedNoRoom and the resident set (and its
// bindings) are byte-for-byte unchanged — the runtime re-budget fails CLOSED.
func TestSetBudgetPinnedOverflowRefused(t *testing.T) {
	r := New(200)
	if _, err := r.Admit("P", newModel(t), 120, "fam", "", true); err != nil {
		t.Fatal(err) // pinned
	}
	if _, err := r.Admit("Q", newModel(t), 60, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	before, ln := r.Used(), r.Len()
	// Shrink to 100 < pinned 120: evicting all unpinned (Q=60) still leaves 120 > 100.
	if _, err := r.SetBudget(100); !errors.Is(err, polymodel.ErrPinnedNoRoom) {
		t.Fatalf("SetBudget below pinned footprint: err=%v, want ErrPinnedNoRoom", err)
	}
	if r.Used() != before || r.Len() != ln || r.Budget() != 200 {
		t.Fatalf("refused SetBudget mutated state: used=%d len=%d budget=%d", r.Used(), r.Len(), r.Budget())
	}
	if _, ok := r.Get("P"); !ok {
		t.Fatal("pinned P lost its binding on a refused resize")
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

// BenchmarkAdmit_SteadyStateEviction measures continuous Admit cycles under memory pressure
// where each admission evicts the coldest unpinned model via LRU and returns its weight handle.
func BenchmarkAdmit_SteadyStateEviction(b *testing.B) {
	const capacity = 10
	const poolSize = 20
	r := New(capacity * 100)
	models := make([]*model.Model, poolSize)
	ids := make([]polymodel.ModelID, poolSize)
	for i := 0; i < poolSize; i++ {
		models[i] = model.NewSynthetic(tinyCfg())
		ids[i] = polymodel.ModelID("bench-model-" + strconv.Itoa(i))
	}
	// Seed up to capacity so subsequent admissions trigger steady-state eviction.
	for i := 0; i < capacity; i++ {
		if _, err := r.Admit(ids[i], models[i], 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := (i + capacity) % poolSize
		if _, err := r.Admit(ids[idx], models[idx], 100, "fam", "", false); err != nil {
			b.Fatalf("admit %s: %v", ids[idx], err)
		}
	}
}

// BenchmarkAdmit_ReAdmitTouch measures re-admitting already resident models, exercising the
// Touch path through Admit without triggering evictions.
func BenchmarkAdmit_ReAdmitTouch(b *testing.B) {
	const count = 10
	r := New(count * 100)
	models := make([]*model.Model, count)
	ids := make([]polymodel.ModelID, count)
	for i := 0; i < count; i++ {
		models[i] = model.NewSynthetic(tinyCfg())
		ids[i] = polymodel.ModelID("bench-readmit-" + strconv.Itoa(i))
		if _, err := r.Admit(ids[i], models[i], 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % count
		if _, err := r.Admit(ids[idx], models[idx], 100, "fam", "", false); err != nil {
			b.Fatalf("re-admit: %v", err)
		}
	}
}

// BenchmarkGet_Hit measures hot weight handle lookup latency.
func BenchmarkGet_Hit(b *testing.B) {
	const count = 16
	r := New(count * 100)
	ids := make([]polymodel.ModelID, count)
	for i := 0; i < count; i++ {
		ids[i] = polymodel.ModelID("bench-get-" + strconv.Itoa(i))
		m := model.NewSynthetic(tinyCfg())
		if _, err := r.Admit(ids[i], m, 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%count]
		if _, ok := r.Get(id); !ok {
			b.Fatalf("get %s: not found", id)
		}
	}
}

// BenchmarkGet_Miss measures lookup latency on non-resident models.
func BenchmarkGet_Miss(b *testing.B) {
	const count = 16
	r := New(count * 100)
	for i := 0; i < count; i++ {
		id := polymodel.ModelID("bench-get-" + strconv.Itoa(i))
		m := model.NewSynthetic(tinyCfg())
		if _, err := r.Admit(id, m, 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}
	missing := polymodel.ModelID("bench-nonexistent")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := r.Get(missing); ok {
			b.Fatalf("expected miss")
		}
	}
}

// BenchmarkTouch measures LRU recency update latency for hot models.
func BenchmarkTouch(b *testing.B) {
	const count = 16
	r := New(count * 100)
	ids := make([]polymodel.ModelID, count)
	for i := 0; i < count; i++ {
		ids[i] = polymodel.ModelID("bench-touch-" + strconv.Itoa(i))
		m := model.NewSynthetic(tinyCfg())
		if _, err := r.Admit(ids[i], m, 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%count]
		if !r.Touch(id) {
			b.Fatalf("touch %s: failed", id)
		}
	}
}

// BenchmarkDescriptor measures metadata descriptor retrieval for resident models.
func BenchmarkDescriptor(b *testing.B) {
	const count = 16
	r := New(count * 100)
	ids := make([]polymodel.ModelID, count)
	for i := 0; i < count; i++ {
		ids[i] = polymodel.ModelID("bench-desc-" + strconv.Itoa(i))
		m := model.NewSynthetic(tinyCfg())
		if _, err := r.Admit(ids[i], m, 100, "fam", "sha256:digest", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%count]
		if _, ok := r.Descriptor(id); !ok {
			b.Fatalf("descriptor %s: not found", id)
		}
	}
}

// BenchmarkEvictAndAdmit measures an explicit eviction followed by immediate re-admission.
func BenchmarkEvictAndAdmit(b *testing.B) {
	r := New(1000)
	m := model.NewSynthetic(tinyCfg())
	id := polymodel.ModelID("bench-evict-admit")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Admit(id, m, 100, "fam", "", false); err != nil {
			b.Fatalf("admit: %v", err)
		}
		if _, ok := r.Evict(id); !ok {
			b.Fatalf("evict: failed")
		}
	}
}

// BenchmarkSetBudget_ShrinkGrow measures runtime dynamic resizing of the memory budget,
// causing batch eviction and handback during shrink followed by expansion.
func BenchmarkSetBudget_ShrinkGrow(b *testing.B) {
	const count = 10
	models := make([]*model.Model, count)
	ids := make([]polymodel.ModelID, count)
	for i := 0; i < count; i++ {
		models[i] = model.NewSynthetic(tinyCfg())
		ids[i] = polymodel.ModelID("bench-resize-" + strconv.Itoa(i))
	}
	r := New(count * 100)
	for i := 0; i < count; i++ {
		if _, err := r.Admit(ids[i], models[i], 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evicted, err := r.SetBudget(500)
		if err != nil || len(evicted) != 5 {
			b.Fatalf("shrink: %v, evicted: %d", err, len(evicted))
		}
		if _, err := r.SetBudget(1000); err != nil {
			b.Fatalf("grow: %v", err)
		}
		for _, e := range evicted {
			if _, err := r.Admit(e.ID, e.Weights, 100, "fam", "", false); err != nil {
				b.Fatalf("re-admit %s: %v", e.ID, err)
			}
		}
	}
}

// BenchmarkResidentIDs measures retrieval of all resident model IDs sorted deterministically.
func BenchmarkResidentIDs(b *testing.B) {
	const count = 16
	r := New(count * 100)
	for i := 0; i < count; i++ {
		id := polymodel.ModelID("bench-resident-" + strconv.Itoa(i))
		m := model.NewSynthetic(tinyCfg())
		if _, err := r.Admit(id, m, 100, "fam", "", false); err != nil {
			b.Fatalf("seed admit: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.Resident()
		if len(res) != count {
			b.Fatalf("unexpected resident count: %d", len(res))
		}
	}
}
