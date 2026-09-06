package residency

import (
	"errors"
	"strconv"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// tinyCfg returns a minimal valid model.Config for synthetic test models.
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

// TestBudgetNeverExceeded verifies that resident weight bytes never exceed the configured budget
// and that resident count matches Len().
func TestBudgetNeverExceeded(t *testing.T) {
	r := New(500)
	for i, id := range []string{"a", "b", "c", "d", "e"} {
		if _, err := r.Admit(polymodel.ModelID(id), newModel(t), int64(100+i*40), "fam", "", false); err != nil {
			t.Fatalf("Admit %s: %v", id, err)
		}
		if r.Used() > r.Budget() {
			t.Fatalf("after %s: used %d > budget %d", id, r.Used(), r.Budget())
		}
		if r.Len() != len(r.Resident()) {
			t.Fatalf("Len() %d != len(Resident()) %d", r.Len(), len(r.Resident()))
		}
	}
}

// TestLRUEvictsColdestUnpinned verifies that over-budget admission evicts coldest unpinned
// models in LRU order and returns their weight handles.
func TestLRUEvictsColdestUnpinned(t *testing.T) {
	r := New(200)
	mA, mB := newModel(t), newModel(t)
	if _, err := r.Admit("A", mA, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("B", mB, 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	if !r.Touch("A") {
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
		t.Fatal("evicted handle is not B's *model.Model")
	}
	if _, ok := r.Get("B"); ok {
		t.Fatal("B still resident after eviction")
	}
	if _, ok := r.Get("A"); !ok {
		t.Fatal("A (hot) was evicted instead of B (cold)")
	}
}

// TestPinnedNeverEvicted verifies that pinned models are exempt from LRU eviction and
// admissions failing due to pinned capacity leave state unchanged.
func TestPinnedNeverEvicted(t *testing.T) {
	r := New(150)
	if _, err := r.Admit("P", newModel(t), 100, "fam", "", true); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("Q", newModel(t), 50, "fam", "", false); err != nil {
		t.Fatal(err)
	}
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
	_, err = r.Admit("S", newModel(t), 60, "fam", "", false)
	if !errors.Is(err, polymodel.ErrPinnedNoRoom) {
		t.Fatalf("expected ErrPinnedNoRoom, got %v", err)
	}
	if r.Len() != 2 || r.Used() != 150 {
		t.Fatalf("failed admit mutated state: len=%d used=%d", r.Len(), r.Used())
	}
}

// TestAdmitAllOrNothing verifies that failed admission leaves the resident set unchanged.
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

// TestReAdmitIsTouch verifies that re-admitting an existing model updates recency without
// duplicating or evicting entries.
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

// TestEvictHandBack verifies that Evict unbinds the model and returns its weight handle.
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

// TestNilWeightsRejected verifies that Admit rejects nil weight models with ErrNilWeights.
func TestNilWeightsRejected(t *testing.T) {
	r := New(1000)
	if _, err := r.Admit("N", nil, 10, "fam", "", false); !errors.Is(err, ErrNilWeights) {
		t.Fatalf("expected ErrNilWeights, got %v", err)
	}
}

// TestDescriptorRoundTrip verifies that descriptor metadata survives admission.
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

// TestSetBudgetShrinkPagesOutLRU verifies that decreasing budget evicts coldest unpinned
// models in LRU order.
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

// TestSetBudgetGrowPagesOutNothing verifies that increasing budget does not evict models.
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

// TestSetBudgetPinnedOverflowRefused verifies that shrinking budget below pinned footprint
// fails without mutating state.
func TestSetBudgetPinnedOverflowRefused(t *testing.T) {
	r := New(200)
	if _, err := r.Admit("P", newModel(t), 120, "fam", "", true); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Admit("Q", newModel(t), 60, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	before, ln := r.Used(), r.Len()
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

// TestNegativeBudgetClamped verifies that negative budgets clamp to zero on New and SetBudget.
func TestNegativeBudgetClamped(t *testing.T) {
	r := New(-500)
	if r.Budget() != 0 {
		t.Fatalf("expected budget 0, got %d", r.Budget())
	}
	_, err := r.Admit("A", newModel(t), 100, "fam", "", false)
	if !errors.Is(err, polymodel.ErrTooLarge) {
		t.Fatalf("expected ErrTooLarge, got %v", err)
	}

	r2 := New(200)
	if _, err := r2.Admit("B", newModel(t), 100, "fam", "", false); err != nil {
		t.Fatal(err)
	}
	evicted, err := r2.SetBudget(-100)
	if err != nil {
		t.Fatalf("SetBudget negative: %v", err)
	}
	if r2.Budget() != 0 {
		t.Fatalf("expected clamped budget 0, got %d", r2.Budget())
	}
	if len(evicted) != 1 || evicted[0].ID != "B" {
		t.Fatalf("expected B evicted, got %v", evicted)
	}
	if r2.Len() != 0 || r2.Used() != 0 {
		t.Fatalf("expected empty manager, len=%d used=%d", r2.Len(), r2.Used())
	}
}

// TestNonResidentOperations verifies that operations on missing IDs return zero values without mutation.
func TestNonResidentOperations(t *testing.T) {
	r := New(500)
	if _, ok := r.Get("missing"); ok {
		t.Fatal("expected Get on missing to return false")
	}
	if ok := r.Touch("missing"); ok {
		t.Fatal("expected Touch on missing to return false")
	}
	if _, ok := r.Descriptor("missing"); ok {
		t.Fatal("expected Descriptor on missing to return false")
	}
	if _, ok := r.Evict("missing"); ok {
		t.Fatal("expected Evict on missing to return false")
	}
	if r.Len() != 0 || len(r.Resident()) != 0 {
		t.Fatalf("state mutated: len=%d resident=%v", r.Len(), r.Resident())
	}
}

// TestConcurrentAdmit verifies concurrent safety and budget enforcement under contention.
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
