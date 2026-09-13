//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type metalQ8OrderCloser struct {
	wantLive int
	closed   bool
}

func (c *metalQ8OrderCloser) Close() error {
	if got := metalgemm.LiveQ8Weights(); got != c.wantLive {
		return fmt.Errorf("checkpoint owner closed before native handles: live=%d want=%d", got, c.wantLive)
	}
	c.closed = true
	return nil
}

func TestMetalQ8AliasParityIdentityReleaseAndSlotReuse(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ8Weights()
	qt := quantizeQ8(randomVecF(64*128, 9044), 64, 128)
	qv := quantizeVecQ8(randomVecF(128, 9045))
	w := metalgemm.AliasQ8(qt.q, qt.d, qt.out, qt.in)
	if w == nil || !w.NoCopy() {
		t.Fatal("page-aligned Q8 owner did not produce a no-copy handle")
	}
	firstID := w.ID()
	got := make([]float32, qt.out)
	w.GEMV(qv.q, qv.d, got)
	ref := qMatRows(qt, qv)
	if cos, rel := cosineAndMaxRel(ref, got); cos < 0.9999 || rel > 5e-3 {
		t.Fatalf("alias GEMV parity cos=%g rel=%g", cos, rel)
	}
	panel := quantizeBatchPanel(randomVecF(5*qt.in, 9046), 5, qt.in)
	gemm := make([]float32, 5*qt.out)
	w.GEMM(panel.q, panel.d, 5, gemm)
	if cos, rel := cosineAndMaxRel(qGemm8(qt, panel), gemm); cos < 0.999 || rel > 2e-2 {
		t.Fatalf("alias GEMM parity cos=%g rel=%g", cos, rel)
	}
	qt2 := quantizeQ8(randomVecF(48*128, 9047), 48, 128)
	wGroup := metalgemm.AliasQ8(qt2.q, qt2.d, qt2.out, qt2.in)
	if wGroup == nil {
		t.Fatal("second no-copy group handle unavailable")
	}
	group := metalgemm.GEMVGroupQ8([]*metalgemm.Q8Weight{w, wGroup}, qv.q, qv.d)
	if len(group) != 2 {
		t.Fatalf("alias GEMV group outputs=%d want 2", len(group))
	}
	if cos, rel := cosineAndMaxRel(qMatRows(qt2, qv), group[1]); cos < 0.9999 || rel > 5e-3 {
		t.Fatalf("alias grouped GEMV parity cos=%g rel=%g", cos, rel)
	}
	wGroup.Release()
	before := append([]float32(nil), got...)
	for i := 0; i < 32; i++ {
		qt.q[i] = -qt.q[i]
	}
	w.GEMV(qv.q, qv.d, got)
	if cos, rel := cosineAndMaxRel(qMatRows(qt, qv), got); cos < 0.9999 || rel > 5e-3 {
		t.Fatalf("mutated owner was not read byte-identically cos=%g rel=%g", cos, rel)
	}
	if reflectFloat32Equal(before, got) {
		t.Fatal("Metal result ignored an in-place owner mutation; alias appears copied")
	}
	w.Release()
	w.Release()
	if w.ID() != -1 || w.NoCopy() || metalgemm.LiveQ8Weights() != base {
		t.Fatalf("release left stale state: id=%d nocopy=%v live=%d base=%d", w.ID(), w.NoCopy(), metalgemm.LiveQ8Weights(), base)
	}
	w2 := metalgemm.AliasQ8(qt.q, qt.d, qt.out, qt.in)
	if w2 == nil {
		t.Fatal("slot-reuse alias failed")
	}
	defer w2.Release()
	if w2.ID() != firstID {
		t.Fatalf("released slot not reused: got %d want %d", w2.ID(), firstID)
	}
}

func reflectFloat32Equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func tinyExactQwen38Q8Model(t *testing.T) *Model {
	t.Helper()
	m := &Model{Cfg: exactQwen38Q8Config(), q8w: make(map[string]*q8Tensor)}
	names, err := qwen38MetalQ8RuntimeNames(m.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, name := range names {
		qt := newQ8Tensor(1, 32, 1)
		qt.d[0] = 1
		qt.q[0] = int8(i%127 + 1)
		m.q8w[name] = qt
	}
	return m
}

func TestMetalQ8ExactPromotionRollbackConcurrencyAndTeardown(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	t.Setenv("FAK_METAL_Q8_UPLOAD", "")
	base := metalgemm.LiveQ8Weights()

	broken := tinyExactQwen38Q8Model(t)
	names, _ := qwen38MetalQ8RuntimeNames(broken.Cfg)
	bad := newQ8Tensor(2, 32, 1)
	bad.d[0] = 1
	bad.q = bad.q[1:33] // full logical length, deliberately not page aligned
	bad.out = 1
	broken.q8w[names[113]] = bad
	if err := broken.promoteMetalQ8Residency(); err == nil {
		t.Fatal("partial unavailable residency was admitted")
	}
	if metalgemm.LiveQ8Weights() != base {
		t.Fatalf("rollback leaked native slots: live=%d base=%d", metalgemm.LiveQ8Weights(), base)
	}
	if broken.q8w[names[0]].q[0] == 0 || broken.q8w[names[113]].d[0] != 1 {
		t.Fatal("rollback damaged CPU-safe backing")
	}

	m := tinyExactQwen38Q8Model(t)
	closer := &metalQ8OrderCloser{wantLive: base}
	m.SetWeightCloser(closer)
	s := m.NewSession()
	const callers = 12
	errCh := make(chan error, callers)
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); errCh <- m.promoteMetalQ8Residency() }()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent promotion: %v", err)
		}
	}
	if got := metalgemm.LiveQ8Weights(); got != base+272 {
		t.Fatalf("publication allocated %d slots, want exactly 272 over base %d", got-base, base)
	}
	if err := m.CloseWeights(); err == nil {
		t.Fatal("CloseWeights released aliases while a session remained")
	}
	if got := metalgemm.LiveQ8Weights(); got != base+272 {
		t.Fatalf("deferred close changed native residency: live=%d", got)
	}
	s.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatalf("completed CloseWeights: %v", err)
	}
	if got := metalgemm.LiveQ8Weights(); got != base {
		t.Fatalf("teardown leaked native slots: live=%d base=%d", got, base)
	}
	if !closer.closed {
		t.Fatal("checkpoint owner was not closed after native teardown")
	}
}

// TestEagerMetalQ8ResidencyPromotesAtLoad pins part A: an exact Qwen3.8 hybrid on Apple Silicon
// publishes its full 272-projection no-copy Q8 band at LOAD time (before any prefill), so the
// startup probe reads a non-zero metal_live_q8_weights and the first request is fully resident.
func TestEagerMetalQ8ResidencyPromotesAtLoad(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	t.Setenv("FAK_METAL_Q8_UPLOAD", "")
	base := metalgemm.LiveQ8Weights()
	m := tinyExactQwen38Q8Model(t)
	if err := m.EagerMetalQ8Residency(); err != nil {
		t.Fatalf("eager promotion declined on an exact hybrid: %v", err)
	}
	if got := metalgemm.LiveQ8Weights(); got != base+272 {
		t.Fatalf("eager promotion allocated %d slots, want 272 over base %d", got-base, base)
	}
	if err, attempted := m.MetalQ8ResidencyError(); attempted || err != nil {
		t.Fatalf("successful promotion reported attempted=%v err=%v", attempted, err)
	}
	// Idempotent: a second eager call must not double-publish.
	if err := m.EagerMetalQ8Residency(); err != nil {
		t.Fatalf("second eager promotion: %v", err)
	}
	if got := metalgemm.LiveQ8Weights(); got != base+272 {
		t.Fatalf("repeat eager promotion double-published: live=%d want %d", got, base+272)
	}
	m.releaseMetalQ8Residency()
	if got := metalgemm.LiveQ8Weights(); got != base {
		t.Fatalf("teardown leaked slots: live=%d base=%d", got, base)
	}
}

// TestEagerMetalQ8ResidencyFailureSurfacesReason pins part B: when the no-copy band cannot be built
// (here a deliberately non-page-aligned owner), the eager call returns a fail-closed reason and
// MetalQ8ResidencyError reports it as attempted — a metal_live_q8_weights=0 is never silent.
func TestEagerMetalQ8ResidencyFailureSurfacesReason(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	t.Setenv("FAK_METAL_Q8_UPLOAD", "")
	base := metalgemm.LiveQ8Weights()
	m := tinyExactQwen38Q8Model(t)
	names, _ := qwen38MetalQ8RuntimeNames(m.Cfg)
	bad := newQ8Tensor(2, 32, 1)
	bad.d[0] = 1
	bad.q = bad.q[1:33] // full logical length, deliberately not page aligned
	bad.out = 1
	m.q8w[names[113]] = bad

	err := m.EagerMetalQ8Residency()
	if err == nil {
		t.Fatal("eager promotion admitted an unbuildable no-copy band")
	}
	var unavailable *MetalQ8ResidencyUnavailableError
	if !errors.As(err, &unavailable) || unavailable.Reason == "" {
		t.Fatalf("eager failure not a fail-closed reason: %T %v", err, err)
	}
	stored, attempted := m.MetalQ8ResidencyError()
	if !attempted || stored == nil {
		t.Fatalf("declined promotion not surfaced: attempted=%v err=%v", attempted, stored)
	}
	if metalgemm.LiveQ8Weights() != base {
		t.Fatalf("failed eager promotion leaked native slots: live=%d base=%d", metalgemm.LiveQ8Weights(), base)
	}
}

// TestEagerMetalQ8ResidencyNoopWhenNotExact proves the guard: a non-exact model is a no-op (no
// slots, no error), so callers may invoke the hook unconditionally on any Metal-armed model.
func TestEagerMetalQ8ResidencyNoopWhenNotExact(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	base := metalgemm.LiveQ8Weights()
	m := &Model{Cfg: Config{ModelType: "llama", NumLayers: 2, LayerTypes: []string{"full_attention", "full_attention"}}}
	if err := m.EagerMetalQ8Residency(); err != nil {
		t.Fatalf("non-exact model eager hook returned an error: %v", err)
	}
	if got := metalgemm.LiveQ8Weights(); got != base {
		t.Fatalf("non-exact model allocated %d Q8 slots", got-base)
	}
}
