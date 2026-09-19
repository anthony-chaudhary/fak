package model

// v41_proj_matrows_test.go — the #2150 witness for the streaming-safe V4.1
// per-layer projection read.
//
// v41ProjF32 resolves a resident k-quant projection by expanding the WHOLE
// tensor to f32 (residentF32Mat's make([]float32, out*in)), an ~13x-for-Q2_K
// allocation charged to no ledger and absent from the serve memory plan. On the
// streamed arm that uncharged per-layer f32 expansion is transient against the
// resident streamed-expert tier cache, so the in-kernel warmup's resident set
// grows monotonically until the kernel OOM-kills the process.
//
// v41ProjMatRows reads the same weight through residentMatRows, whose k-quant
// arm reuses a single block-sized scratch buffer instead of materializing the
// tensor. These tests pin the three contracts this change must hold:
//
//  1. ALLOCATION: v41ProjMatRows allocates dramatically less than v41ProjF32
//     for a resident k-quant weight (the bounded-scratch property). This is the
//     RED symptom: on the parent commit v41ProjMatRows does not exist.
//  2. BYTE-IDENTITY on the f32-manifest path: v41ProjF32+matRows and
//     v41ProjMatRows produce bit-identical output, so the reduced fixture is
//     unchanged (p4 of the leaf).
//  3. FAIL-CLOSED: a projection absent from every store refuses with a typed
//     ErrV41ForwardStage naming the tensor (#13276 preserved) and does not
//     panic through residentMatRowsBase.

import (
	"errors"
	"runtime"
	"strings"
	"testing"
)

// TestV41ProjMatRowsBoundsF32Expansion is the #2150 acceptance witness: reading
// a large resident k-quant projection through v41ProjMatRows must NOT allocate
// the whole-tensor f32 expansion v41ProjF32 pays.
func TestV41ProjMatRowsBoundsF32Expansion(t *testing.T) {
	m := v41ReducedModel(t)

	// A projection large enough that the f32 expansion (out*in*4 bytes) dwarfs
	// the block-scratch path. in must be a multiple of 256 for Q2_K.
	const out, in = 1024, 2048
	name := layerName(0, "attn.wq_a.weight")
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(v41ResidentQ2KRaw(out, in), out, in, kindQ2K)
	x := make([]float32, in)
	for i := range x {
		x[i] = 1
	}

	measure := func(fn func()) uint64 {
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		fn()
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}

	// Warm both paths so one-time lazies are not counted.
	if _, ok := m.residentF32Mat(name); !ok {
		t.Fatalf("residentF32Mat(%s) not resident; fixture is wrong", name)
	}
	m.residentMatRows(name, x, out, in)

	f32Bytes := measure(func() { _, _ = m.residentF32Mat(name) })
	rowsBytes := measure(func() {
		if _, err := m.v41ProjMatRows(0, "attn.wq_a.weight", x, out, in); err != nil {
			t.Fatalf("v41ProjMatRows error = %v, want nil", err)
		}
	})

	// The f32 expansion allocates at least out*in*4 bytes; the streaming path
	// must allocate far less (a small per-row scratch + the out result vector).
	expand := uint64(out) * uint64(in) * 4
	if f32Bytes < expand {
		t.Fatalf("residentF32Mat allocated %d bytes, want >= the %d-byte whole-tensor f32 expansion; the fixture does not exercise the expansion", f32Bytes, expand)
	}
	if rowsBytes >= expand {
		t.Fatalf("v41ProjMatRows allocated %d bytes, want < the %d-byte whole-tensor f32 expansion; the streaming read is materializing the weight", rowsBytes, expand)
	}
	t.Logf("alloc f32=%d bytes rows=%d bytes expansion=%d bytes", f32Bytes, rowsBytes, expand)
}

// TestV41ProjMatRowsF32ManifestByteIdentical pins p4: on the f32 manifest the
// streaming read is exactly matRows(m.tensor(name), ...) — the pre-#2150 read.
func TestV41ProjMatRowsF32ManifestByteIdentical(t *testing.T) {
	m := v41ReducedModel(t)

	name := layerName(0, "attn.wq_a.weight")
	if !m.has(name) {
		t.Fatalf("reduced fixture is missing the f32 manifest entry %s", name)
	}
	shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
	out, in := shape[0], shape[1]
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.25
	}

	want := matRows(m.tensor(name), x, out, in)
	got, err := m.v41ProjMatRows(0, "attn.wq_a.weight", x, out, in)
	if err != nil {
		t.Fatalf("v41ProjMatRows error = %v, want nil", err)
	}
	if len(got) != len(want) {
		t.Fatalf("v41ProjMatRows len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("v41ProjMatRows[%d] = %v, want byte-identical %v", i, got[i], want[i])
		}
	}
}

// TestV41ProjMatRowsFailsClosedWhenAbsent pins #13276: a projection in no store
// refuses with a typed ErrV41ForwardStage naming the tensor, never a panic.
func TestV41ProjMatRowsFailsClosedWhenAbsent(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "attn.wq_a.weight")
	delete(m.manifest, name)
	if m.kqw != nil {
		delete(m.kqw, name)
	}

	x := make([]float32, m.Cfg.HiddenSize)
	_, err := m.v41ProjMatRows(0, "attn.wq_a.weight", x, m.Cfg.QLoraRank, m.Cfg.HiddenSize)
	if err == nil {
		t.Fatalf("v41ProjMatRows on an absent tensor returned nil error, want a typed refusal")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("v41ProjMatRows error = %v, want errors.Is(err, ErrV41ForwardStage)", err)
	}
	if !strings.Contains(err.Error(), name) {
		t.Fatalf("v41ProjMatRows error %q does not name the tensor %s", err.Error(), name)
	}
}
