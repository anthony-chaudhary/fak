package model

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// embedding_tied_q6k_test.go — fak#13567: a tied Q6_K token table is held ONCE (kqw + an aliasing
// packed-embedding view) and serves both the row gather and the LM head. Proves (1) no f32 gather
// table and no duplicate Q8 head remain, (2) gathered rows equal the dequantized table, (3) the
// head equals an independent f32 reference projection, and (4) prefill, the decode loop and Step
// all run on the shared table and agree with each other.

func randomTiedQ6KRaw(t *testing.T, vocab, hidden int, seed int64) []byte {
	t.Helper()
	bb := kindQ6K.blockBytes()
	nblk := hidden / qkK
	raw := make([]byte, vocab*nblk*bb)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// Keep each block's f16 scale small and finite (same discipline as fillDownProjQ6KResident).
	for o := 0; o < vocab*nblk; o++ {
		raw[o*bb+bb-1] = 0x2C | (raw[o*bb+bb-1] & 0x03)
	}
	return raw
}

func dequantQ6KTableForTest(raw []byte, vocab, hidden int) []float32 {
	out := make([]float32, vocab*hidden)
	bb := kindQ6K.blockBytes()
	for b := 0; b < vocab*hidden/qkK; b++ {
		q6kDequantSuperBlock(out[b*qkK:(b+1)*qkK], raw[b*bb:(b+1)*bb])
	}
	return out
}

// tiedQ6KModelPair builds the legacy two-copy tied model (f32 gather table + native-Q8 head of
// the dequantized Q6_K values) and the packed model (one shared Q6_K table) over identical
// weights.
func tiedQ6KModelPair(t *testing.T) (legacy, packed *Model, raw []byte, table []float32) {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	vocab, hidden := cfg.VocabSize, cfg.HiddenSize
	raw = randomTiedQ6KRaw(t, vocab, hidden, 13567)
	table = dequantQ6KTableForTest(raw, vocab, hidden)

	legacy = NewSynthetic(cfg)
	legacy.Quantize()
	copy(legacy.tensor(tiedEmbeddingName), table)
	legacy.q8w[tiedEmbeddingName] = quantizeQ8(table, vocab, hidden)
	legacy.q8head = legacy.q8w[tiedEmbeddingName]

	packed = NewSynthetic(cfg)
	packed.Quantize()
	delete(packed.manifest, tiedEmbeddingName)
	delete(packed.q8w, tiedEmbeddingName)
	packed.q8head = nil
	if err := packed.attachTiedQ6KEmbedding([]int{vocab, hidden}, raw); err != nil {
		t.Fatalf("attachTiedQ6KEmbedding: %v", err)
	}
	return legacy, packed, raw, table
}

func TestTiedQ6KEmbeddingSharesOneResidentCopy(t *testing.T) {
	legacy, packed, raw, table := tiedQ6KModelPair(t)
	cfg := packed.Cfg
	if !packed.TiedQ6KEmbeddingShared() {
		t.Fatal("packed model must share one Q6_K table between gather and head")
	}
	if packed.HasF32(tiedEmbeddingName) || packed.HasQ8(tiedEmbeddingName) {
		t.Fatal("packed tied model retained an f32 gather table or a duplicate Q8 head")
	}
	if got := packed.kqHeadName(); got != tiedEmbeddingName {
		t.Fatalf("kqHeadName = %q, want the tied embedding", got)
	}
	if route := packed.LMHeadRoute(); route != "cpu-q6k" && route != "metal-q6k" {
		t.Fatalf("packed LMHeadRoute = %q, want cpu-q6k or metal-q6k", route)
	}
	if route := legacy.LMHeadRoute(); route != "cpu-q8" {
		t.Fatalf("legacy LMHeadRoute = %q, want cpu-q8", route)
	}

	r := packed.ResidentReport()
	if r.Q6KEmbedTensors != 1 || r.Q6KEmbedBytes != int64(len(raw)) || r.Q6KEmbedParams != int64(cfg.VocabSize*cfg.HiddenSize) {
		t.Fatalf("packed report Q6K embed = %d/%d/%d, want 1/%d/%d", r.Q6KEmbedTensors, r.Q6KEmbedBytes, r.Q6KEmbedParams, len(raw), cfg.VocabSize*cfg.HiddenSize)
	}
	if r.TiedEmbedF32Bytes != 0 || r.TiedHeadQ8Bytes != 0 {
		t.Fatalf("packed report tied two-copy bytes f32=%d q8=%d, want 0/0", r.TiedEmbedF32Bytes, r.TiedHeadQ8Bytes)
	}
	if r.KQuantTensors != 0 || r.KQuantBytes != 0 {
		t.Fatalf("shared tied table double-counted under kquant: %d tensors / %d bytes", r.KQuantTensors, r.KQuantBytes)
	}
	lr := legacy.ResidentReport()
	if lr.TiedEmbedF32Bytes != int64(len(table))*4 || lr.TiedHeadQ8Bytes == 0 {
		t.Fatalf("legacy report tied two-copy bytes f32=%d q8=%d, want %d/>0", lr.TiedEmbedF32Bytes, lr.TiedHeadQ8Bytes, len(table)*4)
	}
	if saved := lr.TotalResidentBytes - r.TotalResidentBytes; saved != lr.TiedEmbedF32Bytes+lr.TiedHeadQ8Bytes-int64(len(raw)) {
		t.Fatalf("resident saving = %d, want f32 %d + q8 %d - q6k %d", saved, lr.TiedEmbedF32Bytes, lr.TiedHeadQ8Bytes, len(raw))
	}
	if r.DecodeBytesPerToken != lr.DecodeBytesPerToken-lr.TiedHeadQ8Bytes+int64(len(raw)) {
		t.Fatalf("decode stream = %d, want the Q6_K head (%d) in place of the Q8 head (%d) from %d", r.DecodeBytesPerToken, len(raw), lr.TiedHeadQ8Bytes, lr.DecodeBytesPerToken)
	}
	replicated, expert, ok := packed.MoEResidentWeightBytes()
	if !ok || expert != 0 || replicated != r.TotalResidentBytes {
		t.Fatalf("MoEResidentWeightBytes = %d/%d/%v, want %d/0/true", replicated, expert, ok, r.TotalResidentBytes)
	}

	// Row gather equals the dequantized table, row for row.
	row := make([]float32, cfg.HiddenSize)
	for _, id := range []int{0, 1, 255, cfg.VocabSize - 1} {
		if err := packed.Q2KEmbedding.GatherRow(id, row, 1); err != nil {
			t.Fatal(err)
		}
		want := table[id*cfg.HiddenSize : (id+1)*cfg.HiddenSize]
		for i := range row {
			if row[i] != want[i] {
				t.Fatalf("gather row %d[%d] = %v, want %v", id, i, row[i], want[i])
			}
		}
	}

	// Head equals an independent f32 projection over the dequantized table (the int8 activation
	// quantization of the resident GEMV is the only residual).
	s := packed.NewSession()
	s.Q4K = true
	x := make([]float32, cfg.HiddenSize)
	rng := rand.New(rand.NewSource(5))
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	got := s.headResident(x)
	want := make([]float32, cfg.VocabSize)
	for o := range want {
		var acc float64
		for i, v := range table[o*cfg.HiddenSize : (o+1)*cfg.HiddenSize] {
			acc += float64(v) * float64(x[i])
		}
		want[o] = float32(acc)
	}
	assertCosineAtLeast(t, "tied Q6_K head vs f32 reference", want, got, 0.9999)
	if argmax(got) != argmax(want) {
		t.Fatalf("tied Q6_K head argmax = %d, want %d", argmax(got), argmax(want))
	}
}

func TestTiedQ6KEmbeddingPrefillDecodeMatchLegacy(t *testing.T) {
	legacy, packed, _, _ := tiedQ6KModelPair(t)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	// Reference on the packed model: per-token decode loop + resident head.
	ref := packed.NewSession()
	ref.Q4K = true
	var hidden []float32
	for _, id := range prompt {
		hidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	want := ref.headResident(hidden)

	got := packed.NewSession()
	got.Q4K = true
	gotLogits := got.Prefill(prompt)
	assertQuantLogitsClose(t, "packed tied prefill vs decode loop", want, gotLogits)

	base := legacy.NewSession()
	base.Q4K = true
	baseLogits := base.Prefill(prompt)
	// Same weights; the legacy head is a Q8 requant of the Q6_K values, so only quantization noise remains.
	assertCosineAtLeast(t, "packed tied prefill vs legacy two-copy", baseLogits, gotLogits, 0.999)

	next := argmax(gotLogits)
	stepped := got.Step(next)
	baseStepped := base.Step(next)
	if len(stepped) != packed.Cfg.VocabSize || hasNaN(stepped) {
		t.Fatalf("packed Step logits invalid: len=%d", len(stepped))
	}
	assertCosineAtLeast(t, "packed tied Step vs legacy two-copy", baseStepped, stepped, 0.999)
}

func hasNaN(xs []float32) bool {
	for _, v := range xs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

// TestTiedQ6KEmbeddingAdmitsHostQ4KSessionShapes: the loader selects the tied table by default,
// so every host Q4K session shape the tools build must run, including MetalQ4K without the f32
// Metal flag (rawdecode, modelbench, fakchat, the native scheduler). A non-Q4K session is refused
// with a typed panic rather than reaching a missing Q8 head.
func TestTiedQ6KEmbeddingAdmitsHostQ4KSessionShapes(t *testing.T) {
	_, packed, _, _ := tiedQ6KModelPair(t)
	prompt := []int{3, 7, 11, 5}
	for _, tc := range []struct {
		label                string
		quant, metal, metalQ bool
	}{
		{"q4k", false, false, false},
		{"q4k+quant", true, false, false},
		{"q4k+metalq4k", true, false, true},
		{"q4k+metal+metalq4k", true, true, true},
	} {
		s := packed.NewSession()
		s.Q4K, s.Quant, s.Metal, s.MetalQ4K = true, tc.quant, tc.metal, tc.metalQ
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%s session panicked on tied Q6_K model: %v", tc.label, r)
				}
			}()
			logits := s.Prefill(prompt)
			if len(logits) != packed.Cfg.VocabSize || hasNaN(logits) {
				t.Fatalf("%s prefill logits invalid", tc.label)
			}
			if next := s.Step(argmax(logits)); len(next) != packed.Cfg.VocabSize || hasNaN(next) {
				t.Fatalf("%s step logits invalid", tc.label)
			}
		}()
	}
	s := packed.NewSession()
	s.Quant = true
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(fmt.Sprint(r), "tied Q6_K") {
				t.Fatalf("non-Q4K session must be refused with a tied Q6_K reason, got %v", r)
			}
		}()
		s.Prefill(prompt)
	}()
}

// TestResidentReportConcurrentWithLazyQ8Fill: health probes run ResidentReport while decode can
// lazily quantize into q8w. Run under -race; the report must take q8Mu.
func TestResidentReportConcurrentWithLazyQ8Fill(t *testing.T) {
	m := NewSynthetic(qwen35HybridQ4KTestCfg())
	m.Quantize()
	var names []string
	for name, meta := range m.manifest {
		if len(meta.Shape) == 2 {
			names = append(names, name)
		}
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for _, name := range names {
			q8Mu.Lock()
			delete(m.q8w, name)
			q8Mu.Unlock()
			m.q8(name)
		}
	}()
	for i := 0; i < 50; i++ {
		_ = m.ResidentReport()
		_, _, _ = m.MoEResidentWeightBytes()
		_ = m.LMHeadRoute()
	}
	<-done
}
