package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func makeTestQ2KPayload(vocab, hidden int) []byte {
	nblk := hidden / qkK
	raw := make([]byte, vocab*nblk*q2kBlockBytes)
	for i := range raw {
		raw[i] = byte((i*17 + 3) % 256)
	}
	pinResidentQuantScales(raw, vocab, nblk, kindQ2K)
	return raw
}

func TestQ2KEmbeddingGatherRowMatchesScalarRef(t *testing.T) {
	const (
		vocab  = 6
		hidden = 512 // 2 superblocks per row
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}

	for _, tokenID := range []int{0, 2, 5, 2, 0, 5} {
		dst := make([]float32, hidden)
		if err := q2k.GatherRow(tokenID, dst, 1.0); err != nil {
			t.Fatalf("GatherRow(%d): %v", tokenID, err)
		}

		// Verify against scalar reference dequantization block by block
		refDst := make([]float32, hidden)
		rowBytes := (hidden / qkK) * q2kBlockBytes
		rowStart := tokenID * rowBytes
		for b := 0; b < hidden/qkK; b++ {
			blk := raw[rowStart+b*q2kBlockBytes : rowStart+(b+1)*q2kBlockBytes]
			q2kDequantSuperBlock(refDst[b*qkK:(b+1)*qkK], blk)
		}

		for i := 0; i < hidden; i++ {
			if math.IsNaN(float64(dst[i])) || math.IsInf(float64(dst[i]), 0) {
				t.Fatalf("row %d elem %d is not finite: %v", tokenID, i, dst[i])
			}
			if dst[i] != refDst[i] {
				t.Fatalf("row %d elem %d mismatch: got %v, want %v", tokenID, i, dst[i], refDst[i])
			}
		}
	}
}

func TestQ2KEmbeddingBoundsAndErrors(t *testing.T) {
	const (
		vocab  = 4
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}

	buf := make([]float32, hidden)
	if err := q2k.GatherRow(-1, buf, 1.0); err == nil {
		t.Error("GatherRow(-1) want error, got nil")
	}
	if err := q2k.GatherRow(vocab, buf, 1.0); err == nil {
		t.Errorf("GatherRow(%d) want error, got nil", vocab)
	}
	shortBuf := make([]float32, hidden-1)
	if err := q2k.GatherRow(0, shortBuf, 1.0); err == nil {
		t.Error("GatherRow with short buffer want error, got nil")
	}

	var nilQ2K *Q2KEmbedding
	if err := nilQ2K.GatherRow(0, buf, 1.0); err == nil {
		t.Error("GatherRow on nil Q2KEmbedding want error, got nil")
	}

	// Invalid dimensions
	if _, err := NewQ2KEmbedding(raw, 0, hidden); err == nil {
		t.Error("NewQ2KEmbedding(vocab=0) want error, got nil")
	}
	if _, err := NewQ2KEmbedding(raw, vocab, 0); err == nil {
		t.Error("NewQ2KEmbedding(hidden=0) want error, got nil")
	}
	if _, err := NewQ2KEmbedding(raw, vocab, 128); err == nil {
		t.Error("NewQ2KEmbedding(hidden=128, not mult of 256) want error, got nil")
	}
	if _, err := NewQ2KEmbedding(raw[:len(raw)-1], vocab, hidden); err == nil {
		t.Error("NewQ2KEmbedding(truncated payload) want error, got nil")
	}
}

func TestQ2KEmbeddingScale(t *testing.T) {
	const (
		vocab  = 2
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}

	base := make([]float32, hidden)
	if err := q2k.GatherRow(0, base, 1.0); err != nil {
		t.Fatal(err)
	}

	// scale = 0 -> no-op
	s0 := make([]float32, hidden)
	if err := q2k.GatherRow(0, s0, 0.0); err != nil {
		t.Fatal(err)
	}
	for i := range base {
		if s0[i] != base[i] {
			t.Fatalf("scale 0 modified elem %d: got %v, want %v", i, s0[i], base[i])
		}
	}

	// scale = 2.5
	s25 := make([]float32, hidden)
	if err := q2k.GatherRow(0, s25, 2.5); err != nil {
		t.Fatal(err)
	}
	for i := range base {
		want := base[i] * 2.5
		if s25[i] != want {
			t.Fatalf("scale 2.5 elem %d mismatch: got %v, want %v", i, s25[i], want)
		}
	}

	// scale = 0.5
	s05 := make([]float32, hidden)
	if err := q2k.GatherRow(0, s05, 0.5); err != nil {
		t.Fatal(err)
	}
	for i := range base {
		want := base[i] * 0.5
		if s05[i] != want {
			t.Fatalf("scale 0.5 elem %d mismatch: got %v, want %v", i, s05[i], want)
		}
	}
}

func TestQ2KEmbeddingImmutability(t *testing.T) {
	const (
		vocab  = 2
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}

	initial := make([]float32, hidden)
	if err := q2k.GatherRow(0, initial, 1.0); err != nil {
		t.Fatal(err)
	}

	// Mutate the caller's raw slice
	raw[0] ^= 0xFF
	raw[1] ^= 0xFF

	after := make([]float32, hidden)
	if err := q2k.GatherRow(0, after, 1.0); err != nil {
		t.Fatal(err)
	}

	for i := range initial {
		if after[i] != initial[i] {
			t.Fatalf("caller mutation leaked into model embedding at elem %d", i)
		}
	}

	// Mutate the returned buffer
	after[0] = 99999.0
	again := make([]float32, hidden)
	if err := q2k.GatherRow(0, again, 1.0); err != nil {
		t.Fatal(err)
	}
	if again[0] != initial[0] {
		t.Fatal("destination buffer mutation affected subsequent GatherRow calls")
	}
}

func TestModelEmbedRowsRefusesWholeTableExpansion(t *testing.T) {
	const (
		vocab  = 2
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Cfg: Config{
			VocabSize:  vocab,
			HiddenSize: hidden,
		},
		manifest:     map[string]tensorMeta{},
		Q2KEmbedding: q2k,
	}

	if !m.HasQ2KEmbedding() {
		t.Fatal("HasQ2KEmbedding() returned false")
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("embedRows() did not panic on packed Q2_K embedding")
		}
		if r != ErrPackedEmbeddingWholeTableRefused {
			t.Fatalf("embedRows() panic = %v, want %v", r, ErrPackedEmbeddingWholeTableRefused)
		}
	}()

	_ = m.embedRows()
}

func TestSessionTokenEmbeddingQ2K(t *testing.T) {
	const (
		vocab  = 4
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Cfg: Config{
			VocabSize:  vocab,
			HiddenSize: hidden,
		},
		manifest:     map[string]tensorMeta{},
		Q2KEmbedding: q2k,
	}

	s := &Session{
		M: m,
	}

	// Bounds checks
	if _, err := s.TokenEmbedding(-1); err == nil {
		t.Error("TokenEmbedding(-1) succeeded, want error")
	}
	if _, err := s.TokenEmbedding(vocab); err == nil {
		t.Errorf("TokenEmbedding(%d) succeeded, want error", vocab)
	}

	// Defensiveness of copy
	tok0, err := s.TokenEmbedding(0)
	if err != nil {
		t.Fatalf("TokenEmbedding(0): %v", err)
	}
	if len(tok0) != hidden {
		t.Fatalf("TokenEmbedding len = %d, want %d", len(tok0), hidden)
	}
	origVal := tok0[0]
	tok0[0] = origVal + 100.0

	tok0Again, err := s.TokenEmbedding(0)
	if err != nil {
		t.Fatalf("TokenEmbedding(0) again: %v", err)
	}
	if tok0Again[0] != origVal {
		t.Fatal("TokenEmbedding returned mutable reference, want defensive copy")
	}
}

func TestResidentReportQ2KEmbedding(t *testing.T) {
	const (
		vocab  = 4
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		manifest:     map[string]tensorMeta{"norm": {Nbytes: 1024}},
		q8w:          map[string]*q8Tensor{},
		q4kw:         map[string]*q4kTensor{},
		kqw:          map[string]*kQuantTensor{},
		Q2KEmbedding: q2k,
	}

	r := m.ResidentReport()
	if r.Q2KEmbedTensors != 1 {
		t.Errorf("Q2KEmbedTensors = %d, want 1", r.Q2KEmbedTensors)
	}
	if r.Q2KEmbedBytes != int64(q2k.Bytes()) {
		t.Errorf("Q2KEmbedBytes = %d, want %d", r.Q2KEmbedBytes, q2k.Bytes())
	}
	if r.Q2KEmbedParams != int64(vocab*hidden) {
		t.Errorf("Q2KEmbedParams = %d, want %d", r.Q2KEmbedParams, vocab*hidden)
	}
	if r.TotalResidentBytes != int64(q2k.Bytes())+1024 {
		t.Errorf("TotalResidentBytes = %d, want %d", r.TotalResidentBytes, int64(q2k.Bytes())+1024)
	}
	if r.DecodeBytesPerToken != 0 {
		t.Errorf("DecodeBytesPerToken = %d, want 0 (Q2K embedding must be excluded)", r.DecodeBytesPerToken)
	}

	repl, exp, ok := m.MoEResidentWeightBytes()
	if !ok || repl != int64(q2k.Bytes())+1024 || exp != 0 {
		t.Errorf("MoEResidentWeightBytes = (%d, %d, %v), want (%d, 0, true)", repl, exp, ok, int64(q2k.Bytes())+1024)
	}
}

func TestSessionGuardsQ2KEmbedding(t *testing.T) {
	const (
		vocab  = 2
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Cfg: Config{
			VocabSize:  vocab,
			HiddenSize: hidden,
			NumLayers:  1,
		},
		manifest:     map[string]tensorMeta{},
		Q2KEmbedding: q2k,
	}

	// CPU session with Backend == nil requires Quant or Q4K
	assertPanic := func(name string, mutate func(*Session)) {
		t.Helper()
		s := &Session{M: m, Quant: true}
		mutate(s)
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		s.validateDenseGPULayers()
	}

	assertPanic("Metal", func(s *Session) { s.Metal = true })
	assertPanic("MetalQ4K", func(s *Session) { s.MetalQ4K = true })
	assertPanic("Q4", func(s *Session) { s.Q4 = true })
	assertPanic("F16", func(s *Session) { s.F16 = true })
	assertPanic("GPTQ", func(s *Session) { s.GPTQ = true })
	assertPanic("PrecisionPolicy", func(s *Session) { s.PrecisionPolicy = &DynamicPrecisionPolicy{} })
	assertPanic("DenseGPULayers", func(s *Session) { s.DenseGPULayers = 1; s.Backend = compute.Default() })
	assertPanic("GPULayers", func(s *Session) { s.GPULayers = 1 })
	assertPanic("CPU without Quant or Q4K", func(s *Session) { s.Quant = false; s.Q4K = false })

	// Allowed configurations
	sQuant := &Session{M: m, Quant: true}
	sQuant.validateDenseGPULayers()

	sQ4K := &Session{M: m, Q4K: true}
	sQ4K.validateDenseGPULayers()

	sHAL := &Session{M: m, Backend: compute.Default()}
	sHAL.validateDenseGPULayers()
}

func TestHALTokenInputUploadsDecodedRowAndDeclinesSequence(t *testing.T) {
	const (
		vocab  = 4
		hidden = 256
	)
	raw := makeTestQ2KPayload(vocab, hidden)
	q2k, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Cfg: Config{
			ModelType:  "qwen35",
			VocabSize:  vocab,
			HiddenSize: hidden,
			NumLayers:  1,
		},
		manifest:     map[string]tensorMeta{},
		Q2KEmbedding: q2k,
	}

	s := &Session{
		M:       m,
		Backend: compute.Default(),
	}

	// tryQwen35SequencePrefill must decline
	res, used, err := s.tryQwen35SequencePrefill([]int{0, 1}, true)
	if used {
		t.Error("tryQwen35SequencePrefill must decline (used=false) when Q2KEmbedding is present")
	}
	if err != nil {
		t.Errorf("tryQwen35SequencePrefill unexpected error: %v", err)
	}
	if res.Tokens != 0 {
		t.Errorf("tryQwen35SequencePrefill returned non-zero tokens: %d", res.Tokens)
	}
}

func TestQwen35PrefillAndStepCPUContinuation(t *testing.T) {
	cfg := Config{
		ModelType:        "qwen35",
		VocabSize:        16,
		HiddenSize:       256,
		IntermediateSize: 512,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          128,
		RMSNormEps:       1e-6,
	}
	m := NewSynthetic(cfg)
	m.Quantize()

	raw := makeTestQ2KPayload(cfg.VocabSize, cfg.HiddenSize)
	q2k, err := NewQ2KEmbedding(raw, cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		t.Fatalf("NewQ2KEmbedding: %v", err)
	}
	m.Q2KEmbedding = q2k
	delete(m.manifest, "model.embed_tokens.weight")

	s := m.NewSession()
	s.Quant = true

	// Prefill
	logits := s.Prefill([]int{1, 2})
	if len(logits) != cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Prefill logits[%d] is not finite: %v", i, v)
		}
	}

	// 3 subsequent Steps
	for step := 0; step < 3; step++ {
		stepLogits := s.Step(step + 3)
		if len(stepLogits) != cfg.VocabSize {
			t.Fatalf("Step %d logits len = %d, want %d", step, len(stepLogits), cfg.VocabSize)
		}
		for i, v := range stepLogits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Step %d logits[%d] is not finite: %v", step, i, v)
			}
		}
	}
}

