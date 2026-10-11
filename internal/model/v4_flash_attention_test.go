package model

import (
	"context"
	"errors"
	"testing"
)

// v4_flash_attention_test.go - the bounded ratio-0 tracer witness for
// DeepSeek-V4-Flash-0731 per-layer attention (parent #12636).
//
// The published contract gives every layer a 128-token sliding window and ONE
// per-layer compression ratio. Ratio 0 is the pure window layer and is the
// smallest token tracer. Ratios 4 and 128 additionally carry compressor state
// that is not implemented yet. This file witnesses three invariants:
//
//  1. regime classification: ratio 0 -> window-only tracer, 4/128 ->
//     compressed (unimplemented), anything else -> invalid.
//  2. fail-closed dispatch: a V4 session whose layers carry an unimplemented
//     compression ratio must REFUSE to run before any prefill/decode, rather
//     than silently falling through to the generic sliding-window Q/K/V path.
//  3. direct Prefill, PrefillNoLogits and Step apply the same refusal before
//     forward state or GPU keepalive begins. Empty prefills remain no-ops and
//     the already-admitted ratio-0/non-Flash paths remain available.

// v4FlashTracerConfig is a small synthetic DeepSeek-V4 config carrying the
// published per-layer compression schedule shape. Dims are miniature so the
// synthetic model is cheap; the identity and the ratio schedule are what the
// attention gate keys on.
func v4FlashTracerConfig(ratios []int) Config {
	cfg := Config{
		ModelType:         "deepseek_v4",
		HiddenSize:        32,
		NumLayers:         len(ratios),
		NumHeads:          4,
		NumKVHeads:        1,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
		CompressRatios:    append([]int(nil), ratios...),
	}
	cfg.Window = make([]int, cfg.NumLayers)
	for i := range cfg.Window {
		cfg.Window[i] = V4FlashWindowSize
	}
	return cfg
}

// TestV4FlashAttentionRegimeClassification pins the closed classification map.
func TestV4FlashAttentionRegimeClassification(t *testing.T) {
	cases := []struct {
		ratio int
		want  V4FlashAttentionRegime
		ok    bool
	}{
		{ratio: 0, want: V4FlashAttentionWindowOnly, ok: true},
		{ratio: 4, want: V4FlashAttentionCompressed, ok: true},
		{ratio: 128, want: V4FlashAttentionCompressed, ok: true},
		{ratio: -1, want: 0, ok: false},
		{ratio: 1, want: 0, ok: false},
		{ratio: 5, want: 0, ok: false},
		{ratio: 256, want: 0, ok: false},
	}
	for _, tc := range cases {
		got, ok := v4FlashAttentionRegime(tc.ratio)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Fatalf("regime(%d) = (%v,%v), want (%v,%v)", tc.ratio, got, ok, tc.want, tc.ok)
		}
	}
}

// TestV4FlashRefusesUnimplementedCompressionBeforeGenerate is the load-bearing
// fail-closed gate. A session over a config whose schedule carries ratio 4/128
// must return the typed unimplemented error from GenerateContext, with zero
// tokens emitted, BEFORE any prefill/decode. Falling through to the generic
// sliding-window path is the exact defect this gate forbids.
func TestV4FlashRefusesUnimplementedCompressionBeforeGenerate(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 4, 128})
	m := NewSynthetic(cfg)
	s := m.NewSession()

	toks, err := s.GenerateContext(context.Background(), []int{1, 2, 3}, 4)
	if !errors.Is(err, ErrV4FlashAttentionUnimplemented) {
		t.Fatalf("GenerateContext error = %v, want ErrV4FlashAttentionUnimplemented", err)
	}
	if len(toks) != 0 {
		t.Fatalf("emitted %d tokens before refusing compressed attention; want 0", len(toks))
	}
	if s.Cache.Len() != 0 {
		t.Fatalf("session cache holds %d rows after refusal; want 0 (no prefill ran)", s.Cache.Len())
	}
}

// TestV4FlashRatioZeroTracerRunsThroughWindowPath proves the ratio-0 tracer is
// admitted: an all-zero schedule generates without the compressed refusal, and
// the session cache advances (the window attention path executed).
func TestV4FlashRatioZeroTracerRunsThroughWindowPath(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 0, 0, 0})
	m := NewSynthetic(cfg)
	s := m.NewSession()

	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		t.Fatalf("ratio-0 tracer refused: %v", err)
	}
	toks, err := s.GenerateContext(context.Background(), []int{1, 2, 3}, 4)
	if err != nil {
		t.Fatalf("ratio-0 tracer GenerateContext error: %v", err)
	}
	if len(toks) != 4 {
		t.Fatalf("emitted %d tokens, want 4", len(toks))
	}
	if s.Cache.Len() == 0 {
		t.Fatal("session cache did not advance; window attention path did not run")
	}
}

// TestV4FlashInvalidRatioFailsClosed: a schedule outside {0,4,128} is malformed
// metadata and must refuse with the typed error rather than run.
func TestV4FlashInvalidRatioFailsClosed(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 5})
	s := NewSynthetic(cfg).NewSession()
	if _, err := s.GenerateContext(context.Background(), []int{1}, 1); !errors.Is(err, ErrV4FlashAttentionRatioInvalid) {
		t.Fatalf("invalid ratio error = %v, want ErrV4FlashAttentionRatioInvalid", err)
	}
}

// TestV4FlashGateLeavesNonV4FamiliesAlone: a Llama-shaped model with no
// compression schedule must not be refused by the V4 gate.
func TestV4FlashGateLeavesNonV4FamiliesAlone(t *testing.T) {
	cfg := swaTestCfg()
	cfg.ModelType = "llama"
	cfg.CompressRatios = nil
	s := NewSynthetic(cfg).NewSession()
	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		t.Fatalf("non-V4 family refused by V4 gate: %v", err)
	}
}

// TestV4FlashRatioScheduleIsSingleSourced pins the #1714 invariant: the closed
// set of admitted Flash 0731 compression ratios is DERIVED from the declared
// schedule (deepSeekV4FlashCompressRatios), never a second hardcoded literal.
//
// The official DeepSeek V4.1 config publishes compress_ratios drawn from
// {0,1,2} - a DIFFERENT closed set. Before single-sourcing, both the attention
// regime and the KV layout carried their own "{0,4,128}" literal, so a reader
// (or a later edit) could change one and silently diverge from the admitted
// schedule. This witness fixes the derivation and the cross-family separation.
func TestV4FlashRatioScheduleIsSingleSourced(t *testing.T) {
	// Membership is derived exactly from the declared schedule: every declared
	// ratio is admitted, nothing else in a wide probe range is.
	declared := make(map[int]bool)
	for _, r := range deepSeekV4FlashCompressRatios {
		declared[r] = true
	}
	for ratio := -2; ratio <= 260; ratio++ {
		got := V4FlashScheduleAdmitsRatio(ratio)
		if got != declared[ratio] {
			t.Fatalf("V4FlashScheduleAdmitsRatio(%d) = %v, want %v (declared schedule membership)", ratio, got, declared[ratio])
		}
	}

	// The derived closed set is exactly {0,4,128} for the 0731 schedule and is
	// a subset of the declared schedule.
	set := V4FlashRatioScheduleClosedSet()
	if len(set) != 3 || set[0] != 0 || set[1] != 4 || set[2] != 128 {
		t.Fatalf("Flash 0731 derived closed set = %v, want [0 4 128]", set)
	}
	for _, r := range set {
		if !declared[r] {
			t.Fatalf("derived closed-set member %d is not in the declared schedule", r)
		}
	}

	// Cross-family separation: the V4.1 official ratios (1 and 2) are NOT Flash
	// 0731 members, and the Flash 0731 compressed ratios are not V4.1 members.
	for _, v41Ratio := range []int{1, 2} {
		if V4FlashScheduleAdmitsRatio(v41Ratio) {
			t.Fatalf("V4.1 ratio %d wrongly admitted by the Flash 0731 closed set", v41Ratio)
		}
		if _, ok := v4FlashAttentionRegime(v41Ratio); ok {
			t.Fatalf("v4FlashAttentionRegime(%d) admitted a V4.1-only ratio", v41Ratio)
		}
	}
	for _, flashRatio := range []int{0, 4, 128} {
		if _, ok := v4FlashAttentionRegime(flashRatio); !ok {
			t.Fatalf("v4FlashAttentionRegime(%d) refused a declared Flash 0731 ratio", flashRatio)
		}
	}
}

// TestV4FlashDirectEntriesRefuseBeforeMutation closes the direct-entry gap
// recorded by #12935. Every call uses the existing session API, so this same
// witness can run on the parent without introducing candidate-only symbols.
// fak-test:runtime fast est=100ms
func TestV4FlashDirectEntriesRefuseBeforeMutation(t *testing.T) {
	begin, end := 0, 0
	previous := beginSessionGPUKeepAlive
	beginSessionGPUKeepAlive = func(*Session) func() {
		begin++
		return func() { end++ }
	}
	t.Cleanup(func() { beginSessionGPUKeepAlive = previous })
	entries := []struct {
		name string
		call func(*Session)
	}{
		{"prefill", func(s *Session) { s.Prefill([]int{1, 2}) }},
		{"prefill-no-logits", func(s *Session) { s.PrefillNoLogits([]int{1, 2}) }},
		{"step", func(s *Session) { s.Step(1) }},
	}
	for _, schedule := range []struct {
		name   string
		ratios []int
		want   error
	}{
		{"ratio-4", []int{4}, ErrV4FlashAttentionUnimplemented},
		{"ratio-128", []int{128}, ErrV4FlashAttentionUnimplemented},
		{"mixed", []int{0, 4, 128}, ErrV4FlashAttentionUnimplemented},
		{"invalid", []int{0, 5}, ErrV4FlashAttentionRatioInvalid},
	} {
		for _, entry := range entries {
			t.Run(schedule.name+"/"+entry.name, func(t *testing.T) {
				s := NewSynthetic(v4FlashTracerConfig(schedule.ratios)).NewSession()
				t.Cleanup(s.Close)
				beforeBegin, beforeEnd := begin, end
				var raised any
				func() {
					defer func() { raised = recover() }()
					entry.call(s)
				}()
				err, ok := raised.(error)
				if !ok || !errors.Is(err, schedule.want) {
					t.Errorf("direct entry panic = %v, want %v", raised, schedule.want)
				}
				if got := s.Cache.Len(); got != 0 {
					t.Errorf("cache advanced to %d rows before refusal", got)
				}
				if begin != beforeBegin || end != beforeEnd {
					t.Errorf("keepalive began/ended before refusal: %d/%d", begin-beforeBegin, end-beforeEnd)
				}
				if s.BackendSessionClosed() || s.halFailure != nil {
					t.Errorf("admission refusal retired backend session: closed=%v failure=%v", s.BackendSessionClosed(), s.halFailure)
				}
			})
		}
	}
	t.Run("empty-prefills", func(t *testing.T) {
		s := NewSynthetic(v4FlashTracerConfig([]int{4})).NewSession()
		t.Cleanup(s.Close)
		if got := s.Prefill(nil); got != nil {
			t.Fatalf("empty prefill = %v, want nil", got)
		}
		s.PrefillNoLogits(nil)
		if s.Cache.Len() != 0 {
			t.Fatal("empty prefill changed cache")
		}
	})
	t.Run("cancelled-generate", func(t *testing.T) {
		s := NewSynthetic(v4FlashTracerConfig([]int{4})).NewSession()
		t.Cleanup(s.Close)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := s.GenerateContext(ctx, []int{1}, 1); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled GenerateContext = %v, want cancellation before architecture refusal", err)
		}
		if s.Cache.Len() != 0 {
			t.Fatal("cancelled generation changed cache")
		}
	})
}

// These are the existing generic tracer paths, not a claim that compressed
// Flash attention or the complete V4 Pro architecture has been implemented.
// fak-test:runtime fast est=100ms
func TestV4FlashDirectEntriesPreserveAdmittedPaths(t *testing.T) {
	for _, kind := range []string{"ratio-zero", "no-schedule", "non-v4"} {
		t.Run(kind, func(t *testing.T) {
			cfg := v4FlashTracerConfig([]int{0, 0})
			switch kind {
			case "no-schedule":
				cfg.CompressRatios = nil
			case "non-v4":
				cfg.ModelType = "llama"
				cfg.CompressRatios = []int{4, 5}
			}
			s := NewSynthetic(cfg).NewSession()
			t.Cleanup(s.Close)
			if got := s.Prefill([]int{1, 2}); len(got) != cfg.VocabSize {
				t.Fatalf("prefill logits = %d, want %d", len(got), cfg.VocabSize)
			}
			if got := s.Step(3); len(got) != cfg.VocabSize {
				t.Fatalf("step logits = %d, want %d", len(got), cfg.VocabSize)
			}
			s.PrefillNoLogits([]int{4, 5})
			if s.Cache.Len() != 5 {
				t.Fatalf("direct entries advanced cache to %d, want 5", s.Cache.Len())
			}
		})
	}
	// V4.1 owns a distinct forward and ratio contract. Classify it without
	// manufacturing V4.1 weights for this generic tracer fixture.
	cfg := v4FlashTracerConfig([]int{0, 1, 2})
	cfg.ModelType = "deepseek_v41"
	s := &Session{M: &Model{Cfg: cfg}}
	if err := s.refuseUnimplementedV4FlashAttention(); err != nil {
		t.Fatalf("V4.1 intercepted by Flash guard: %v", err)
	}
}
