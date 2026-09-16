package model

// v41_decoder_test.go — the witness for the DeepSeek V4.1 decoder OUTPUT TAIL
// and the ordinary Prefill/Step dispatch seam (issue #12909). It proves three
// properties:
//
//  1. Determinism: a Prefill followed by Step (which reuses the continuation
//     state) is bit-identical to a single Prefill over the grown history, and
//     two independent decoders are bit-identical.
//  2. Tail agreement: the leaf explicit output tail (final norm + output head)
//     is bit-identical to the assembly shared tail on the same hidden row, so
//     the decoder owns no second arithmetic transcription.
//  3. Fail-closed: a malformed geometry, a malformed token id, a wrong-width
//     tail row, and a weightless model each return a typed error with NO
//     logits — never a silently-degraded result.
//
// Geometry. Two derivations exist: V41DecoderGeometryFor enforces the PINNED
// 40-layer / width-5120 official envelope; v41DecoderGeometryOf admits any
// internally-consistent V4.1 geometry and rejects malformed axes. The runtime
// seam uses the latter so the tiny reduced fixture is runnable; the guard test
// pins the former against the reduced envelope.

import (
	"errors"
	"math"
	"testing"
)

// v41DecoderPinnedConfig is the pinned-envelope config (depth 40, width 5120)
// used to witness V41DecoderGeometryFor admission. It carries no weights.
func v41DecoderPinnedConfig(t *testing.T) Config {
	t.Helper()
	_, cfg := readDeepSeekV41Config(t)
	cfg.NumLayers = V41DecoderLayers
	cfg.HiddenSize = V41DecoderWidth
	cfg.NumHeads = 8
	cfg.HeadDim = 64
	cfg.QLoraRank = 64
	cfg.OLoraRank = 32
	cfg.OGroups = 4
	cfg.MoEIntermediateSize = 64
	cfg.VocabSize = 8
	if !cfg.IsDeepSeekV41() {
		t.Fatal("pinned config lost V4.1 identity")
	}
	return cfg
}

// v41DecoderTailModel builds a tail-only Model (final norm + head + tied
// embedding) at the reduced fixture width so Tail can be exercised without a
// full assembly weight set.
func v41DecoderTailModel(t *testing.T) *Model {
	t.Helper()
	cfg := v41TestReducedConfig(t, 1, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	H, V := cfg.HiddenSize, cfg.VocabSize
	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{V, H}},
		{"lm_head.weight", []int{V, H}},
		{"model.norm.weight", []int{H}},
	}
	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if name == "model.norm.weight" {
			return 1.0
		}
		return synthMatmulFill(name, next)
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func v41DecoderEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want %v (not bit-identical)", name, i, got[i], want[i])
		}
	}
}

// TestV41DecoderGeometryGuard pins the 40-layer / width-5120 envelope: the
// published axes admit, and the reduced fixture, a short depth, a narrow width,
// and a non-V4.1 config each fail closed with ErrV41DecoderGeometry.
func TestV41DecoderGeometryGuard(t *testing.T) {
	good := v41DecoderPinnedConfig(t)
	if _, err := V41DecoderGeometryFor(good); err != nil {
		t.Fatalf("pinned 40/5120 geometry rejected: %v", err)
	}

	reduced := v41ReducedModel(t).Cfg
	if _, err := V41DecoderGeometryFor(reduced); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("reduced geometry err = %v, want ErrV41DecoderGeometry", err)
	}

	shortLayer := good
	shortLayer.NumLayers = V41DecoderLayers - 1
	if _, err := V41DecoderGeometryFor(shortLayer); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("short-depth geometry err = %v, want ErrV41DecoderGeometry", err)
	}

	narrow := good
	narrow.HiddenSize = V41DecoderWidth / 2
	if _, err := V41DecoderGeometryFor(narrow); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("narrow-width geometry err = %v, want ErrV41DecoderGeometry", err)
	}

	plain := good
	plain.DeepSeekV41 = nil
	plain.ModelType = "llama"
	if _, err := V41DecoderGeometryFor(plain); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("non-V4.1 geometry err = %v, want ErrV41DecoderGeometry", err)
	}

	// v41DecoderGeometryOf admits the reduced fixture (internally consistent)
	// but still rejects a malformed axis.
	if _, err := v41DecoderGeometryOf(reduced); err != nil {
		t.Fatalf("reduced geometry rejected by runtime derivation: %v", err)
	}
	zeroWidth := reduced
	zeroWidth.HiddenSize = 0
	if _, err := v41DecoderGeometryOf(zeroWidth); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("zero-width runtime geometry err = %v, want ErrV41DecoderGeometry", err)
	}
}

// TestV41DecoderTailMatchesSharedTail proves the leaf output tail (explicit
// final norm + head) is bit-identical to the assembly shared tail
// (m.logitsFromHidden) on the same hidden row, so Prefill/Step logits carry the
// same arithmetic as Model.Forward.
func TestV41DecoderTailMatchesSharedTail(t *testing.T) {
	m := v41DecoderTailModel(t)
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
	}
	d, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder: %v", err)
	}
	got, err := d.Tail(x)
	if err != nil {
		t.Fatalf("Tail: %v", err)
	}
	want := m.logitsFromHidden(x)
	v41DecoderEqual(t, "output tail vs shared tail", got, want)
}

// TestV41DecoderPrefillStepDeterminism is the core acceptance witness: Prefill
// then Step (reusing the continuation state) equals a single Prefill over the
// grown history, and two independent decoders are bit-identical.
func TestV41DecoderPrefillStepDeterminism(t *testing.T) {
	m := v41ReducedModel(t)
	prompt := []int{1, 3, 5}
	next := 2

	a, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder: %v", err)
	}
	if _, err := a.Prefill(prompt); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	stepLogits, err := a.Step(next)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if len(stepLogits) != m.Cfg.VocabSize {
		t.Fatalf("Step logits = %d, want %d", len(stepLogits), m.Cfg.VocabSize)
	}
	if got := a.State.History(); len(got) != len(prompt)+1 || got[len(got)-1] != next {
		t.Fatalf("history = %v, want prompt+%d", got, next)
	}

	// A fresh decoder over the SAME grown history in ONE Prefill must produce
	// the identical tail logits: Step is exactly a longer Prefill.
	tail, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder(2): %v", err)
	}
	full, err := tail.Prefill(append(append([]int(nil), prompt...), next))
	if err != nil {
		t.Fatalf("Prefill(full): %v", err)
	}
	v41DecoderEqual(t, "Step vs single full Prefill", stepLogits, full)

	// Determinism across two independent sessions.
	b, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder(3): %v", err)
	}
	if _, err := b.Prefill(prompt); err != nil {
		t.Fatalf("Prefill(b): %v", err)
	}
	stepB, err := b.Step(next)
	if err != nil {
		t.Fatalf("Step(b): %v", err)
	}
	v41DecoderEqual(t, "deterministic Step", stepLogits, stepB)
}

// TestV41DecoderFailsClosed proves the decoder never returns logits for
// malformed input: a bad token id, a Step before any Prefill, a wrong-width
// tail row, and a weightless model each return a typed error and no logits.
func TestV41DecoderFailsClosed(t *testing.T) {
	m := v41ReducedModel(t)

	// Malformed prompt id (negative and out of vocabulary).
	d, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder: %v", err)
	}
	for _, bad := range [][]int{{-1}, {m.Cfg.VocabSize}} {
		logits, err := d.Prefill(bad)
		if !errors.Is(err, ErrV41DecoderToken) {
			t.Fatalf("Prefill(%v) err = %v, want ErrV41DecoderToken", bad, err)
		}
		if logits != nil {
			t.Fatalf("Prefill(%v) returned logits %v despite a malformed id", bad, logits)
		}
		if len(d.State.History()) != 0 {
			t.Fatalf("malformed Prefill(%v) mutated history to %v", bad, d.State.History())
		}
	}

	// Step before any Prefill is refused rather than conditioning on nothing.
	fresh, err := NewV41TextDecoder(m)
	if err != nil {
		t.Fatalf("NewV41TextDecoder(fresh): %v", err)
	}
	if logits, err := fresh.Step(1); !errors.Is(err, ErrV41DecoderGeometry) || logits != nil {
		t.Fatalf("Step-before-Prefill = (%v, %v), want (nil, ErrV41DecoderGeometry)", logits, err)
	}

	// A malformed Step id after a valid Prefill.
	if _, err := d.Prefill([]int{1}); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if logits, err := d.Step(m.Cfg.VocabSize); !errors.Is(err, ErrV41DecoderToken) || logits != nil {
		t.Fatalf("malformed Step = (%v, %v), want (nil, ErrV41DecoderToken)", logits, err)
	}

	// Wrong-width tail row.
	if _, err := d.Tail(make([]float32, m.Cfg.HiddenSize-1)); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("short tail err = %v, want ErrV41DecoderGeometry", err)
	}

	// Non-V4.1 model refuses construction.
	plain := &Model{Cfg: Config{ModelType: "llama", HiddenSize: 32, VocabSize: 4}}
	if _, err := NewV41TextDecoder(plain); !errors.Is(err, ErrV41DecoderGeometry) {
		t.Fatalf("non-V4.1 construct err = %v, want ErrV41DecoderGeometry", err)
	}

	// A weightless V4.1 model refuses the forward through the assembly admission.
	weightless := &Model{Cfg: v41DecoderPinnedConfig(t)}
	wd, err := NewV41TextDecoder(weightless)
	if err != nil {
		t.Fatalf("NewV41TextDecoder(weightless): %v", err)
	}
	logits, err := wd.Prefill([]int{1})
	if !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("weightless Prefill err = %v, want ErrV41NativeUnsupported", err)
	}
	if logits != nil {
		t.Fatalf("weightless Prefill returned logits %v", logits)
	}
}
