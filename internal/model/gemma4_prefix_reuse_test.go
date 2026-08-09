package model

import (
	"math"
	"strings"
	"testing"
)

// gemma4_prefix_reuse_test.go — the model-level half of the #5548 witness.
//
// internal/agent/inkernel_gemma4_reuse_test.go proves the SERVING consequence (a partial
// radix hit served wrong logits). These arms pin the rule underneath it at the API every
// reuser goes through, so the next caller to build a session from a cached prefix inherits
// the refusal instead of re-deriving — and re-missing — the reasoning.

// tinyGemma4ReuseCfg mirrors internal/ggufload's tinyGemma4GGUF geometry: interleaved
// sliding/global layers with DIFFERENT per-layer head_dim and kv-head counts. The
// heterogeneity is not decorative here — it is why gemma4 has a dedicated cacheless
// forward, which is why its Session is a recompute bridge, which is why its cache is empty.
func tinyGemma4ReuseCfg() Config {
	const H, I, V, nH = 32, 64, 41, 4
	return Config{
		ModelType:          "gemma4",
		HiddenSize:         H,
		IntermediateSize:   I,
		VocabSize:          V,
		NumLayers:          4,
		NumHeads:           nH,
		HeadDim:            16,
		NumKVHeads:         1,
		RMSNormEps:         1e-6,
		RopeTheta:          1e6,
		BlockTopology:      SandwichNorm,
		ActGeluTanh:        true,
		QKNorm:             true,
		EmbedScale:         math.Sqrt(H),
		LogitSoftcap:       30,
		EOSTokenID:         -1,
		LayerTypes:         []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		HeadDimPerLayer:    []int{8, 8, 8, 16},
		NumKVHeadsPerLayer: []int{2, 2, 2, 1},
		RopeDimPerLayer:    []int{8, 8, 8, 16},
		RopeThetaPerLayer:  []float64{10000, 10000, 10000, 1e6},
		Window:             []int{2, 2, 2, -1},
	}
}

// TestGemma4CacheIsNotAPrefix is the premise the whole fix rests on: after ingesting a
// prompt, a gemma4 session's KVCache is EMPTY while its real state (the token history) is
// full. A reuser holding only the *KVCache therefore cannot tell "no prefix" from "a
// zero-length prefix", which is why the predicate has to live on the Config.
func TestGemma4CacheIsNotAPrefix(t *testing.T) {
	m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
	ids := []int{3, 9, 14, 27, 6, 31, 2, 18}

	s := m.NewSession()
	s.Prefill(ids)

	if got := len(s.gemma4Hist); got != len(ids) {
		t.Fatalf("gemma4 session ingested %d ids into its history, want %d", got, len(ids))
	}
	if got := s.Cache.Len(); got != 0 {
		t.Fatalf("premise changed: gemma4 session cache holds %d rows — if the bridge now caches, revisit KVPrefixReuseSupported", got)
	}
	if s.Cache.Clone().Len() != 0 {
		t.Fatal("a clone of the empty gemma4 cache must also be empty — this is the silent data loss #5548 reported")
	}
	t.Logf("gemma4 session after %d ids: history=%d, cache rows=%d (a clone carries 0/%d of the prefix)",
		len(ids), len(s.gemma4Hist), s.Cache.Len(), len(ids))
}

// TestKVPrefixReuseSupportedIsNarrow keeps the refusal from becoming a blanket one: it
// must exclude the recompute bridge and nothing else.
func TestKVPrefixReuseSupportedIsNarrow(t *testing.T) {
	if tinyGemma4ReuseCfg().KVPrefixReuseSupported() {
		t.Error("gemma4's empty cache is not a complete prefix — KVPrefixReuseSupported must be false")
	}
	for _, mt := range []string{"llama", "qwen3", "gemma3", "glm4_moe", "minimax_m2"} {
		if !(Config{ModelType: mt}).KVPrefixReuseSupported() {
			t.Errorf("%s caches its whole state in the KVCache — reuse must stay supported", mt)
		}
	}
}

// TestSessionFromPrefixRefusesGemma4 pins the loud failure. Before #5548 this call
// returned a session that had ingested nothing while its caller believed it held the
// prefix; a panic naming the architecture is strictly better than logits that are wrong
// with no error and no panic.
func TestSessionFromPrefixRefusesGemma4(t *testing.T) {
	m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
	prime := m.NewSession()
	prime.Prefill([]int{5, 12, 30, 7})

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("SessionFromPrefix accepted a gemma4 cache: it would hand back a session holding none of the prefix")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("SessionFromPrefix panicked with %T, want a string naming the architecture", r)
		}
		if !strings.Contains(msg, "gemma4") {
			t.Errorf("refusal must name the architecture so an operator can act on it, got %q", msg)
		}
		t.Logf("SessionFromPrefix refusal: %s", msg)
	}()

	m.SessionFromPrefix(prime.Cache)
}

// TestNewBatchFromPrefixRefusesGemma4 closes the one construction site the #5548 fix left
// ungated. SessionFromPrefix is not the only way to build a session from a cached prefix:
// NewBatchFromPrefixReserve assembles Session structs by hand (batch.go), so it never asked
// the predicate and inherited the exact defect the ticket named — one rung further out,
// because it fans the empty prefix across every lane at once.
//
// The batch path is the cross-agent reuse lever: prefill a shared system prompt ONCE, clone
// it into n agents. For gemma4 that clone carries nothing, so all n lanes decode as if the
// shared prefix did not exist. Nothing downstream notices, for the same reason nothing
// noticed on the radix path: the empty cache is a well-formed zero-length prefix, and
// StepBatch's serial fallback happily routes each lane to the recompute Step over a history
// holding only that lane's own generated tokens.
//
// It refuses instead of carrying gemma4Hist for the reason c11aedade already settled: the
// gemma4 forward is cacheless, so a "reused" batch would recompute every lane's prefix
// anyway — there is no prefill saving to preserve, and reporting one would misstate the
// cross-agent lever this constructor exists to measure.
func TestNewBatchFromPrefixRefusesGemma4(t *testing.T) {
	m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
	shared := []int{5, 12, 30, 7, 21, 9}

	base := m.NewSession()
	base.Prefill(shared)

	const lanes = 3
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("NewBatchFromPrefix accepted a gemma4 cache: it returned %d lanes each "+
				"believing they hold the %d-token shared prefix while carrying 0 of it, so every "+
				"lane decodes as if the shared prompt did not exist", lanes, len(shared))
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("NewBatchFromPrefix panicked with %T, want a string naming the architecture", r)
		}
		if !strings.Contains(msg, "gemma4") {
			t.Errorf("refusal must name the architecture so an operator can act on it, got %q", msg)
		}
		t.Logf("NewBatchFromPrefix refusal: %s", msg)
	}()

	m.NewBatchFromPrefix(base.Cache, lanes)
}

// TestNewBatchFromPrefixStaysOpenForCachedArches keeps the batch refusal as narrow as the
// session one: the cross-agent prefix-clone lever must still work for every architecture
// whose K/V rows really are its whole state.
func TestNewBatchFromPrefixStaysOpenForCachedArches(t *testing.T) {
	m := NewSynthetic(Config{
		HiddenSize: 32, IntermediateSize: 64, VocabSize: 41, NumLayers: 2,
		NumHeads: 4, HeadDim: 8, NumKVHeads: 4, RMSNormEps: 1e-6, RopeTheta: 1e4,
		EOSTokenID: -1,
	})
	base := m.NewSession()
	base.Prefill([]int{5, 12, 30, 7, 21, 9})

	bs := m.NewBatchFromPrefix(base.Cache, 3)
	if bs.N() != 3 {
		t.Fatalf("batch has %d lanes, want 3", bs.N())
	}
	for i, s := range bs.Seqs {
		if got := s.Cache.Len(); got != base.Cache.Len() {
			t.Errorf("lane %d cloned %d prefix rows, want %d", i, got, base.Cache.Len())
		}
	}
}
