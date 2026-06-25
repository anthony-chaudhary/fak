package main

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPrefixReuseFanoutWitness grounds fanbench's prefix-reuse GEOMETRY claim
// ((N−1)·prefix_tokens prefill saved) in the REAL kernel, fanned out across N
// sub-agents. It is the multi-sibling analogue of model.TestKVPrefixReuseMatchesRecompute:
// the master-goal prefix is prefilled ONCE, cloned into N sub-agents via
// NewBatchFromPrefix, and every clone's suffix decode is asserted bit-identical to an
// independent full prefill — so cross-agent prefix reuse provably never changes a
// sub-agent's result, and each clone sits at exactly the prefix length (the prefix was
// materialized a single time, not N times).
func TestPrefixReuseFanoutWitness(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 101, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)

	prefix := []int{3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26} // the shared master-goal prefix
	suffix := []int{27, 18, 28, 5, 9}                              // one sub-agent's private slice

	// compute the shared prefix ONCE
	base := m.NewSession()
	base.Prefill(prefix)
	if base.Cache.Len() != len(prefix) {
		t.Fatalf("base prefix len = %d, want %d", base.Cache.Len(), len(prefix))
	}

	// full recompute reference: a fresh session that prefills prefix+suffix
	full := m.NewSession()
	lFull := full.Prefill(concatInts(prefix, suffix))

	const N = 64 // a fan-out wide enough to exercise the "1000s of sub-agents" regime cheaply
	bs := m.NewBatchFromPrefix(base.Cache, N)
	if bs.N() != N {
		t.Fatalf("batch N = %d, want %d", bs.N(), N)
	}
	for a := 0; a < N; a++ {
		s := bs.Seqs[a]
		if s.Cache.Len() != len(prefix) {
			t.Fatalf("clone %d sits at len %d, want prefix len %d — prefix was NOT materialized once + cloned", a, s.Cache.Len(), len(prefix))
		}
		lReuse := s.Prefill(suffix) // prefill only the SUFFIX — the prefix's prefill FLOPs are skipped
		if d := maxAbsDiff(lReuse, lFull); d != 0 {
			t.Fatalf("clone %d: reuse not bit-identical to full recompute (max|Δ|=%.3e) — reuse changed a sub-agent's result", a, d)
		}
		if s.Cache.Len() != len(prefix)+len(suffix) {
			t.Fatalf("clone %d post-suffix len = %d, want %d", a, s.Cache.Len(), len(prefix)+len(suffix))
		}
	}
	t.Logf("witness: %d clones bit-identical to full recompute; prefix prefill positions SHARED=%d vs naive ISOLATED=%d (saved=(N-1)*P=%d)",
		N, len(prefix), N*len(prefix), (N-1)*len(prefix))
}

func TestBuildAgentGrid(t *testing.T) {
	// explicit wins
	if got := buildAgentGrid("1,4,16", 1024, "log"); !reflect.DeepEqual(got, []int{1, 4, 16}) {
		t.Fatalf("explicit grid = %v", got)
	}
	if got := buildAgentGrid("", 1024, "canonical"); !reflect.DeepEqual(got, []int{1, 100, 500, 1000}) {
		t.Fatalf("canonical grid = %v", got)
	}
	// log ladder always includes 1 and max
	got := buildAgentGrid("", 1000, "log")
	if got[0] != 1 || got[len(got)-1] != 1000 {
		t.Fatalf("log grid endpoints = %v", got)
	}
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("log grid not strictly increasing: %v", got)
		}
	}
	// full grid is 1..max
	if got := buildAgentGrid("", 5, "full"); !reflect.DeepEqual(got, []int{1, 2, 3, 4, 5}) {
		t.Fatalf("full grid = %v", got)
	}
	if got := buildAgentGrid("", 0, "log"); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("zero max grid = %v, want [1]", got)
	}
}

func TestBuildPrefixGrid(t *testing.T) {
	got, err := buildPrefixGrid("", 2048, defaultModelContextTokens, 256, 120, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int{2048}) {
		t.Fatalf("default prefix grid = %v", got)
	}

	got, err = buildPrefixGrid("smoke,8k,big", 2048, 131072, 256, 120, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{1024, 8192, 130336}; !reflect.DeepEqual(got, want) {
		t.Fatalf("mixed prefix grid = %v, want %v", got, want)
	}

	got, err = buildPrefixGrid("all", 2048, 262144, 256, 120, []int{4, 8})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{1024, 2048, 8192, 32768, 260928}; !reflect.DeepEqual(got, want) {
		t.Fatalf("all prefix grid = %v, want %v", got, want)
	}

	if _, err := buildPrefixGrid("big", 2048, 512, 256, 120, []int{4}); err == nil {
		t.Fatal("big prefix fit unexpectedly succeeded")
	}
	if _, err := buildPrefixGrid("nope", 2048, 131072, 256, 120, []int{4}); err == nil {
		t.Fatal("bad prefix name unexpectedly succeeded")
	}
}

func TestContextWindowFromConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"max_position_embeddings":262144}`), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := contextWindowFromConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}

	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"model_max_length":32768}`), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err = contextWindowFromConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != 32768 {
		t.Fatalf("model_max_length context window = %d, want 32768", got)
	}
}

func concatInts(a, b []int) []int {
	out := make([]int, 0, len(a)+len(b))
	out = append(out, a...)
	out = append(out, b...)
	return out
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}
