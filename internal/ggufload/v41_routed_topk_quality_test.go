package ggufload

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// v41RoutedTopKKLGate is the mean next-token KL(top-6 || top-N) a reduced
// routed count must stay under before it may become a default or auto rule.
// It is Strata's 0.006 gate (ds4_experts_rule.hpp header @46c152a1, MIT, rule
// only; upstream-reported).
const v41RoutedTopKKLGate = 0.006

// v41RoutedTopKQualityPrompts is the fixed prompt set for the quality gate.
var v41RoutedTopKQualityPrompts = []string{
	"The capital of France is",
	"def fibonacci(n):\n    if n < 2:\n        return n\n    return",
	"Water boils at sea level at a temperature of",
	"In 1969, the first humans to walk on the Moon were",
	"The derivative of x squared with respect to x is",
	"Translate to French: Good morning, how are you?",
	"A haiku about autumn leaves:\n",
	"SELECT name, COUNT(*) FROM orders GROUP BY",
}

// v41LogSoftmax returns log-probabilities of logits in float64.
func v41LogSoftmax(logits []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	var z float64
	for _, v := range logits {
		z += math.Exp(float64(v) - mx)
	}
	lz := mx + math.Log(z)
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = float64(v) - lz
	}
	return out
}

func v41Argmax(logits []float32) int {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// v41RoutedTopKQuality reports mean KL(ref || got) over next-token
// distributions and the fraction of rows whose greedy top-1 agrees.
func v41RoutedTopKQuality(ref, got [][]float32) (meanKL, top1 float64) {
	if len(ref) == 0 || len(ref) != len(got) {
		return math.NaN(), math.NaN()
	}
	var agree int
	for i := range ref {
		if len(ref[i]) == 0 || len(ref[i]) != len(got[i]) {
			return math.NaN(), math.NaN()
		}
		lp, lq := v41LogSoftmax(ref[i]), v41LogSoftmax(got[i])
		var kl float64
		for j := range lp {
			kl += math.Exp(lp[j]) * (lp[j] - lq[j])
		}
		meanKL += kl
		if v41Argmax(ref[i]) == v41Argmax(got[i]) {
			agree++
		}
	}
	n := float64(len(ref))
	return meanKL / n, float64(agree) / n
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RoutedTopKQualityMetric(t *testing.T) {
	ref := [][]float32{{2, 1, 0, -1}, {0, 3, 1, 0}}
	if kl, top1 := v41RoutedTopKQuality(ref, ref); math.Abs(kl) > 1e-12 || top1 != 1 {
		t.Fatalf("identical rows: KL=%g top1=%g, want 0 and 1", kl, top1)
	}
	// A shifted row keeps its argmax; a swapped row loses it.
	got := [][]float32{{2, 1.2, 0, -1}, {0, 1, 3, 0}}
	kl, top1 := v41RoutedTopKQuality(ref, got)
	if !(kl > 0) || top1 != 0.5 {
		t.Fatalf("perturbed rows: KL=%g top1=%g, want KL>0 and top1=0.5", kl, top1)
	}
	// Hand-computed KL for the first row alone must match the definition.
	lp, lq := v41LogSoftmax(ref[0]), v41LogSoftmax(got[0])
	var want float64
	for j := range lp {
		want += math.Exp(lp[j]) * (lp[j] - lq[j])
	}
	if one, _ := v41RoutedTopKQuality(ref[:1], got[:1]); math.Abs(one-want) > 1e-12 {
		t.Fatalf("row KL=%g, want %g", one, want)
	}
	if kl, top1 := v41RoutedTopKQuality(nil, nil); !math.IsNaN(kl) || !math.IsNaN(top1) {
		t.Fatalf("empty set must be NaN (no verdict), got KL=%g top1=%g", kl, top1)
	}
}

// TestV41RoutedTopKQualityGate is the quality gate for FAK_V41_ROUTED_TOPK=5:
// on a real V4.1 checkpoint it prefills each fixed prompt at the published
// top-6 and at top-5 and requires mean next-token KL(top-6 || top-5) <= 0.006.
// Top-1 agreement is reported alongside. A host without the checkpoint skips;
// a skip is not a pass and does not admit 5 as a default.
//
// FAK_V41_GGUF names the checkpoint (shard 1), as in the tokenizer oracle.
//
// fak-test:runtime slow est=30m lane=optin
func TestV41RoutedTopKQualityGate(t *testing.T) {
	path := os.Getenv("FAK_V41_GGUF")
	if path == "" {
		t.Skip("FAK_V41_GGUF not set: no V4.1 checkpoint on this host, so the top-5 vs top-6 KL gate is NOT evaluated (skip is not a pass)")
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatalf("%s has no embedded tokenizer", filepath.Base(path))
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	m, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}
	if !m.Cfg.IsDeepSeekV41() {
		t.Fatalf("%s is not a DeepSeek V4.1 checkpoint", filepath.Base(path))
	}

	run := func(topK string) [][]float32 {
		t.Setenv(model.V41RoutedTopKEnv, topK)
		out := make([][]float32, 0, len(v41RoutedTopKQualityPrompts))
		for _, p := range v41RoutedTopKQualityPrompts {
			ids, err := tok.Encode(p)
			if err != nil {
				t.Fatalf("encode %q: %v", p, err)
			}
			s := m.NewSession()
			s.Quant = true
			logits := s.Prefill(ids)
			if len(logits) == 0 {
				t.Fatalf("top-%s prefill of %q returned no logits", topK, p)
			}
			out = append(out, append([]float32(nil), logits...))
		}
		return out
	}
	ref := run("6")
	got := run("5")
	kl, top1 := v41RoutedTopKQuality(ref, got)
	t.Logf("V4.1 routed top-5 vs top-6 on %s: mean KL=%.6f top-1 agreement=%.4f over %d prompts (gate KL<=%.3f)",
		filepath.Base(path), kl, top1, len(ref), v41RoutedTopKKLGate)
	if math.IsNaN(kl) || kl > v41RoutedTopKKLGate {
		t.Fatalf("top-5 quality gate failed: mean KL=%.6f > %.3f (top-1 agreement %.4f)", kl, v41RoutedTopKKLGate, top1)
	}
}
