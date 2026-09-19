package model

// v41_attention_nonfinite_guard_test.go - the permanent RED->GREEN regression
// for fak#13290: the layer-36 non-finite compressed/sparse attention output.
//
// Physical failure (strix3, trunk 193c17360):
//
//	panic: model: V4.1 forward stage attention layer=36:
//	  model: V4.1 grouped output non-finite value element 0
//
// The layer-36 schedule resolves Ratio=1, is an index source, and its KV source
// (layer 20) is ALSO ratio 1, so sharedKV stays nil and layer 36 executes the
// PER-POSITION V41SparseAttentionSink path (witnessed structurally below). Both
// V41SparseAttentionSink and V41AttentionCompressedForward validated every
// INPUT was finite but never guarded their OUTPUT: with HeadDim=512 a finite
// per-element magnitude above ~8.15e17 overflows the 512-term `dot += q*kv`
// accumulation to +Inf; `dot*Softmax` stays +Inf and becomes maxScore=+Inf;
// exp32(dot*Softmax-maxScore) = exp32(Inf-Inf) = exp32(NaN) = NaN; `sum` becomes
// NaN (the `sum == 0` guard does not fire on NaN); weight=NaN/NaN=NaN poisons
// the head row; and the failure then surfaced MISLEADINGLY as a downstream
// V41GroupedOutputProjection "non-finite value element 0" refusal - naming the
// wrong producer.
//
// POST-FIX WITNESS. These tests assert the new fail-closed behavior: the
// contraction itself refuses with an error that NAMES the true producing stage
// (score accumulate / softmax denominator / weighted value accumulate), the
// position, head and offending element. The "finite inputs still match the
// scalar oracle" sub-tests pin that the guard is purely additive and does NOT
// perturb finite-input arithmetic.

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// v41NonFiniteTestScale is the model's real attention scale at HeadDim=512.
func v41NonFiniteTestScale(headDim int) float32 {
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

// TestV41Layer36RealScheduleBranch witnesses the branch resolution that puts
// layer 36 on the PER-POSITION sink path (not the compressed contraction), from
// the actual published config. It is the structural precondition for the guard
// regression: it fails loudly if role/source resolution drifts.
func TestV41Layer36RealScheduleBranch(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	roles := v41AttentionRoles(cfg)

	plan, err := v41AttentionPlanFor(cfg, 36, roles)
	if err != nil {
		t.Fatalf("layer 36 plan: %v", err)
	}
	if plan.Ratio != 1 {
		t.Fatalf("layer 36 ratio = %d, want 1", plan.Ratio)
	}
	if plan.Role != V41AttentionRoleReader {
		t.Fatalf("layer 36 role = %v, want reader", plan.Role)
	}
	if plan.KVSourceLayer != 20 {
		t.Fatalf("layer 36 KV source = %d, want 20", plan.KVSourceLayer)
	}
	if !indexSourceAt(cfg.DeepSeekV41, 36) {
		t.Fatal("layer 36 must itself be an index source")
	}
	// The resolved source (20) is ALSO ratio 1, so it publishes no compressed
	// Latent; sharedKV stays nil and the compressed branch is not selected.
	if srcRatio := v41CompressRatioAt(cfg, plan.KVSourceLayer); srcRatio != 1 {
		t.Fatalf("layer %d ratio = %d; layer 36 reaches the sink path only if it is 1",
			plan.KVSourceLayer, srcRatio)
	}
	t.Log("layer 36 executes the PER-POSITION V41SparseAttentionSink path (sharedKV nil), NOT V41AttentionCompressedForward")
}

// TestV41AttentionCompressedNonFiniteGuard is the symmetric RED->GREEN witness
// for the compressed contraction (the twin latent hole). Layer 36 does not take
// this path, but the function carries the identical unguarded output.
func TestV41AttentionCompressedNonFiniteGuard(t *testing.T) {
	const (
		heads   = 64
		headDim = 512
		ratio   = 1
		groups  = 8
		seq     = 1
	)
	scale := v41NonFiniteTestScale(headDim)
	overflowMag := float32(1e20)

	q := make([]float32, seq*heads*headDim)
	for i := range q {
		q[i] = overflowMag
	}
	values := make([][]float32, groups)
	for g := range values {
		values[g] = make([]float32, headDim)
		for d := range values[g] {
			values[g][d] = overflowMag
		}
	}
	sink := make([]float32, heads)
	for h := range sink {
		sink[h] = 0.1 * float32(h+1)
	}

	out, err := V41AttentionCompressedForward(q, values, V41AttentionSharedKVOptions{
		Layer: 36, Ratio: ratio, Groups: groups, HeadDim: headDim,
		Heads: heads, Softmax: scale, Sink: sink,
	})
	if err == nil {
		t.Fatalf("compressed forward returned %d outputs from overflowing finite inputs; want a typed refusal (RED-on-parent)", len(out))
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("compressed forward error = %v, want errors.Is(err, ErrV41ForwardStage)", err)
	}
	msg := err.Error()
	for _, want := range []string{"compressed attention", "produced a non-finite value", "t=0", "h=", "layer=36", "value="} {
		if !strings.Contains(msg, want) {
			t.Fatalf("compressed refusal message %q does not name %q", msg, want)
		}
	}
	t.Logf("GREEN compressed refusal: %v", err)
}

// TestV41AttentionCompressedFiniteMatchesOracle pins that the guard does not
// perturb the finite compressed path; it reuses the independent scalar oracle
// established in v41_attention_test.go.
func TestV41AttentionCompressedFiniteMatchesOracle(t *testing.T) {
	const (
		ratio  = 2
		groups = 3
		hd     = 4
		heads  = 2
		seq    = 6
	)
	rows := [][]float32{
		{0.5, -0.25, 0.75, 0.1},
		{-0.3, 0.4, -0.6, 0.2},
		{0.15, -0.5, 0.25, -0.35},
	}
	q := make([][]float32, seq)
	for pos := 0; pos < seq; pos++ {
		row := make([]float32, heads*hd)
		for i := range row {
			row[i] = float32(math.Sin(float64(pos*7+i))) * 0.5
		}
		q[pos] = row
	}
	sink := []float32{0.25, -0.1}
	scale := float32(0.125)

	got, err := V41AttentionCompressedForward(flatten(q), rows, V41AttentionSharedKVOptions{
		Layer: 0, Ratio: ratio, Groups: groups, HeadDim: hd, Heads: heads,
		TopK: 0, Softmax: scale, Sink: sink,
	})
	if err != nil {
		t.Fatalf("V41AttentionCompressedForward (finite): %v", err)
	}
	want := flatten(v41OracleCompressedForward(t, ratio, q, rows, sink, scale))
	if len(got) != len(want) {
		t.Fatalf("output length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-5 {
			t.Fatalf("compressed out[%d] = %g, want %g (|delta| = %.3e)", i, got[i], want[i], d)
		}
	}
}
