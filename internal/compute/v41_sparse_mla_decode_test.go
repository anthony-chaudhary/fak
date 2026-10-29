package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func v41SparseMLAInputs(seed int64, rows, heads, dim int) (q, kv, sink []float32) {
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, span float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * span
		}
		return v
	}
	return fill(heads*dim, 1), fill(rows*dim, 1), fill(heads, 4)
}

// v41SparseMLAFloat64LSE is the binary64 logsumexp of the scaled scores alone.
func v41SparseMLAFloat64LSE(q, kv []float32, h, dim int, scale float32) float64 {
	rows := len(kv) / dim
	logits := make([]float64, rows)
	top := math.Inf(-1)
	for r := range rows {
		for d := range dim {
			logits[r] += float64(q[h*dim+d]) * float64(kv[r*dim+d])
		}
		logits[r] *= float64(scale)
		top = math.Max(top, logits[r])
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(v - top)
	}
	return top + math.Log(sum)
}

func v41SparseMLAMaxDiff(a, b []float32) float64 {
	var worst float64
	for i := range a {
		worst = math.Max(worst, math.Abs(float64(a[i])-float64(b[i])))
	}
	return worst
}

// The FlashMLA-class reference (tiled online softmax, split-KV combine, sink
// folded last) must stay within a pinned F32 reordering distance of the
// binary64 oracle and of the serial three-pass control, at the V4.1 head width
// and across tile/split partitions that straddle the 64-row tile boundary.
// fak-test:runtime fast est=2s lane=default
func TestV41SparseMLADecodeMatchesSerialOracles(t *testing.T) {
	const (
		dim       = 512
		outBound  = 2e-6 // max |out - oracle| over outputs bounded by max|kv| = 1
		lseBound  = 2e-6 // relative LSE error vs binary64
		ctrlBound = 2e-6 // max |out - serial three-pass F32 control|
	)
	scale := float32(1 / math.Sqrt(dim))
	for _, tc := range []struct{ rows, heads int }{{1, 4}, {63, 4}, {64, 4}, {65, 4}, {130, 4}, {640, 2}, {2049, 1}} {
		for _, hasSink := range []bool{false, true} {
			for _, opt := range []v41SparseMLADecodeOptions{{64, 1}, {64, 2}, {64, 7}, {16, 3}, {1, 5}} {
				name := fmt.Sprintf("rows%d_sink%t_tile%d_split%d", tc.rows, hasSink, opt.TileRows, opt.Splits)
				t.Run(name, func(t *testing.T) {
					q, kv, sink := v41SparseMLAInputs(int64(tc.rows*31+opt.Splits), tc.rows, tc.heads, dim)
					var useSink []float32
					if hasSink {
						useSink = sink
					}
					got, err := v41SparseMLADecodeReference(q, kv, useSink, tc.rows, tc.heads, dim, scale, hasSink, opt)
					if err != nil {
						t.Fatal(err)
					}
					want := v41SharedAttentionFloat64Oracle(q, kv, useSink, tc.heads, dim, scale)
					if diff := v41SparseMLAMaxDiff(got.Out, want); diff > outBound {
						t.Fatalf("max |out-oracle| = %g, bound %g", diff, outBound)
					}
					control, err := v41SharedAttentionF32Control(q, kv, useSink, tc.heads, dim, scale, V41SharedAttentionPlain)
					if err != nil {
						t.Fatal(err)
					}
					if diff := v41SparseMLAMaxDiff(got.Out, control); diff > ctrlBound {
						t.Fatalf("max |out-control| = %g, bound %g", diff, ctrlBound)
					}
					for h := range tc.heads {
						ref := v41SparseMLAFloat64LSE(q, kv, h, dim, scale)
						if rel := math.Abs(float64(got.LSE[h])-ref) / math.Max(1, math.Abs(ref)); rel > lseBound {
							t.Fatalf("head %d lse %g, binary64 %g (rel %g)", h, got.LSE[h], ref, rel)
						}
					}
				})
			}
		}
	}
}

// One full V4.1 Flash decode head group (64 heads x 512) at a compressed-tier
// selection width, so the pinned bound is witnessed at the shipped geometry.
// fak-test:runtime fast est=2s lane=default
func TestV41SparseMLADecodeFullHeadGroup(t *testing.T) {
	const heads, dim, rows = 64, 512, 640
	scale := float32(1 / math.Sqrt(dim))
	q, kv, sink := v41SparseMLAInputs(41, rows, heads, dim)
	got, err := v41SparseMLADecodeReference(q, kv, sink, rows, heads, dim, scale, true, v41SparseMLADecodeOptions{TileRows: 64, Splits: 4})
	if err != nil {
		t.Fatal(err)
	}
	if diff := v41SparseMLAMaxDiff(got.Out, v41SharedAttentionFloat64Oracle(q, kv, sink, heads, dim, scale)); diff > 2e-6 {
		t.Fatalf("max |out-oracle| = %g", diff)
	}
}

// A dominant sink drives the output toward zero through the final rescale
// alone, and an empty selection is zero output with LSE -Inf (no split runs).
// fak-test:runtime fast est=1s lane=default
func TestV41SparseMLADecodeSinkAndEmpty(t *testing.T) {
	const heads, dim, rows = 2, 8, 5
	q, kv, _ := v41SparseMLAInputs(7, rows, heads, dim)
	sink := []float32{40, -40}
	got, err := v41SparseMLADecodeReference(q, kv, sink, rows, heads, dim, 0.5, true, v41SparseMLADecodeOptions{TileRows: 2, Splits: 2})
	if err != nil {
		t.Fatal(err)
	}
	want := v41SharedAttentionFloat64Oracle(q, kv, sink, heads, dim, 0.5)
	if diff := v41SparseMLAMaxDiff(got.Out, want); diff > 1e-6 {
		t.Fatalf("sink-dominated max diff %g", diff)
	}
	for d := range dim {
		if math.Abs(float64(got.Out[d])) > 1e-15 {
			t.Fatalf("dominant sink left out[%d]=%g", d, got.Out[d])
		}
	}
	empty, err := v41SparseMLADecodeReference(q, nil, sink, 0, heads, dim, 0.5, true, v41SparseMLADecodeOptions{TileRows: 64, Splits: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range empty.Out {
		if v != 0 {
			t.Fatalf("empty selection produced %g", v)
		}
	}
	for _, v := range empty.LSE {
		if !math.IsInf(float64(v), -1) {
			t.Fatalf("empty selection lse %g, want -Inf", v)
		}
	}
}

// Malformed requests are geometry refusals; an overflowing finite score is a
// typed arithmetic fault at the score stage with no output.
// fak-test:runtime fast est=1s lane=default
func TestV41SparseMLADecodeRefusals(t *testing.T) {
	q, kv, sink := v41SparseMLAInputs(3, 4, 2, 4)
	ok := v41SparseMLADecodeOptions{TileRows: 2, Splits: 1}
	nan := float32(math.NaN())
	for name, call := range map[string]func() error{
		"zero heads": func() error { _, err := v41SparseMLADecodeReference(q, kv, sink, 4, 0, 4, 1, true, ok); return err },
		"zero tile": func() error {
			_, err := v41SparseMLADecodeReference(q, kv, sink, 4, 2, 4, 1, true, v41SparseMLADecodeOptions{Splits: 1})
			return err
		},
		"zero splits": func() error {
			_, err := v41SparseMLADecodeReference(q, kv, sink, 4, 2, 4, 1, true, v41SparseMLADecodeOptions{TileRows: 1})
			return err
		},
		"short kv": func() error {
			_, err := v41SparseMLADecodeReference(q, kv[:15], sink, 4, 2, 4, 1, true, ok)
			return err
		},
		"short sink": func() error { _, err := v41SparseMLADecodeReference(q, kv, sink[:1], 4, 2, 4, 1, true, ok); return err },
		"zero scale": func() error { _, err := v41SparseMLADecodeReference(q, kv, sink, 4, 2, 4, 0, true, ok); return err },
		"nan scale":  func() error { _, err := v41SparseMLADecodeReference(q, kv, sink, 4, 2, 4, nan, true, ok); return err },
		"nan query": func() error {
			bad := append([]float32(nil), q...)
			bad[3] = nan
			_, err := v41SparseMLADecodeReference(bad, kv, sink, 4, 2, 4, 1, true, ok)
			return err
		},
		"nan sink": func() error {
			bad := []float32{0, nan}
			_, err := v41SparseMLADecodeReference(q, kv, bad, 4, 2, 4, 1, true, ok)
			return err
		},
		"negative rows": func() error { _, err := v41SparseMLADecodeReference(q, kv, sink, -1, 2, 4, 1, true, ok); return err },
	} {
		if err := call(); !errors.Is(err, errV41SparseMLADecodeGeometry) {
			t.Fatalf("%s: err %v, want geometry refusal", name, err)
		}
	}
	big := []float32{3e38, 3e38}
	_, err := v41SparseMLADecodeReference(big, big, nil, 1, 1, 2, 1, false, v41SparseMLADecodeOptions{TileRows: 1, Splits: 1})
	var fault *v41SparseMLADecodeError
	if !errors.As(err, &fault) || fault.Stage != "scaled score" || fault.Head != 0 || fault.Row != 0 {
		t.Fatalf("overflow err %v, want score-stage arithmetic fault at head 0 row 0", err)
	}
}
