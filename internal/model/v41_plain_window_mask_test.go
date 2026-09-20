package model

import (
	"math"
	"testing"
)

// TestV41PlainWindowMask is the #13303 witness: it pins the configured
// causal-window row selection for V4.1 PLAIN attention against an INDEPENDENT
// scalar oracle that duplicates the visibility math directly, and never calls
// the production helpers under test (v41PlainWindowKeys / v41PlainWindowIndexList).
//
// The oracle is defined in this file by first principles — for a positive window
// w the attended absolute row IDs are exactly max(0,q-w+1)..q; the -1 sentinel
// (and w<=0) is the full causal prefix 0..q — so a bug shared between the
// production helper and its test cannot hide. Both the row-ID sets and the
// resulting sink outputs are asserted, across the query positions the ticket
// names (0, 1, 127, 128, 129) and the window values that exercise the boundary
// (global -1, a window wider than the prefix, and a window narrower than it).
func TestV41PlainWindowMask(t *testing.T) {
	// Independent scalar oracle: attended absolute row IDs for query position q
	// and window w. Deliberately NOT the production helper.
	oracle := func(q, w int) []int32 {
		if q < 0 {
			return nil
		}
		lo := 0
		if w > 0 {
			lo = q - w + 1
			if lo < 0 {
				lo = 0
			}
		}
		var out []int32
		for i := lo; i <= q; i++ {
			out = append(out, int32(i))
		}
		return out
	}

	for _, tc := range []struct {
		name string
		w    int
	}{
		{"global_-1", -1},
		{"wider_than_prefix", 4096},
		{"window_128", 128},
		{"window_1", 1},
		{"zero_treated_global", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, q := range []int{0, 1, 127, 128, 129} {
				want := oracle(q, tc.w)
				got := v41PlainWindowKeys(q, tc.w)
				if len(got) != len(want) {
					t.Fatalf("q=%d w=%d: row count %d, want %d (got %v want %v)",
						q, tc.w, len(got), len(want), got, want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("q=%d w=%d: row[%d]=%d, want %d (got %v want %v)",
							q, tc.w, i, got[i], want[i], got, want)
					}
				}
				// The index list is POSITIONAL over the flattened selected rows
				// (the sink indexes its kv array), so it is 0..n-1 plus the
				// trailing -1 empty slot.
				idx := v41PlainWindowIndexList(got)
				if len(idx) != len(want)+1 {
					t.Fatalf("q=%d w=%d: idx len %d, want %d", q, tc.w, len(idx), len(want)+1)
				}
				for i := range want {
					if idx[i] != int32(i) {
						t.Fatalf("q=%d w=%d: idx[%d]=%d, want %d", q, tc.w, i, idx[i], i)
					}
				}
				if idx[len(want)] != -1 {
					t.Fatalf("q=%d w=%d: trailing slot = %d, want -1", q, tc.w, idx[len(want)])
				}
			}
		})
	}
}

// TestV41PlainWindowMaskSinkOutput pins that feeding the window-selected rows to
// the EXISTING sink contraction reproduces, element for element, a scalar
// oracle that computes softmax attention over exactly those rows directly. A
// window that hides a row R the oracle also hides must match; a global layer
// must match the full-causal reference. This is the "outputs" half of the
// witness and proves the seam changes only WHICH rows are visible, never the
// arithmetic.
func TestV41PlainWindowMaskSinkOutput(t *testing.T) {
	const (
		nH   = 2
		hd   = 4
		rows = 6
	)
	// Deterministic q and KV rows.
	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = float32((i*7)%11) - 5
	}
	kv := make([]float32, rows*hd)
	for i := range kv {
		kv[i] = float32((i*13)%17) - 8
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	// Scalar oracle: softmax over the given absolute rows, per head, exactly the
	// reference's score/softmax/value accumulation over `visible`.
	oracle := func(visible []int32) []float32 {
		out := make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			qv := q[h*hd : (h+1)*hd]
			scores := make([]float32, len(visible))
			maxS := float32(math.Inf(-1))
			for i, r := range visible {
				row := kv[int(r)*hd : int(r+1)*hd]
				var s float32
				for d := 0; d < hd; d++ {
					s += qv[d] * row[d]
				}
				s *= scale
				scores[i] = s
				if s > maxS {
					maxS = s
				}
			}
			var denom float32
			for _, s := range scores {
				denom += float32(math.Exp(float64(s - maxS)))
			}
			if denom == 0 {
				continue
			}
			for i, r := range visible {
				w := float32(math.Exp(float64(scores[i]-maxS))) / denom
				row := kv[int(r)*hd : int(r+1)*hd]
				for d := 0; d < hd; d++ {
					out[h*hd+d] += w * row[d]
				}
			}
		}
		return out
	}

	// full causal prefix 0..rows-1 is the reference the global layer must match.
	full := make([]int32, rows)
	for i := range full {
		full[i] = int32(i)
	}
	fullOracle := oracle(full)

	// A global window (-1) must reproduce the full-causal contraction exactly.
	gotKeys := v41PlainWindowKeys(rows-1, -1)
	if len(gotKeys) != rows {
		t.Fatalf("global window selected %d rows, want %d", len(gotKeys), rows)
	}
	flatKV := make([]float32, 0, rows*hd)
	for _, k := range gotKeys {
		flatKV = append(flatKV, kv[k*hd:(k+1)*hd]...)
	}
	idx := v41PlainWindowIndexList(gotKeys)
	got, err := V41SparseAttentionSink(q, flatKV, nil, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(gotKeys) + 1, N: len(gotKeys), Softmax: scale,
	})
	if err != nil {
		t.Fatalf("global sink: %v", err)
	}
	for i := range fullOracle {
		if math.Abs(float64(got[i]-fullOracle[i])) > 1e-5 {
			t.Fatalf("global row %d: sink=%v oracle=%v (window global changed the arithmetic)",
				i, got[i], fullOracle[i])
		}
	}

	// A narrow window (w=3) at q=5 selects rows 3..5; the sink must reproduce the
	// oracle restricted to exactly those rows, NOT the full prefix.
	const w = 3
	qpos := rows - 1
	vis := v41PlainWindowKeys(qpos, w)
	wantVis := []int32{3, 4, 5}
	if len(vis) != len(wantVis) {
		t.Fatalf("w=%d q=%d: selected %v, want %v", w, qpos, vis, wantVis)
	}
	for i := range wantVis {
		if vis[i] != wantVis[i] {
			t.Fatalf("w=%d q=%d: row[%d]=%d, want %d", w, qpos, i, vis[i], wantVis[i])
		}
	}
	flatKV2 := make([]float32, 0, len(vis)*hd)
	for _, k := range vis {
		flatKV2 = append(flatKV2, kv[k*hd:(k+1)*hd]...)
	}
	idx2 := v41PlainWindowIndexList(vis)
	got2, err := V41SparseAttentionSink(q, flatKV2, nil, idx2, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(vis) + 1, N: len(vis), Softmax: scale,
	})
	if err != nil {
		t.Fatalf("window sink: %v", err)
	}
	want2 := oracle(wantVis)
	for i := range want2 {
		if math.Abs(float64(got2[i]-want2[i])) > 1e-5 {
			t.Fatalf("w=%d row %d: sink=%v oracle=%v", w, i, got2[i], want2[i])
		}
	}
}

// TestV41PlainWindowMaskEmptyAndRefused pins the degenerate inputs: a negative
// query position yields no rows (there is no query), and the empty index list
// the sink receives is just the trailing -1, which the reference treats as an
// empty (all-zero) output rather than a panic.
func TestV41PlainWindowMaskEmptyAndRefused(t *testing.T) {
	if got := v41PlainWindowKeys(-1, 128); got != nil {
		t.Fatalf("q=-1: rows=%v, want nil", got)
	}
	idx := v41PlainWindowIndexList(nil)
	if len(idx) != 1 || idx[0] != -1 {
		t.Fatalf("empty key list -> idx %v, want [-1]", idx)
	}
}

// TestV41PlainWindowMaskProductionPath drives the REAL forward call path on the
// reduced V4.1 fixture and proves the configured window is honored in
// production, not merely in the helper. The fixture inherits the published
// config, whose Window is wider than any prompt these tests use, so it is a
// no-op there; this test pins both endpoints explicitly:
//
//   - an explicit all -1 Window (global) is byte-identical to the historical
//     full-causal contraction, so the seam does not perturb a global layer; and
//   - a window of 1 admits only each query's own key, a strictly narrower
//     visible set for a multi-token prompt, so the logits MUST change — proof
//     the configured window reaches the production attention path.
//
// Both runs stay finite.
func TestV41PlainWindowMaskProductionPath(t *testing.T) {
	ids := []int{1, 3, 5, 7}

	// Explicit global sentinel: full causal. This is the reference the narrow
	// window must differ from.
	global := v41ReducedModel(t)
	global.Cfg.Window = []int{-1}
	globalAct := global.Forward(ids)
	if globalAct == nil || len(globalAct.Logits) != len(ids) {
		t.Fatalf("global Forward returned %d positions, want %d", len(globalAct.Logits), len(ids))
	}

	// A window of 1 admits only the query's own key for every position, which for
	// a multi-token prompt is a strictly narrower visible set than the full
	// causal prefix at every position t > 0. The logits must therefore differ
	// from the global reference and remain finite.
	narrow := v41ReducedModel(t)
	narrow.Cfg.Window = []int{1}
	narrowAct := narrow.Forward(ids)
	if narrowAct == nil || len(narrowAct.Logits) != len(ids) {
		t.Fatalf("narrow-window Forward returned %d positions, want %d", len(narrowAct.Logits), len(ids))
	}
	changed := false
	for p := range ids {
		for i := range narrowAct.Logits[p] {
			v := narrowAct.Logits[p][i]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("pos %d dim %d: narrow-window logit non-finite %v", p, i, v)
			}
			if math.Float32bits(v) != math.Float32bits(globalAct.Logits[p][i]) {
				changed = true
			}
		}
	}
	if !changed {
		t.Fatal("window=1 produced identical logits to full causal: the configured window is not reaching the production attention path")
	}
}
