package model

import (
	"math"
	"reflect"
	"testing"
)

// The unchanged wrapper is the fresh-flat control: both arms execute the same
// scalar projections, score reduction and selection. No hardware rate is proved.
// fak-test:runtime medium est=5s lane=default
func TestV41ColdIndexerKeyPanel(t *testing.T) {
	m := v41IndexerTestFixture(t)
	dim := m.Cfg.IndexHeadDim
	qLat, hidden := make([]float32, m.Cfg.QLoraRank), make([]float32, m.Cfg.HiddenSize)
	for i := range qLat {
		qLat[i] = float32(i%7-3) / 8
	}
	for i := range hidden {
		hidden[i] = float32(i%5-2) / 8
	}
	keys := make([][]float32, 8)
	for r := range keys {
		keys[r] = make([]float32, dim)
		for d := range keys[r] {
			keys[r][d] = float32((r*7+d)%19-9) / 16
		}
	}
	selectRows := func(pos, groups int, panel []float32, score v41IndexerScoreFunc) ([]int32, error) {
		return m.v41IndexRowsWithPreparedKeys(0, pos, qLat, hidden, keys[:groups], nil, score, nil, nil, panel)
	}

	t.Run("causal-prefixes-and-allocations", func(t *testing.T) {
		const seq, ratio = 16, 2
		panel := v41ColdIndexerKeyPanel(keys, dim)
		if len(panel) != len(keys)*dim || cap(panel) != len(panel) || &panel[0] == &keys[0][0] {
			t.Fatal("key snapshot is not independently owned and exactly bounded")
		}
		oldCopyElements := 0
		for pos := 0; pos < seq; pos++ {
			groups := (pos + 1) / ratio
			end := groups * dim
			prefix := panel[:end:end]
			if len(prefix) != end || cap(prefix) != end {
				t.Fatal("causal prefix exposes a future row")
			}
			got, err := selectRows(pos, groups, prefix, nil)
			want, wantErr := m.v41IndexRowsWithDeviceOps(0, pos, qLat, hidden, keys[:groups], nil, nil, nil, nil)
			if err != nil || wantErr != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("query %d: rows=%v want=%v errors=%v/%v", pos, got, want, err, wantErr)
			}
			oldCopyElements += end
		}
		run := func(prepared bool) {
			var packed []float32
			if prepared {
				packed = v41ColdIndexerKeyPanel(keys, dim)
			}
			for pos := 0; pos < seq; pos++ {
				groups := (pos + 1) / ratio
				var err error
				if prepared {
					end := groups * dim
					_, err = selectRows(pos, groups, packed[:end:end], nil)
				} else {
					_, err = m.v41IndexRowsWithDeviceOps(0, pos, qLat, hidden, keys[:groups], nil, nil, nil, nil)
				}
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		freshAllocs := testing.AllocsPerRun(5, func() { run(false) })
		panelAllocs := testing.AllocsPerRun(5, func() { run(true) })
		// Fifteen nonempty query flats are replaced by one layer allocation.
		if got, want := freshAllocs-panelAllocs, float64(seq-ratio); got != want {
			t.Fatalf("fresh=%g panel=%g allocations, reduction=%g want=%g", freshAllocs, panelAllocs, got, want)
		}
		t.Logf("actual allocations: fresh=%g panel=%g; key payload copy bound: fresh=%d panel=%d float32 elements", freshAllocs, panelAllocs, oldCopyElements, len(panel))
		before := panel[0]
		keys[0][0]++
		if panel[0] != before {
			t.Fatal("source mutation changed the key snapshot")
		}
		keys[0][0]--
		panel[0]++
		if keys[0][0] != before {
			t.Fatal("snapshot mutation changed source storage")
		}
	})

	t.Run("future-poison-and-late-malformed-key", func(t *testing.T) {
		original := keys[7]
		defer func() { keys[7] = original }()
		keys[7] = append([]float32(nil), original...)
		keys[7][0] = float32(math.NaN())
		panel := v41ColdIndexerKeyPanel(keys, dim)
		for _, groups := range []int{7, 8} {
			end := groups * dim
			got, err := selectRows(15, groups, panel[:end:end], nil)
			want, wantErr := m.v41IndexRowsWithDeviceOps(0, 15, qLat, hidden, keys[:groups], nil, nil, nil, nil)
			if groups == 7 && (err != nil || wantErr != nil || !reflect.DeepEqual(got, want)) {
				t.Fatalf("invisible poisoned row affected query: %v/%v", err, wantErr)
			}
			if groups == 8 && (err == nil || wantErr == nil || err.Error() != wantErr.Error()) {
				t.Fatalf("visible poisoned row changed refusal: %v/%v", err, wantErr)
			}
		}
		keys[7] = original[:dim-1]
		panel = v41ColdIndexerKeyPanel(keys, dim)
		if panel != nil {
			t.Fatal("malformed future key did not decline panel reuse")
		}
		for _, groups := range []int{7, 8} {
			got, err := selectRows(15, groups, panel, nil)
			want, wantErr := m.v41IndexRowsWithDeviceOps(0, 15, qLat, hidden, keys[:groups], nil, nil, nil, nil)
			if groups == 7 && (err != nil || wantErr != nil || !reflect.DeepEqual(got, want)) {
				t.Fatalf("malformed later key changed earlier visible query: %v/%v", err, wantErr)
			}
			if groups == 8 && (err == nil || wantErr == nil || err.Error() != wantErr.Error()) {
				t.Fatalf("malformed key changed visible-query refusal: %v/%v", err, wantErr)
			}
		}
	})

	t.Run("mutable-callback-keeps-independent-inputs", func(t *testing.T) {
		panel := v41ColdIndexerKeyPanel(keys, dim)
		original := append([]float32(nil), panel...)
		var retained [][]float32
		score := func(_ int, q, flat, weights []float32, heads, width, rows int) ([]float32, error) {
			if !reflect.DeepEqual(flat, original) {
				t.Fatal("prior callback mutation reached a later query")
			}
			out, err := V41IndexerScore(q, flat, weights, heads, width, rows)
			retained = append(retained, flat)
			flat[0] = float32(math.Inf(1))
			return out, err
		}
		first, err := selectRows(15, len(keys), panel, score)
		second, secondErr := selectRows(15, len(keys), panel, score)
		if err != nil || secondErr != nil || !reflect.DeepEqual(first, second) || len(retained) != 2 || &retained[0][0] == &retained[1][0] || !reflect.DeepEqual(panel, original) {
			t.Fatalf("callback independence changed: %v/%v", err, secondErr)
		}
		if keys[0][0] != original[0] {
			t.Fatal("callback mutation reached original key storage")
		}
	})

	t.Run("cold-step-suffix-state-parity", func(t *testing.T) {
		for _, count := range []int{1, 3, 16} {
			left, right := m.NewSession(), m.NewSession()
			t.Cleanup(left.Close)
			t.Cleanup(right.Close)
			right.v41State().indexerScore = func(_ int, q, flat, weights []float32, heads, width, rows int) ([]float32, error) {
				return V41IndexerScore(q, flat, weights, heads, width, rows)
			}
			prefix, next := v41IncrementalPrefix(m.Cfg, count)
			check := func(got, want []float32) {
				v41ScoreTestBits(t, got, want)
				if !reflect.DeepEqual(captureV41ForwardSnapshot(left.v41Forward), captureV41ForwardSnapshot(right.v41Forward)) {
					t.Fatalf("prefix %d changed retained KV/index/top-k state", count)
				}
			}
			check(left.Prefill(prefix), right.Prefill(prefix))
			check(left.Step(next), right.Step(next))
			check(left.Prefill([]int{2, 5}), right.Prefill([]int{2, 5}))
		}
	})
}
