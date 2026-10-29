package model

import (
	"fmt"
	"math"
	"testing"
)

// This reference preserves the pre-tile scalar F32 reduction, not a new
// numerical oracle. Existing pinned-reference witnesses remain independent.
func TestV41IndexerKeyTileExactScalarOrder(t *testing.T) {
	for _, heads := range []int{1, 3, 8} {
		for _, dim := range []int{1, 7, 128} {
			for _, rows := range []int{0, 1, 3, 4, 5, 7, 8, 9, 33, 513} {
				t.Run(fmt.Sprintf("h%d_d%d_r%d", heads, dim, rows), func(t *testing.T) {
					q, keys, weights := make([]float32, heads*dim), make([]float32, rows*dim), make([]float32, heads)
					values := []float32{0, math.Float32frombits(0x80000000), 1, -1, 0.125, -0.25, 1e-20, 1e10, -1e10, 3.1415927}
					for i := range q {
						q[i] = values[(i*7+3)%len(values)]
					}
					for i := range keys {
						keys[i] = values[(i*3+1)%len(values)]
					}
					for i := range weights {
						weights[i] = values[(i*9+4)%len(values)]
					}
					calls := 0
					got, err := v41IndexerScoreObserved(q, keys, weights, heads, dim, rows, func(stage string) {
						if stage != "attention-indexer-score" {
							t.Fatalf("stage %q", stage)
						}
						calls++
					})
					if err != nil {
						t.Fatal(err)
					}
					wantCalls := 0
					if rows > 0 {
						wantCalls = 1
					}
					if calls != wantCalls {
						t.Fatalf("observations %d want %d", calls, wantCalls)
					}
					for row := 0; row < rows; row++ {
						var acc float32
						for h := 0; h < heads; h++ {
							var dot float32
							for d := 0; d < dim; d++ {
								dot = float32(dot + float32(q[h*dim+d]*keys[row*dim+d]))
							}
							if dot < 0 {
								dot = 0
							}
							acc = float32(acc + float32(dot*weights[h]))
						}
						if math.Float32bits(got[row]) != math.Float32bits(acc) {
							t.Fatalf("row %d got %08x want %08x", row, math.Float32bits(got[row]), math.Float32bits(acc))
						}
					}
				})
			}
		}
	}
}

func TestV41IndexerKeyTileRefusalBeforeObservation(t *testing.T) {
	for _, site := range []string{"weight", "query", "key"} {
		t.Run(site, func(t *testing.T) {
			q, keys, w := []float32{1, 2}, make([]float32, 18), []float32{1}
			switch site {
			case "weight":
				w[0] = float32(math.Inf(1))
			case "query":
				q[1] = float32(math.NaN())
			case "key":
				keys[17] = float32(math.Inf(-1))
			}
			calls := 0
			out, err := v41IndexerScoreObserved(q, keys, w, 1, 2, 9, func(string) { calls++ })
			if err == nil || out != nil || calls != 0 {
				t.Fatalf("out=%v err=%v observations=%d", out, err, calls)
			}
		})
	}
}
