package model

import (
	"math"
	"testing"
)

// TestQMatRowsIntoManyMatchesSeparate proves the grouped decode GEMV is byte-for-byte identical to
// calling qMatRowsInto on each target separately (the q/k/v and gate/up groups the decode path uses).
func TestQMatRowsIntoManyMatchesSeparate(t *testing.T) {
	const in = 1536
	x := mkVec(in, 12345)
	qv := quantizeVecQ8(x)
	// q/k/v-shaped group (1536 / 256 / 256 rows) and a gate/up-shaped group (8960 / 8960).
	for _, outs := range [][]int{{1536, 256, 256}, {8960, 8960}, {1, 7, 3}, {64}} {
		targets := make([]qMatTarget, len(outs))
		sep := make([][]float32, len(outs))
		for i, out := range outs {
			w := mkVec(out*in, uint64(out*1009+i*31+7))
			qt := quantizeQ8(w, out, in)
			targets[i] = qMatTarget{qt: qt, dst: make([]float32, out)}
			sep[i] = qMatRows(qt, qv) // the per-target reference
		}
		qMatRowsIntoMany(qv, targets...)
		for i := range outs {
			for o := 0; o < outs[i]; o++ {
				if math.Float32bits(targets[i].dst[o]) != math.Float32bits(sep[i][o]) {
					t.Fatalf("outs=%v target=%d row=%d: grouped %08x != separate %08x",
						outs, i, o, math.Float32bits(targets[i].dst[o]), math.Float32bits(sep[i][o]))
				}
			}
		}
	}
}
