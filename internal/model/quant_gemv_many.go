package model

// quant_gemv_many.go — grouped decode GEMV. The single-session quantized decode does q/k/v as three
// separate qMatRowsInto calls and gate/up as two; each pays its own parFor dispatch, and the small
// projections (k/v are only 256 rows) parallelize poorly split 12 ways. qMatRowsIntoMany runs a set
// of GEMVs that share ONE quantized activation under a SINGLE parFor over the CONCATENATED output-row
// space, so k/v fold into the q sweep (2048 rows instead of 256) and the dispatch is paid once.
//
// Bit-identity: y[o] = qdot8GEMV(weight row o, qv) is independent of every other output row and of
// the grouping, so this produces byte-for-byte the same outputs as calling qMatRowsInto on each
// target separately (TestQMatRowsIntoManyMatchesSeparate). Only the dispatch is shared.

type qMatTarget struct {
	qt  *q8Tensor
	dst []float32
}

func qMatRowsIntoMany(qv q8Vec, targets ...qMatTarget) {
	total := 0
	for _, tg := range targets {
		total += tg.qt.out
	}
	if total == 0 {
		return
	}
	body := func(lo, hi int) {
		base := 0
		for _, tg := range targets {
			end := base + tg.qt.out
			a, b := lo, hi
			if a < base {
				a = base
			}
			if b > end {
				b = end
			}
			if a < b {
				qMatRowsRange(tg.qt, qv, tg.dst[:tg.qt.out], a-base, b-base)
			}
			base = end
		}
	}
	workers := q8DecodeWorkers()
	if workers <= 1 || total*targets[0].qt.in < parThreshold {
		body(0, total)
		return
	}
	parFor(total, workers, body)
}
