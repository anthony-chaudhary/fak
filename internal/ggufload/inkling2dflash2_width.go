package ggufload

import (
	"os"
	"strconv"
)

// inkling2DFlash2Cost is a step's cost model in units of one width-1 verify:
// a verify of width w costs 1+(w-1)*row, and a step that drafts also pays
// draft. The numbers are measured constants, never live timings, so a
// request's draft lengths and therefore its ids are the same on every run.
type inkling2DFlash2Cost struct{ row, draft float64 }

// Measured on an M3 Pro: a width-1 verify is 43.8 ms and each extra row 13.7
// ms on the GEMV path (w8 139.5 ms); on the MMA path 8.4 ms per row of a 40.8
// ms width-1. The fused draft, its fence included, is 11.9 ms in the loop.
var (
	inkling2DFlash2CostGEMV = inkling2DFlash2Cost{row: 13.7 / 43.8, draft: 11.9 / 43.8}
	inkling2DFlash2CostMMA  = inkling2DFlash2Cost{row: 8.4 / 40.8, draft: 11.9 / 43.8}
)

func (c inkling2DFlash2Cost) step(n int) float64 {
	v := 1 + float64(n)*c.row
	if n > 0 {
		v += c.draft
	}
	return v
}

// inkling2DFlash2CostTable is the table of the verify path this process runs.
func inkling2DFlash2CostTable() inkling2DFlash2Cost {
	if os.Getenv("FAK_INKLING2_VERIFY_MMA") == "1" {
		return inkling2DFlash2CostMMA
	}
	return inkling2DFlash2CostGEMV
}

// inkling2DFlash2FixedDepth is FAK_INKLING2_DFLASH2_DEPTH_FIXED, a depth every
// step drafts regardless of acceptance (for A/B against the chooser), or -1.
func inkling2DFlash2FixedDepth() int {
	v := os.Getenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED")
	if v == "" {
		return -1
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return -1
	}
	return n
}

const (
	// Old outcomes fade so one bad phase does not end drafting for the reply,
	// and a refused step fades them too, so acceptance drifts back to the prior
	// and a collapsed phase is probed again.
	inkling2DFlash2WidthDecay = 0.9
	// a draft must beat a width-1 verify by this much to pay for the gamble
	inkling2DFlash2WidthMargin = 1.05
)

// inkling2DFlash2WidthPrior is each position's accept:reject pseudo-count, the
// conditional acceptance the DFlash2 draft shows across code, chat and math.
var inkling2DFlash2WidthPrior = [2]float64{1.4, 0.6}

// inkling2DFlash2Width picks how many tokens a DFlash2 step drafts: the n in
// [0, max] with the most expected tokens per unit of cost, from decayed
// per-position acceptance (position j accepted given j-1 were).
type inkling2DFlash2Width struct {
	max      int
	fixed    int
	cost     inkling2DFlash2Cost
	acc, rej [inkling2SpecMaxDraft]float64
	pending  int // tokens drafted by the step whose outcome is not yet observed
}

func newInkling2DFlash2Width(depth int) *inkling2DFlash2Width {
	return &inkling2DFlash2Width{max: min(depth, inkling2SpecMaxDraft), fixed: inkling2DFlash2FixedDepth(), cost: inkling2DFlash2CostTable()}
}

// p is position j's conditional acceptance.
func (w *inkling2DFlash2Width) p(j int) float64 {
	pa, pr := inkling2DFlash2WidthPrior[0], inkling2DFlash2WidthPrior[1]
	return (w.acc[j] + pa) / (w.acc[j] + w.rej[j] + pa + pr)
}

// expect is the tokens a step of n drafted tokens emits: 1+sum_i prod_{j<=i} p_j.
func (w *inkling2DFlash2Width) expect(n int) float64 {
	e, pi := 1.0, 1.0
	for j := 0; j < n; j++ {
		pi *= w.p(j)
		e += pi
	}
	return e
}

// choose returns the draft length of the next step and remembers it for
// observe; 0 means a width-1 verify with no draft.
func (w *inkling2DFlash2Width) choose() int {
	n := 0
	if w.fixed >= 0 {
		n = min(w.fixed, w.max)
	} else {
		best := inkling2DFlash2WidthMargin
		for k := 1; k <= w.max; k++ {
			if r := w.expect(k) / w.cost.step(k); r > best {
				n, best = k, r
			}
		}
		if n == 0 {
			w.fade()
		}
	}
	w.pending = n
	return n
}

func (w *inkling2DFlash2Width) fade() {
	for j := range w.acc {
		w.acc[j] *= inkling2DFlash2WidthDecay
		w.rej[j] *= inkling2DFlash2WidthDecay
	}
}

// observe records that of the n tokens drafted last, accepted were kept:
// positions before accepted were accepted, position accepted was rejected,
// and later ones were never judged.
func (w *inkling2DFlash2Width) observe(drafted, accepted int) {
	if drafted <= 0 {
		return
	}
	w.fade()
	accepted = min(max(accepted, 0), drafted)
	for j := 0; j < accepted; j++ {
		w.acc[j]++
	}
	if accepted < drafted {
		w.rej[accepted]++
	}
}
