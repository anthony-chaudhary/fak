package model

// Adapted from ashhart/TensorFold, zig/src/core/draft_depth.zig
// (Costs / DepthRule) at revision 9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// (Apache-2.0): https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/draft_depth.zig
//
// Re-expressed in idiomatic Go over fak's MTP depth-governor seam; no upstream
// code blocks are copied. See THIRD_PARTY_NOTICES.md ("TensorFold draft-depth
// pooled verify costs").

import (
	"errors"
	"math"
)

// DraftDepthMaxRows is the widest verify window a draft-depth cost table holds.
const DraftDepthMaxRows = 32

var (
	// ErrDraftDepthVerifyTable refuses measured verify tables that are too
	// short to price a two-row window or wider than DraftDepthMaxRows.
	ErrDraftDepthVerifyTable = errors.New("model: draft depth verify table out of shape")
	// ErrDraftDepthRegimeInvalid refuses a kernel-regime width below 1.
	ErrDraftDepthRegimeInvalid = errors.New("model: draft depth regime must be at least 1")
	// ErrDraftDepthMostOutOfRange refuses a deepest-draft bound outside
	// [1, Rows-1].
	ErrDraftDepthMostOutOfRange = errors.New("model: draft depth most out of range")
	// ErrDraftDepthPowerInvalid refuses a non-positive confidence power.
	ErrDraftDepthPowerInvalid = errors.New("model: draft depth power must be positive")
)

// roundDraftDepth4 rounds to four decimal places, matching the upstream table
// rounding so pooled and raw entries stay comparable.
func roundDraftDepth4(v float64) float64 {
	return math.Round(v*1e4) / 1e4
}

// DraftDepthCosts prices a speculative verify window on this device:
// Verify[r] is the modeled ms for a window of r rows (index 0 unused), Level
// is the ms of one chain level, and Rows is the widest priced window.
type DraftDepthCosts struct {
	Verify [DraftDepthMaxRows + 1]float64
	Rows   int
	Level  float64
}

// NewDraftDepthCosts pools the measured per-row marginal ms so a pooled row
// never adds more ms than the row before it inside one kernel regime of
// `regime` widths. verify[0] is unused and verify[1] is the one-row window;
// every wider row joins the pooling. Pools merge adjacent marginals whose
// average rises with rows, preserving each pool's total ms so cumulative
// windows at pool boundaries match the measured table; a regime boundary row
// starts a fresh pool and can never merge back across it.
func NewDraftDepthCosts(verify []float64, level float64, regime int) (DraftDepthCosts, error) {
	rows := len(verify) - 1
	if rows < 2 || rows > DraftDepthMaxRows {
		return DraftDepthCosts{}, ErrDraftDepthVerifyTable
	}
	if regime < 1 {
		return DraftDepthCosts{}, ErrDraftDepthRegimeInvalid
	}
	if level < 0 {
		level = 0
	}
	sums := make([]float64, 0, rows-1)
	counts := make([]int, 0, rows-1)
	first := 0 // index of the first pool a row may join; a regime's first row keeps its own ms
	for r := 2; r <= rows; r++ {
		if (r-1)%regime == 0 {
			first = len(sums) + 1
		}
		sums = append(sums, verify[r]-verify[r-1])
		counts = append(counts, 1)
		for len(sums) > first+1 && sums[len(sums)-2]*float64(counts[len(sums)-1]) < sums[len(sums)-1]*float64(counts[len(sums)-2]) {
			i := len(sums) - 2
			sums[i] += sums[i+1]
			counts[i] += counts[i+1]
			sums = sums[:i+1]
			counts = counts[:i+1]
		}
	}
	c := DraftDepthCosts{Rows: rows, Level: roundDraftDepth4(level)}
	c.Verify[1] = verify[1]
	at := 2
	for i := range sums {
		base := c.Verify[at-1]
		for j := 0; j < counts[i]; j++ {
			c.Verify[at] = base + sums[i]/float64(counts[i])*float64(j+1)
			at++
		}
	}
	for i := 1; i < at; i++ {
		c.Verify[i] = roundDraftDepth4(c.Verify[i])
	}
	return c, nil
}

// Row is the ms the row carrying draft k adds to its window, floored at zero.
// It returns 0 for an invalid table shape or k outside [1, Rows-1], which
// callers treat as unpriceable.
func (c DraftDepthCosts) Row(k int) float64 {
	if c.Rows < 2 || c.Rows > DraftDepthMaxRows || k < 1 || k >= c.Rows {
		return 0
	}
	d := c.Verify[k+1] - c.Verify[k]
	if d < 0 {
		return 0
	}
	return d
}

// DraftDepthRule decides how deep a draft chain runs: the first draft always
// verifies, then draft k only while run^power pays for its row at the recent
// tokens-per-ms rate earned by the cost model. A confidence floor, when set,
// overrides the cost test. Done follows each round from its modeled cost,
// never a wall clock, so every rank of a split model chooses alike.
//
// The rule is not safe for concurrent use; the owning governor serializes
// calls under its lock.
type DraftDepthRule struct {
	costs  DraftDepthCosts
	most   int
	floor  *float64
	power  float64
	weight float64
	tokens float64
	ms     float64
}

// NewDraftDepthRule binds a cost table to a deepest-draft bound most, an
// optional confidence floor, and a confidence power. The initial token EMA is
// 2.0 and the initial ms EMA is the two-row verify plus one level.
func NewDraftDepthRule(costs DraftDepthCosts, most int, floor *float64, power float64) (*DraftDepthRule, error) {
	if costs.Rows < 2 || costs.Rows > DraftDepthMaxRows {
		return nil, ErrDraftDepthVerifyTable
	}
	if most < 1 || most > costs.Rows-1 {
		return nil, ErrDraftDepthMostOutOfRange
	}
	if !(power > 0) {
		return nil, ErrDraftDepthPowerInvalid
	}
	return &DraftDepthRule{
		costs:  costs,
		most:   most,
		floor:  floor,
		power:  power,
		weight: 0.125,
		tokens: 2.0,
		ms:     costs.Verify[2] + costs.Level,
	}, nil
}

// Rate is the tokens a ms the recent rounds earned, by the cost model.
func (r *DraftDepthRule) Rate() float64 {
	return r.tokens / r.ms
}

// Tokens reports the current tokens-per-round EMA.
func (r *DraftDepthRule) Tokens() float64 {
	return r.tokens
}

// MS reports the current modeled round-cost EMA in ms.
func (r *DraftDepthRule) MS() float64 {
	return r.ms
}

// Keep reports whether draft k, whose running confidence is run, pays for its
// row. Draft 1 always verifies: every drafted window verifies its first
// draft. Out-of-table drafts refuse fail-closed.
func (r *DraftDepthRule) Keep(k int, run float64) bool {
	if k < 1 || k >= r.costs.Rows {
		return false
	}
	if k == 1 {
		return true
	}
	if r.floor != nil {
		return run >= *r.floor
	}
	return math.Pow(run, r.power) >= r.Rate()*r.costs.Row(k)
}

// More reports whether drafting level k+1 is worth it: a certain draft there
// must pay for its row and its level. Nothing is drafted past Most.
func (r *DraftDepthRule) More(k int, run float64) bool {
	if k >= r.most {
		return false
	}
	if r.floor != nil {
		return run >= *r.floor
	}
	return math.Pow(run, r.power) >= r.Rate()*(r.costs.Row(k+1)+r.costs.Level)
}

// Done folds one finished round into the token and ms EMAs from the cost
// model: tokens earned against the modeled cost of a `rows`-wide verify plus
// `levels` chain levels. Rounds whose window the table cannot price leave
// both EMAs untouched.
func (r *DraftDepthRule) Done(tokens, rows, levels int) {
	if rows < 1 || rows > r.costs.Rows {
		return
	}
	if levels < 0 {
		levels = 0
	}
	r.tokens += r.weight * (float64(tokens) - r.tokens)
	r.ms += r.weight * (r.costs.Verify[rows] + float64(levels)*r.costs.Level - r.ms)
}
