package quality

import "fmt"

// ctxCliffOracle is the context-length / position-scaling rubric oracle (#4546):
// generation quality must not fall off a cliff as the decode runs deeper into the
// context window (e.g. RoPE positions beyond the trained length). The failure
// mode it gates is well known and fluent-looking in aggregate: the engine decodes
// normally for the trained range, then past some position collapses into loops —
// a whole-text metric averages the collapse away, while a per-position metric
// shows a sharp cliff exactly where position scaling broke.
//
// The quality metric is a deterministic per-position repetition rate: at each
// engine position, the fraction of duplicated tokens inside the trailing window
// of ctxCliffWindow tokens (1 - distinct/window). The bound is derived from the
// REFERENCE trace — its worst windowed repetition plus ctxCliffMargin — so a
// reference that legitimately repeats (refrains, table rows) raises the bound
// instead of false-failing the engine. A reference with no tokens falls back to
// the absolute ctxCliffDefaultBound. The engine trace may be LONGER than the
// reference (that is the point: positions beyond the reference length model
// extrapolation past the trained window) and every engine position is judged
// against the same reference-derived bound.
//
// Score = positions within the bound / positions evaluated; Pass iff Score >=
// Rubric.MinScore (default 1: quality must stay bounded at EVERY position). On
// failure the verdict localizes the cliff: FirstDivergence.Index is the first
// position whose windowed repetition crossed the bound, and Detail reports the
// metric value against the bound there — "position 43 cliffed at 0.19 > 0.15",
// not "the long decode looked repetitive".
type ctxCliffOracle struct{}

func (ctxCliffOracle) Name() string { return "context-cliff" }
func (ctxCliffOracle) Kind() string { return "rubric" }

func init() { Register(ctxCliffOracle{}) }

const (
	// ctxCliffWindow is the trailing window (in tokens) the per-position
	// repetition rate is computed over.
	ctxCliffWindow = 16
	// ctxCliffMargin is the slack added to the reference's worst windowed
	// repetition to form the engine bound.
	ctxCliffMargin = 0.15
	// ctxCliffDefaultBound is the absolute bound used when the reference trace
	// carries no tokens to derive one from.
	ctxCliffDefaultBound = 0.5
)

func (ctxCliffOracle) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "context-cliff", Kind: "rubric", Pass: true, Score: 1}
	points := ctxCliffRates(eng.Tokens)
	if len(points) == 0 {
		v.Detail = "engine emitted no tokens; no positions to judge"
		return v
	}
	bound := ctxCliffBound(ref.Tokens)
	bounded := 0
	cliffAt := -1
	cliffRate := 0.0
	for _, p := range points {
		if p.rate <= bound {
			bounded++
		} else if cliffAt < 0 {
			cliffAt = p.pos
			cliffRate = p.rate
		}
	}
	v.Score = float64(bounded) / float64(len(points))
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: quality must stay bounded at every position
	}
	if v.Score < min {
		v.Pass = false
		v.FirstDivergence = &Divergence{
			Index:     cliffAt,
			Reference: fmt.Sprintf("windowed repetition <= %.4f", bound),
			Engine:    fmt.Sprintf("windowed repetition %.4f", cliffRate),
		}
		v.Detail = fmt.Sprintf(
			"quality cliff at position %d: windowed repetition %.4f > bound %.4f (reference-derived); %d/%d positions bounded",
			cliffAt, cliffRate, bound, bounded, len(points))
		return v
	}
	if cliffAt >= 0 {
		v.Detail = fmt.Sprintf("bounded score %.2f >= %.2f (tolerated crossing at position %d: %.4f > %.4f)",
			v.Score, min, cliffAt, cliffRate, bound)
		return v
	}
	v.Detail = fmt.Sprintf("repetition stayed within bound %.4f across all %d positions (window %d)",
		bound, len(points), ctxCliffWindow)
	return v
}

// ctxCliffPoint is the quality metric sampled at one position: the repetition
// rate of the ctxCliffWindow tokens ending at pos.
type ctxCliffPoint struct {
	pos  int
	rate float64
}

// ctxCliffRates computes the windowed repetition rate at each position of toks.
// A trace shorter than the window is judged as one window over its whole length
// (so short faithful traces are not skipped, and a short trace that is already a
// loop still registers). An empty trace yields no points.
func ctxCliffRates(toks []string) []ctxCliffPoint {
	n := len(toks)
	if n == 0 {
		return nil
	}
	if n < ctxCliffWindow {
		return []ctxCliffPoint{{pos: n - 1, rate: ctxCliffRepetition(toks)}}
	}
	out := make([]ctxCliffPoint, 0, n-ctxCliffWindow+1)
	for i := ctxCliffWindow - 1; i < n; i++ {
		out = append(out, ctxCliffPoint{pos: i, rate: ctxCliffRepetition(toks[i-ctxCliffWindow+1 : i+1])})
	}
	return out
}

// ctxCliffRepetition is 1 - distinct/len over one window: 0 for all-unique
// tokens, approaching 1 as the window collapses into a single repeated token.
func ctxCliffRepetition(window []string) float64 {
	if len(window) == 0 {
		return 0
	}
	distinct := make(map[string]struct{}, len(window))
	for _, t := range window {
		distinct[t] = struct{}{}
	}
	return 1 - float64(len(distinct))/float64(len(window))
}

// ctxCliffBound derives the engine bound from the reference trace: its worst
// windowed repetition plus ctxCliffMargin, capped at 1. A reference with no
// tokens yields the absolute ctxCliffDefaultBound.
func ctxCliffBound(refToks []string) float64 {
	points := ctxCliffRates(refToks)
	if len(points) == 0 {
		return ctxCliffDefaultBound
	}
	peak := 0.0
	for _, p := range points {
		if p.rate > peak {
			peak = p.rate
		}
	}
	b := peak + ctxCliffMargin
	if b > 1 {
		b = 1
	}
	return b
}
