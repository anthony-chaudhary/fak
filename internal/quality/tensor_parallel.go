package quality

import (
	"fmt"
	"math"
	"strings"
)

// tensor_parallel.go is the tensor-parallel parity child of the quality spine
// (#4542): decoding under tensor-parallel degree N must produce the same output
// as the single-device TP=1 decode. This file models tensor parallelism as a
// partitioned reduction — every logit is a sum of per-hidden-unit
// contributions, TP=N splits the hidden units across N contiguous shards, each
// shard folds its slice in ascending order, and the all-reduce folds the shard
// partials in stable ascending shard order. Floating-point addition is not
// associative, so a faithful sharded decode reproduces TP=1 logits only within
// fp tolerance — but it MUST reproduce the emitted tokens exactly. The
// registered "tp-parity" differential oracle asserts exactly that contract:
// token-exact parity plus within-tolerance logit parity, localizing the first
// step at which a reduction-order/associativity bug changed a token.

// tpVocab is the small fixed vocabulary the deterministic decode draws from.
// Eight entries keep the token space tiny while making an accidental argmax
// collision between the stable and the reordered reduction impossible by
// construction (the bonus margins below dominate the noise).
var tpVocab = []string{"alloy", "bronze", "cinder", "dovetail", "eddy", "fern", "gale", "hollow"}

// tpHiddenDim is the number of per-token hidden contributions summed into each
// logit — the dimension tensor-parallel shards partition.
const tpHiddenDim = 8

// tpHuge is the magnitude of the engineered cancellation pair at hidden units
// 0 and 1 (+tpHuge then -tpHuge). Folded in order the pair cancels exactly to
// zero before any small contribution arrives, so the stable reduction is
// full-precision. Folded out of order the huge term absorbs every small
// contribution summed next to it (float64 spacing at 2^60 is 256), which is
// what makes a reduction-order bug MATERIALLY wrong — a changed token — rather
// than ulp-level wrong.
const tpHuge = float64(1 << 60)

// tpWinnerBonus and tpRunnerBonus pin each step's argmax by construction: the
// step's winner token carries a +32 bonus at hidden unit 2 (first shard for
// any degree <= 4), the runner-up a +8 bonus at the LAST hidden unit (last
// shard). Per-unit noise is < 1, so under the stable reduction the winner
// dominates (>= 32 vs < 14) and under the reordered shard-0 reduction — which
// erases shard 0, bonus included — the runner-up dominates (>= 8 vs < 4). Both
// margins are astronomically above reassociation ulps.
const (
	tpWinnerBonus = 32.0
	tpRunnerBonus = 8.0
)

// tpBugStep is the decode step the injected reduction-order defect fires at:
// mid-sequence, so the passing prefix proves the localization is doing work
// (the failure pins to the step, not to "the whole decode looked wrong").
const tpBugStep = 2

// tpLogitTolerance is the fp tolerance the parity oracle allows between the
// TP=1 and the sharded logits: comfortably above the ~1e-14 reassociation
// error of a correct partitioned reduction, far below any material drift.
const tpLogitTolerance = 1e-9

// tpDefectReorder names the injected associativity defect: shard 0's
// within-shard summation order is reversed at tpBugStep, separating the
// cancellation pair so the huge term absorbs the shard's small contributions.
const tpDefectReorder = "reduce-reorder"

// tpMix maps (step, v, h) to one deterministic splitmix64-style draw. A pure
// function of its inputs — no carried state, no ambient entropy — so every
// contribution replays identically on both decode paths.
func tpMix(step, v, h uint64) uint64 {
	z := step*0x9e3779b97f4a7c15 + v*0xbf58476d1ce4e5b9 + h*0x94d049bb133111eb + 0x2545f4914f6cdd1d
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// tpNoise is the small pseudorandom contribution in [0, 1) of hidden unit h to
// token v's logit at step.
func tpNoise(step, v, h int) float64 {
	return float64(tpMix(uint64(step), uint64(v), uint64(h))>>11) / float64(1<<53)
}

// tpWinner is the token index the stable reduction decodes at step; tpRunner
// is the distinct token the reordered reduction decodes instead.
func tpWinner(step int) int { return int(tpMix(uint64(step), 0x5157, 0) % uint64(len(tpVocab))) }
func tpRunner(step int) int { return (tpWinner(step) + 1) % len(tpVocab) }

// tpContrib is the contribution of hidden unit h to token v's logit at step:
// the leading cancellation pair, the winner/runner bonuses, noise elsewhere.
func tpContrib(step, v, h int) float64 {
	switch {
	case h == 0:
		return tpHuge
	case h == 1:
		return -tpHuge
	case h == 2 && v == tpWinner(step):
		return tpWinnerBonus
	case h == tpHiddenDim-1 && v == tpRunner(step):
		return tpRunnerBonus
	default:
		return tpNoise(step, v, h)
	}
}

// tpShards partitions the hidden dimension into degree contiguous [start, end)
// slices (the last may be shorter). degree <= 1 yields the single TP=1 shard —
// the degenerate partition where sharded reduction IS the sequential sum.
func tpShards(degree int) [][2]int {
	if degree < 1 {
		degree = 1
	}
	size := (tpHiddenDim + degree - 1) / degree
	var out [][2]int
	for start := 0; start < tpHiddenDim; start += size {
		end := start + size
		if end > tpHiddenDim {
			end = tpHiddenDim
		}
		out = append(out, [2]int{start, end})
	}
	return out
}

// tpLogit computes token v's logit at step under the given shard partition:
// each shard folds its hidden slice in ascending order into a partial, then
// the all-reduce folds shard partials in stable ascending shard order. reorder
// reverses shard 0's within-shard order — the injected associativity bug: the
// -tpHuge term then lands on the shard's small running sum and absorbs it
// (anything below half the fp spacing at 2^60 rounds away), so the shard's
// entire small mass, winner bonus included, vanishes from the logit.
func tpLogit(step, v int, shards [][2]int, reorder bool) float64 {
	total := 0.0
	for si, sh := range shards {
		partial := 0.0
		if reorder && si == 0 {
			for h := sh[1] - 1; h >= sh[0]; h-- {
				partial += tpContrib(step, v, h)
			}
		} else {
			for h := sh[0]; h < sh[1]; h++ {
				partial += tpContrib(step, v, h)
			}
		}
		total += partial
	}
	return total
}

// tpArgmax returns the index of the row's maximum (first index on a tie, for
// determinism; the engineered margins make ties impossible in practice).
func tpArgmax(row []float64) int {
	best := 0
	for i := 1; i < len(row); i++ {
		if row[i] > row[best] {
			best = i
		}
	}
	return best
}

// tpDecode runs the deterministic greedy decode for steps tokens under the
// given tensor-parallel degree, capturing per-step logits. defect "" is the
// faithful stable reduction; tpDefectReorder fires the reordered shard-0
// reduction at tpBugStep only.
func tpDecode(degree, steps int, defect string) Trace {
	shards := tpShards(degree)
	toks := make([]string, 0, steps)
	logits := make([][]float64, 0, steps)
	for i := 0; i < steps; i++ {
		reorder := defect == tpDefectReorder && i == tpBugStep
		row := make([]float64, len(tpVocab))
		for v := range tpVocab {
			row[v] = tpLogit(i, v, shards, reorder)
		}
		logits = append(logits, row)
		toks = append(toks, tpVocab[tpArgmax(row)])
	}
	return Trace{Tokens: toks, Logits: logits, Text: strings.Join(toks, " ")}
}

// TpShardedRunner decodes under tensor-parallel degree Degree via the
// partitioned reduction. The zero defect is a faithful engine; the defect
// field (set via TpEngine) injects the reduction-order bug. It is the
// ScriptedRunner-style adapter for the TP seam: a real sharded engine wires in
// behind the same Runner interface and is judged the same way.
type TpShardedRunner struct {
	Label  string
	Degree int
	defect string
}

func (r TpShardedRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "tp-sharded"
}

func (r TpShardedRunner) Run(c QualityCase) (Trace, error) {
	t := tpDecode(r.Degree, c.Params.MaxTokens, r.defect)
	t.Runner = r.Name()
	return t, nil
}

// TpEngine returns a tensor-parallel engine runner at the given degree with an
// optional injected defect: "" reduces in stable order (token-exact parity
// with TP=1, logits within fp tolerance at ANY degree); "reduce-reorder"
// reverses shard 0's fp summation order at step tpBugStep so the decoded token
// there flips from the winner to the runner-up. This is the deterministic
// mutant source the tests use to prove the parity gate trips.
func TpEngine(degree int, defect string) TpShardedRunner {
	switch defect {
	case tpDefectReorder:
		return TpShardedRunner{Label: "engine-tp-reorder", Degree: degree, defect: defect}
	default:
		return TpShardedRunner{Label: "engine-tp-sharded", Degree: degree}
	}
}

// TpParityCase builds the deterministic tensor-parallel parity case: a
// temperature-zero decode budget and a reference trace produced by the TP=1
// (single shard) path itself. The TP degree deliberately does NOT appear in
// the case — parity must hold for EVERY degree, so the degree is engine
// configuration, not case data.
func TpParityCase() QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 6}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "tp-parity-demo",
		Version:   1,
		Prompt:    "Decode the rollup under tensor-parallel sharding.",
		Params:    params,
		Reference: tpDecode(1, params.MaxTokens, ""),
		Oracles:   []string{"tp-parity"},
	}
}

// TpParity is the differential oracle for #4542: the sharded engine's token
// stream must equal the TP=1 reference stream exactly, and where both traces
// carry per-step logits those must agree within tpLogitTolerance. A correct
// partitioned reduction (stable shard order) reproduces TP=1 tokens exactly
// and logits to within reassociation ulps; a reduction-order/associativity bug
// that materially changes a logit flips a token, and the verdict pins the
// FIRST step it happened at — "TP=8 broke the report" localizes to "token 2
// decoded 'fern' where TP=1 decoded 'cinder'".
type TpParity struct{}

func (TpParity) Name() string { return "tp-parity" }
func (TpParity) Kind() string { return "differential" }

func init() { Register(TpParity{}) }

func (TpParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "tp-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("tensor-parallel decode diverged from TP=1 at token %d: reference (TP=1) %q, engine (sharded) %q",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
		if i < len(ref.Logits) && i < len(eng.Logits) {
			if j, delta, drifted := tpLogitDrift(ref.Logits[i], eng.Logits[i]); drifted {
				v.Pass = false
				v.FirstDivergence = &Divergence{
					Index:     i,
					Reference: fmt.Sprintf("%s (logit[%d]=%s)", ref.Tokens[i], j, tpLogitStr(ref.Logits[i], j)),
					Engine:    fmt.Sprintf("%s (logit[%d]=%s)", eng.Tokens[i], j, tpLogitStr(eng.Logits[i], j)),
				}
				v.Detail = fmt.Sprintf("logit drift beyond tolerance at token %d, vocab %d: |delta| %.3g > %.3g — reduction is not within fp tolerance of TP=1",
					i, j, delta, tpLogitTolerance)
				return v
			}
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("tensor-parallel decode length diverged at %d: TP=1 reference has %d tokens, sharded engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("tensor-parallel decode matched TP=1: %d tokens identical, logits within %.3g",
		len(ref.Tokens), tpLogitTolerance)
	return v
}

// tpLogitDrift returns the first vocab index at which two logit rows deviate
// beyond tpLogitTolerance (or the shorter length when the rows differ in
// length), the absolute deviation there, and whether such a drift exists.
func tpLogitDrift(ref, eng []float64) (int, float64, bool) {
	n := len(ref)
	if len(eng) < n {
		n = len(eng)
	}
	for j := 0; j < n; j++ {
		if d := math.Abs(ref[j] - eng[j]); d > tpLogitTolerance {
			return j, d, true
		}
	}
	if len(ref) != len(eng) {
		return n, math.Inf(1), true
	}
	return 0, 0, false
}

// tpLogitStr renders row[j] for a divergence report, tolerating a row too
// short to carry index j (the row-length-mismatch drift).
func tpLogitStr(row []float64, j int) string {
	if j < len(row) {
		return fmt.Sprintf("%.12g", row[j])
	}
	return "<missing>"
}
