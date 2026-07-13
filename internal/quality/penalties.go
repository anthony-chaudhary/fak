package quality

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// penalties.go — #4527: validate repetition, presence, and frequency penalty
// MATH and ORDERING against a recomputed reference.
//
// Given one raw logits row and a generation history (how many times each vocab
// token was already emitted), a faithful engine must apply the three sampling
// penalties with the standard rules AND in a fixed order. The canonical
// pipeline this oracle pins, applied per vocabulary token with count = the
// token's occurrences in the history (a token never seen is left untouched):
//
//  1. REPETITION penalty r — multiplicative and sign-aware (the CTRL-paper /
//     HF rule): for any token seen at least once, if logit > 0 the logit
//     becomes logit / r, else logit * r. Applied FIRST, on the RAW logit.
//  2. PRESENCE penalty p — flat subtraction: logit -= p for any token that
//     appeared at least once, regardless of count.
//  3. FREQUENCY penalty f — count-scaled subtraction: logit -= f * count.
//
// Steps 2 and 3 commute with each other (both are subtractions), but NEITHER
// commutes with step 1: the sign-aware scaling must see the raw logit, so an
// engine that subtracts frequency before scaling by r produces
// (x - f*c)/r != x/r - f*c whenever both penalties bind and r != 1. That
// ordering bug — and the classic wrong-sign (scaling negative logits as if
// positive) and double-apply variants — all surface here as a numeric
// divergence at a specific token, which the oracle names.
//
// Case encoding: the raw logits row and the vocabulary ride in the standard
// slots (Reference.Logits[0] aligned with Reference.Tokens), and the penalty
// parameters + history counts are serialized into the case Prompt by
// PenaltyCase via penEncodeSpec — the case stays pure replayable data, no
// change to the QualityCase struct. The engine emits its penalized logits as
// Trace.Logits[0]; Judge recomputes the reference pipeline and compares
// element-wise within penTolerance.

// penaltySpecPrefix versions the prompt-embedded penalty spec. The spec line is
//
//	penalty-spec/1 rep=<r> presence=<p> freq=<f> | tok=<count> tok=<count> ...
//
// with history entries sorted by token so encoding is deterministic. Tokens
// containing spaces, '=', or '|' are refused by the parser's strict grammar.
const penaltySpecPrefix = "penalty-spec/1"

// penTolerance is the element-wise comparison tolerance. The reference and a
// faithful engine run the same float64 pipeline, so their outputs agree
// exactly; the tolerance only absorbs benign association-order noise in a real
// engine, while every injected defect here disagrees by orders of magnitude
// more.
const penTolerance = 1e-9

// penSpec is the parsed penalty configuration for one case: the three penalty
// strengths and the per-token history counts.
type penSpec struct {
	Repetition float64
	Presence   float64
	Frequency  float64
	Counts     map[string]int
}

// PenaltyOrdering is the penalty math + ordering differential oracle (#4527).
type PenaltyOrdering struct{}

func (PenaltyOrdering) Name() string { return "penalty-ordering" }
func (PenaltyOrdering) Kind() string { return "differential" }

func init() { Register(PenaltyOrdering{}) }

func (PenaltyOrdering) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "penalty-ordering", Kind: "differential"}
	spec, err := penParseSpec(c.Prompt)
	if err != nil {
		v.Detail = "penalty spec malformed (a case that cannot be recomputed is not green): " + err.Error()
		return v
	}
	vocab := ref.Tokens
	if len(vocab) == 0 || len(ref.Logits) != 1 || len(ref.Logits[0]) != len(vocab) {
		v.Detail = fmt.Sprintf("reference malformed: want raw logits as one row of %d values aligned with Reference.Tokens", len(vocab))
		return v
	}
	if len(eng.Logits) == 0 {
		v.Detail = "engine emitted no logits row; penalized logits cannot be judged"
		return v
	}
	want := penApply(ref.Logits[0], vocab, spec)
	got := eng.Logits[0]

	n := min(len(want), len(got))
	for i := 0; i < n; i++ {
		if penEqual(want[i], got[i]) {
			continue
		}
		v.FirstDivergence = &Divergence{
			Index:     i,
			Reference: penFmtLogit(vocab[i], want[i]),
			Engine:    penFmtLogit(vocab[i], got[i]),
		}
		v.Detail = fmt.Sprintf(
			"token %d %q: penalized logit diverged: reference %.9g, engine %.9g (raw %.9g, history count %d; order: repetition r=%g first, then presence %g + frequency %g subtractions)",
			i, vocab[i], want[i], got[i], ref.Logits[0][i], spec.Counts[vocab[i]],
			spec.Repetition, spec.Presence, spec.Frequency)
		return v
	}
	if len(want) != len(got) {
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(vocab, n), Engine: "<missing>"}
		if len(got) > len(want) {
			v.FirstDivergence.Reference = "<end>"
			v.FirstDivergence.Engine = fmt.Sprintf("%.9g", got[n])
		}
		v.Detail = fmt.Sprintf("logit row length diverged: reference has %d, engine has %d", len(want), len(got))
		return v
	}
	v.Pass = true
	v.Detail = fmt.Sprintf("%d penalized logits matched the reference within %g (repetition first, then presence+frequency)",
		len(want), penTolerance)
	return v
}

// penApply recomputes the reference penalized logits: the documented pipeline
// (sign-aware repetition scaling on the raw logit FIRST, then the presence and
// frequency subtractions) applied per token. Tokens absent from the history are
// returned unchanged.
func penApply(raw []float64, vocab []string, s penSpec) []float64 {
	out := make([]float64, len(raw))
	for i, x := range raw {
		count := 0
		if i < len(vocab) {
			count = s.Counts[vocab[i]]
		}
		if count <= 0 {
			out[i] = x
			continue
		}
		if x > 0 {
			x /= s.Repetition
		} else {
			x *= s.Repetition
		}
		x -= s.Presence
		x -= s.Frequency * float64(count)
		out[i] = x
	}
	return out
}

// penEqual compares two penalized logits within penTolerance; NaNs never match
// (a NaN logit is a defect, not a wildcard).
func penEqual(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	return math.Abs(a-b) <= penTolerance
}

// penFmtLogit renders "token=value" for Divergence fields so the localized
// evidence names the token whose penalized logit is wrong.
func penFmtLogit(tok string, v float64) string {
	return tok + "=" + strconv.FormatFloat(v, 'g', -1, 64)
}

// penEncodeSpec serializes a penSpec into the versioned prompt line. History
// tokens are emitted in sorted order so the same spec always encodes to the
// same string (the replay contract).
func penEncodeSpec(s penSpec) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s rep=%s presence=%s freq=%s |",
		penaltySpecPrefix,
		strconv.FormatFloat(s.Repetition, 'g', -1, 64),
		strconv.FormatFloat(s.Presence, 'g', -1, 64),
		strconv.FormatFloat(s.Frequency, 'g', -1, 64))
	keys := make([]string, 0, len(s.Counts))
	for k, c := range s.Counts {
		if c > 0 {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, " %s=%d", k, s.Counts[k])
	}
	return b.String()
}

// penParseSpec parses the versioned spec line back out of a case prompt. It is
// strict: a missing prefix, an unparseable field, a non-positive repetition
// factor, or a non-positive count is an error, so a malformed case fails closed
// in Judge instead of passing vacuously.
func penParseSpec(prompt string) (penSpec, error) {
	rest, ok := strings.CutPrefix(prompt, penaltySpecPrefix+" ")
	if !ok {
		return penSpec{}, fmt.Errorf("prompt does not begin with %q", penaltySpecPrefix)
	}
	head, tail, ok := strings.Cut(rest, "|")
	if !ok {
		return penSpec{}, fmt.Errorf("spec has no '|' history separator")
	}
	s := penSpec{Counts: map[string]int{}}
	fields := strings.Fields(head)
	if len(fields) != 3 {
		return penSpec{}, fmt.Errorf("want 3 penalty fields (rep, presence, freq), got %d", len(fields))
	}
	for i, want := range []string{"rep", "presence", "freq"} {
		key, val, ok := strings.Cut(fields[i], "=")
		if !ok || key != want {
			return penSpec{}, fmt.Errorf("field %d: want %s=<float>, got %q", i, want, fields[i])
		}
		f, err := strconv.ParseFloat(val, 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return penSpec{}, fmt.Errorf("field %q: not a finite float", fields[i])
		}
		switch want {
		case "rep":
			s.Repetition = f
		case "presence":
			s.Presence = f
		case "freq":
			s.Frequency = f
		}
	}
	if s.Repetition <= 0 {
		return penSpec{}, fmt.Errorf("repetition factor must be positive, got %v", s.Repetition)
	}
	for _, kv := range strings.Fields(tail) {
		tok, val, ok := strings.Cut(kv, "=")
		if !ok || tok == "" {
			return penSpec{}, fmt.Errorf("history entry %q: want token=<count>", kv)
		}
		c, err := strconv.Atoi(val)
		if err != nil || c <= 0 {
			return penSpec{}, fmt.Errorf("history entry %q: count must be a positive integer", kv)
		}
		if _, dup := s.Counts[tok]; dup {
			return penSpec{}, fmt.Errorf("history entry %q: duplicate token", kv)
		}
		s.Counts[tok] = c
	}
	return s, nil
}

// PenaltyCase builds a penalty-ordering quality case: the raw logits row and
// vocabulary ride in the standard Reference slots, and the penalty strengths +
// history counts are serialized into the Prompt so the case remains pure,
// hermetic, replayable data. counts maps a vocab token to how many times it
// already appeared in the generation history; tokens absent (or with count 0)
// are unpenalized.
func PenaltyCase(id string, vocab []string, raw []float64, counts map[string]int, rep, presence, freq float64) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  penEncodeSpec(penSpec{Repetition: rep, Presence: presence, Frequency: freq, Counts: counts}),
		Params:  SamplingParams{Temperature: 0, MaxTokens: len(vocab)},
		Reference: Trace{
			Tokens: append([]string(nil), vocab...),
			Logits: [][]float64{append([]float64(nil), raw...)},
		},
		Oracles: []string{"penalty-ordering"},
	}
}
