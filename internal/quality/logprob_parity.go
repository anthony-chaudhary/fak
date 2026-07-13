package quality

import (
	"fmt"
	"math"
	"strings"
)

// logprob_parity.go is the logprob-parity child of the quality spine (#4524):
// the per-token logprobs an engine returns — for the PROMPT tokens it echoes
// and for the tokens it GENERATES — must match the reference within tolerance
// AND be correctly aligned, i.e. token i's logprob row is the row computed FOR
// token i, not for a neighbor. Logprobs are the numeric surface downstream
// consumers (rerankers, calibration, best-of-n selection) actually read, and
// the two classic ways an engine corrupts them are silent: an off-by-one
// alignment bug (every row shifted by one position) and a normalization bug
// (raw pre-softmax logits reported where log-probabilities were promised).
// Both leave the token text byte-identical, so no token-differential oracle can
// see them. This file models per-token logprob rows in Trace.Logits via a tiny
// deterministic scorer, provides a faithful engine and the two defect mutants
// behind the shared Runner seam, and registers the "logprob-parity"
// differential oracle that pins the FIRST misaligned or out-of-tolerance row
// with a classified Detail.

// lpVocab is the small fixed vocabulary generated tokens draw from. The tokens
// are only labels here — the differential surface of this child is the logprob
// rows — but distinct names keep the localized divergence human-readable.
var lpVocab = []string{"audit", "beacon", "candor", "delta", "ember", "fulcrum", "garnet", "hollow"}

const (
	// lpOracleName is the registered name cases reference.
	lpOracleName = "logprob-parity"
	// lpTolerance is the absolute per-value tolerance for logprob parity.
	// Reference and faithful engine share one deterministic scorer, so the
	// tolerance only has to absorb float formatting/transport noise — while both
	// defect classes produce errors many orders of magnitude larger.
	lpTolerance = 1e-6
	// lpCandidates is the number of candidate logprobs each per-token row carries
	// (an OpenAI-style top-k logprob row).
	lpCandidates = 4
	// lpDefectShift names the off-by-one alignment defect: every logprob row is
	// shifted one position, so token i carries the row computed for token i-1.
	lpDefectShift = "logprob-shift"
	// lpDefectRaw names the normalization defect: raw pre-softmax logits are
	// reported where log-softmax logprobs were promised.
	lpDefectRaw = "raw-logits"
)

// lpMix maps (step, slot) to one pseudo-random draw via a splitmix64-style
// finalizer. A pure function of its inputs — no carried state, no ambient
// entropy — so every trace this file builds is hermetic and replayable.
func lpMix(step, slot int) uint64 {
	z := (uint64(step)+1)*0x9e3779b97f4a7c15 + (uint64(slot)+1)*0xbf58476d1ce4e5b9
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// lpRawStep returns the lpCandidates raw (pre-softmax) candidate logits for one
// token position, values in [0, 4). Raw logits are deliberately non-negative:
// a raw logit reported as a logprob is then detectable both by sign (a
// log-probability can never be positive) and by the constant logsumexp offset
// it carries relative to the normalized reference row.
func lpRawStep(step int) []float64 {
	raw := make([]float64, lpCandidates)
	for j := range raw {
		raw[j] = float64(lpMix(step, j)%4000) / 1000
	}
	return raw
}

// lpLogSoftmax normalizes one raw-logit row into logprobs:
// out[j] = raw[j] - logsumexp(raw). This is the normalization step the
// raw-logits defect skips.
func lpLogSoftmax(raw []float64) []float64 {
	max := raw[0]
	for _, x := range raw {
		if x > max {
			max = x
		}
	}
	var sum float64
	for _, x := range raw {
		sum += math.Exp(x - max)
	}
	lse := max + math.Log(sum)
	out := make([]float64, len(raw))
	for j, x := range raw {
		out[j] = x - lse
	}
	return out
}

// lpFaithfulTrace builds the correct full trace for a prompt: the echoed prompt
// tokens followed by maxTokens generated tokens, with one normalized logprob
// row PER token — prompt and generation alike — so len(Logits) == len(Tokens)
// and row i scores exactly token i.
func lpFaithfulTrace(prompt string, maxTokens int) Trace {
	toks := append([]string(nil), strings.Fields(prompt)...)
	for i := 0; i < maxTokens; i++ {
		toks = append(toks, lpVocab[lpMix(len(toks), lpCandidates)%uint64(len(lpVocab))])
	}
	logits := make([][]float64, len(toks))
	for i := range toks {
		logits[i] = lpLogSoftmax(lpRawStep(i))
	}
	return Trace{Tokens: toks, Logits: logits, Text: strings.Join(toks, " ")}
}

// lpSegment labels a token index as prompt or generation for localization
// detail, given the prompt token count.
func lpSegment(i, promptLen int) string {
	if i < promptLen {
		return "prompt"
	}
	return "generation"
}

// lpApproxEqual reports whether two logprob rows agree elementwise within
// lpTolerance. It is what the off-by-one classifier uses to recognize a
// shifted row as its neighbor's row rather than as arbitrary noise.
func lpApproxEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for j := range a {
		if math.Abs(a[j]-b[j]) > lpTolerance {
			return false
		}
	}
	return true
}

// lpConstantOffset reports the constant d with eng[j] == ref[j] + d (within
// tolerance) when one exists and is materially nonzero — the signature of a
// row that skipped log-softmax, since raw = logprob + logsumexp per row.
func lpConstantOffset(eng, ref []float64) (float64, bool) {
	if len(eng) != len(ref) || len(eng) == 0 {
		return 0, false
	}
	d := eng[0] - ref[0]
	for j := range eng {
		if math.Abs(eng[j]-ref[j]-d) > lpTolerance {
			return 0, false
		}
	}
	return d, math.Abs(d) > lpTolerance
}

// lpEntry renders one (token, candidate logprob) coordinate for a Divergence
// side, degrading to the bare token when the row or slot is absent.
func lpEntry(t Trace, i, j int) string {
	tok := tokenAt(t.Tokens, i)
	if i < len(t.Logits) && j < len(t.Logits[i]) {
		return fmt.Sprintf("%s logprob[%d]=%.6f", tok, j, t.Logits[i][j])
	}
	return tok
}

// lpClassify names the defect class of the first bad row: an off-by-one
// alignment (the engine row is the reference row of the PREVIOUS token), a
// wrong normalization (the engine row is the reference row plus a constant —
// raw logits that skipped log-softmax), or a generic value mismatch.
func lpClassify(ref Trace, i int, engRow []float64) string {
	if i > 0 && i-1 < len(ref.Logits) && lpApproxEqual(engRow, ref.Logits[i-1]) {
		return fmt.Sprintf("off-by-one alignment: token %d carries the logprob row computed for token %d (%q)",
			i, i-1, tokenAt(ref.Tokens, i-1))
	}
	if i < len(ref.Logits) {
		if d, ok := lpConstantOffset(engRow, ref.Logits[i]); ok {
			return fmt.Sprintf("wrong normalization: engine row = reference row + constant %.6f — raw logits reported without log-softmax", d)
		}
	}
	return "logprob value out of tolerance"
}

// LogprobParityRunner is the engine-path adapter for #4524: it computes the
// faithful prompt+generation trace and then applies an optional injected
// logprob defect. The token text is left byte-identical in every defect mode —
// exactly the property that makes these bugs invisible to token-differential
// oracles and the reason this oracle reads Logits.
type LogprobParityRunner struct {
	Label  string
	defect string
}

func (r LogprobParityRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "logprob-engine"
}

func (r LogprobParityRunner) Run(c QualityCase) (Trace, error) {
	t := lpFaithfulTrace(c.Prompt, c.Params.MaxTokens)
	switch r.defect {
	case lpDefectShift:
		// The off-by-one bug: row i reports the logprobs computed for token i-1,
		// with the first row duplicated as the pad — so the first divergence
		// localizes at index 1, proving the intact index 0 did work.
		shifted := make([][]float64, len(t.Logits))
		if len(t.Logits) > 0 {
			shifted[0] = t.Logits[0]
			for i := 1; i < len(t.Logits); i++ {
				shifted[i] = t.Logits[i-1]
			}
		}
		t.Logits = shifted
	case lpDefectRaw:
		// The normalization bug: every row reports the raw pre-softmax logits.
		for i := range t.Logits {
			t.Logits[i] = lpRawStep(i)
		}
	}
	t.Runner = r.Name()
	return t, nil
}

// LogprobParityEngine returns an engine runner with an optional injected
// logprob defect: "" reports faithful aligned, normalized logprobs;
// "logprob-shift" shifts every row by one position (off-by-one alignment);
// "raw-logits" reports raw pre-softmax logits (wrong normalization). This is
// the deterministic mutant source the tests use to prove the gate trips.
func LogprobParityEngine(defect string) LogprobParityRunner {
	switch defect {
	case lpDefectShift:
		return LogprobParityRunner{Label: "engine-logprob-shift", defect: defect}
	case lpDefectRaw:
		return LogprobParityRunner{Label: "engine-raw-logits", defect: defect}
	default:
		return LogprobParityRunner{Label: "engine-logprob-clean"}
	}
}

// LogprobParityCase builds the deterministic logprob-parity case: a fixed
// multi-token prompt, a temperature-zero generation budget, and a reference
// trace carrying one normalized logprob row per prompt AND generated token,
// produced by the same scorer the faithful engine runs.
func LogprobParityCase() QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 5}
	prompt := "audit the weekly throughput ledger"
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "logprob-parity-demo",
		Version:   1,
		Prompt:    prompt,
		Params:    params,
		Reference: lpFaithfulTrace(prompt, params.MaxTokens),
		Oracles:   []string{lpOracleName},
	}
}

// LogprobParity is the differential oracle for #4524: every returned logprob
// row — prompt echo and generation alike — must be ALIGNED (row i scores token
// i, one row per token, same candidate width as the reference) and within
// lpTolerance of the reference values. The first offending row is reported as
// the first divergence with a classified Detail, so "the logprobs look off"
// localizes to "token 1's row is token 0's row: off-by-one" or "row 0 carries
// a +2.48 constant offset: raw logits".
type LogprobParity struct{}

func (LogprobParity) Name() string { return lpOracleName }
func (LogprobParity) Kind() string { return "differential" }

func init() { Register(LogprobParity{}) }

func (LogprobParity) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: lpOracleName, Kind: "differential", Pass: true}
	promptLen := len(strings.Fields(c.Prompt))

	// Alignment rule 0: coverage. One logprob row per emitted token — a trace
	// whose row count disagrees with its own token count cannot be aligned.
	if len(eng.Logits) != len(eng.Tokens) {
		i := len(eng.Logits)
		if len(eng.Tokens) < i {
			i = len(eng.Tokens)
		}
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: i, Reference: lpEntry(ref, i, 0), Engine: lpEntry(eng, i, 0)}
		v.Detail = fmt.Sprintf("logprob coverage broken: engine returned %d logprob rows for %d tokens; alignment breaks at index %d",
			len(eng.Logits), len(eng.Tokens), i)
		return v
	}
	if len(eng.Logits) != len(ref.Logits) {
		i := len(eng.Logits)
		if len(ref.Logits) < i {
			i = len(ref.Logits)
		}
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: i, Reference: lpEntry(ref, i, 0), Engine: lpEntry(eng, i, 0)}
		v.Detail = fmt.Sprintf("logprob row count diverged at %d: reference has %d rows, engine has %d",
			i, len(ref.Logits), len(eng.Logits))
		return v
	}

	for i := range ref.Logits {
		refRow, engRow := ref.Logits[i], eng.Logits[i]
		if len(engRow) != len(refRow) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: lpEntry(ref, i, 0), Engine: lpEntry(eng, i, 0)}
			v.Detail = fmt.Sprintf("candidate width diverged at token %d (%q, %s segment): reference row has %d entries, engine %d",
				i, tokenAt(eng.Tokens, i), lpSegment(i, promptLen), len(refRow), len(engRow))
			return v
		}
		for j := range refRow {
			e := engRow[j]
			// A log-probability can never be positive: a materially positive value
			// is a raw logit wearing a logprob's name, regardless of the reference.
			if e > lpTolerance {
				v.Pass = false
				v.FirstDivergence = &Divergence{Index: i, Reference: lpEntry(ref, i, j), Engine: lpEntry(eng, i, j)}
				v.Detail = fmt.Sprintf("token %d (%q, %s segment) candidate %d reports logprob %.6f > 0 — a probability's log cannot be positive; raw logit reported as logprob",
					i, tokenAt(eng.Tokens, i), lpSegment(i, promptLen), j, e)
				return v
			}
			if math.Abs(e-refRow[j]) > lpTolerance {
				v.Pass = false
				v.FirstDivergence = &Divergence{Index: i, Reference: lpEntry(ref, i, j), Engine: lpEntry(eng, i, j)}
				v.Detail = fmt.Sprintf("logprob for token %d (%q, %s segment) diverged at candidate %d: reference %.6f, engine %.6f — %s",
					i, tokenAt(eng.Tokens, i), lpSegment(i, promptLen), j, refRow[j], e, lpClassify(ref, i, engRow))
				return v
			}
		}
	}

	p := promptLen
	if n := len(ref.Logits); p > n {
		p = n
	}
	v.Detail = fmt.Sprintf("%d logprob rows aligned and within %g of reference (%d prompt + %d generation)",
		len(ref.Logits), lpTolerance, p, len(ref.Logits)-p)
	return v
}
