package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// dtype_delta.go is the dtype-parity child of the quality spine (#4541): it
// qualifies FP32, BF16, and FP16 execution against the FP32 reference logits
// under DECLARED per-dtype accuracy budgets. A numeric-dtype defect is the
// classic "fluent but shifted" engine failure: every emitted token still reads
// fine while the logits drift, so parity is judged on the logit surface, per
// dtype, against the band each dtype's build declared — never on the text.
//
// Each dtype is modeled as the reference logits pushed through a
// dtype-specific rounding grid (step 2^-bits), and each dtype's band is 4x its
// grid's worst-case rounding error: a faithful lane always passes with
// headroom, and only an execution defect — not honest rounding — can exceed
// the band. The budgets are intentionally per-dtype and NOT interchangeable:
// BF16's band is tighter than FP16's, so a delta FP16 may absorb is still a
// BF16 failure.

// dtSpec pins one modeled dtype: its rounding grid (step 2^-bits) and the
// declared absolute logit-delta band it must stay within vs the FP32 reference.
type dtSpec struct {
	name string
	bits int     // rounding grid step is 2^-bits
	band float64 // declared max |engine - reference| per logit
}

// dtSpecs is the fixed dtype ladder, judged in this order. Each band is
// 2^-(bits-1) — 4x that grid's worst-case rounding error 2^-(bits+1) — so
// honest rounding always fits and the declared budgets stay strictly ordered:
// fp32 tightest, bf16 tighter than fp16.
var dtSpecs = []dtSpec{
	{name: "fp32", bits: 20, band: math.Ldexp(1, -19)},
	{name: "bf16", bits: 10, band: math.Ldexp(1, -9)},
	{name: "fp16", bits: 7, band: math.Ldexp(1, -6)},
}

// dtBand returns the declared band for a dtype name (0 for an unknown dtype).
func dtBand(name string) float64 {
	for _, s := range dtSpecs {
		if s.name == name {
			return s.band
		}
	}
	return 0
}

// Fixture geometry and defect coordinates. The defects sit mid-sequence so the
// passing prefix proves the localization is doing work.
const (
	dtSteps          = 6 // decode steps in the fixture
	dtWidth          = 4 // logit candidates captured per step
	dtDefectCand     = 1 // candidate column the mutants corrupt
	dtFP16DefectStep = 4 // step the fp16-blowout mutant corrupts
	dtBF16DefectStep = 2 // step the bf16-band mutant corrupts
)

// dtMidBandBump is the bf16-band mutant's drift: 2^-7 sits strictly between
// the bf16 band (2^-9) and the fp16 band (2^-6), so the SAME delta fails the
// tighter bf16 budget while remaining within fp16's — the per-dtype witness.
var dtMidBandBump = math.Ldexp(1, -7)

// dtRound quantizes x onto the 2^-bits grid — the hermetic model of one
// dtype's execution: same math, coarser representable values.
func dtRound(x float64, bits int) float64 {
	s := math.Ldexp(1, bits)
	return math.Round(x*s) / s
}

// dtLogit maps (step, candidate) to a deterministic pseudo-random logit in
// [-8, 8) via a splitmix64-style mix — a pure function of its inputs, so the
// fixture carries no ambient entropy and replays byte-identically.
func dtLogit(step, cand int) float64 {
	z := uint64(step*dtWidth+cand+1) * 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return (float64(z%100000)/100000 - 0.5) * 16
}

// dtReferenceLogits builds the FP32 reference logit rows for the fixture.
func dtReferenceLogits(steps, width int) [][]float64 {
	rows := make([][]float64, steps)
	for i := range rows {
		row := make([]float64, width)
		for j := range row {
			row[j] = dtLogit(i, j)
		}
		rows[i] = row
	}
	return rows
}

// dtLanes is the per-dtype logit payload an engine trace carries, serialized
// as JSON into Trace.Text (the additive seam for per-case data that
// Tokens/Logits do not carry): dtype name -> per-step logit rows, mirroring
// the shape of the reference logits.
type dtLanes map[string][][]float64

// dtEncodeLanes serializes lanes into the Trace.Text payload. json.Marshal
// sorts map keys, so the encoding is deterministic, and Go's float64 JSON
// encoding round-trips exactly, so decoded deltas are exact.
func dtEncodeLanes(l dtLanes) string {
	b, _ := json.Marshal(l)
	return string(b)
}

// dtDecodeLanes parses an engine trace's dtype-lane payload.
func dtDecodeLanes(text string) (dtLanes, error) {
	var l dtLanes
	if err := json.Unmarshal([]byte(text), &l); err != nil {
		return nil, err
	}
	return l, nil
}

// dtCleanLanes runs the faithful multi-dtype execution model: every dtype lane
// is the reference logits rounded onto that dtype's own grid, nothing more.
func dtCleanLanes(ref [][]float64) dtLanes {
	lanes := make(dtLanes, len(dtSpecs))
	for _, s := range dtSpecs {
		rows := make([][]float64, len(ref))
		for i, row := range ref {
			out := make([]float64, len(row))
			for j, x := range row {
				out[j] = dtRound(x, s.bits)
			}
			rows[i] = out
		}
		lanes[s.name] = rows
	}
	return lanes
}

// DtypeDeltaCase builds the dtype-parity fixture: a temperature-zero decode
// whose reference trace carries the FP32 logits every dtype lane is judged
// against.
func DtypeDeltaCase() QualityCase {
	toks := []string{"Throughput", "rose", "under", "mixed", "precision", "decode"}
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "dtype-parity-demo",
		Version: 1,
		Prompt:  "Decode the fixture under FP32, BF16, and FP16 execution and report each dtype's logit lane.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: dtSteps},
		Reference: Trace{
			Tokens: toks,
			Logits: dtReferenceLogits(dtSteps, dtWidth),
			Text:   strings.Join(toks, " "),
		},
		Oracles: []string{"dtype-parity"},
	}
}

// DtypeDeltaEngine returns an engine runner for the dtype fixture with an
// optional injected defect: "" executes every dtype lane faithfully (honest
// per-dtype rounding only); "fp16-blowout" pushes one fp16 logit 4 bands past
// the reference at step dtFP16DefectStep — the half-precision kernel drift
// this child exists to catch; "bf16-band" drifts one bf16 logit by a delta the
// fp16 band would absorb, proving the budgets are per-dtype. These are the
// deterministic mutant sources the tests use to prove the gate trips.
func DtypeDeltaEngine(defect string) ScriptedRunner {
	ref := DtypeDeltaCase().Reference
	lanes := dtCleanLanes(ref.Logits)
	label := "engine-dtype-clean"
	switch defect {
	case "fp16-blowout":
		lanes["fp16"][dtFP16DefectStep][dtDefectCand] += 4 * dtBand("fp16")
		label = "engine-fp16-blowout"
	case "bf16-band":
		lanes["bf16"][dtBF16DefectStep][dtDefectCand] += dtMidBandBump
		label = "engine-bf16-band"
	}
	return ScriptedRunner{
		Label: label,
		Trace: Trace{Tokens: append([]string(nil), ref.Tokens...), Text: dtEncodeLanes(lanes)},
	}
}

// DtypeDelta is the dtype-parity differential oracle (#4541): every declared
// dtype lane in the engine payload must stay within its OWN band of the FP32
// reference logits. Lanes are judged in the fixed dtSpecs order, tokens
// ascending, and the first out-of-band logit is reported as the first
// divergence with the offending dtype and token named — "FP16 looked off"
// becomes "dtype fp16 diverged beyond its band at token 4".
type DtypeDelta struct{}

func (DtypeDelta) Name() string { return "dtype-parity" }
func (DtypeDelta) Kind() string { return "differential" }

func init() { Register(DtypeDelta{}) }

func (DtypeDelta) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "dtype-parity", Kind: "differential", Pass: true}
	if len(ref.Logits) == 0 {
		v.Pass = false
		v.Detail = "reference trace carries no logits; dtype parity cannot be judged"
		return v
	}
	lanes, err := dtDecodeLanes(eng.Text)
	if err != nil {
		v.Pass = false
		v.Detail = fmt.Sprintf("engine trace text is not a dtype-lane payload: %v", err)
		return v
	}
	worst := make([]float64, len(dtSpecs))
	for si, spec := range dtSpecs {
		rows, ok := lanes[spec.name]
		if !ok {
			v.Pass = false
			v.Detail = fmt.Sprintf("engine payload is missing dtype lane %q; a lane never executed cannot be within budget", spec.name)
			return v
		}
		if len(rows) != len(ref.Logits) {
			v.Pass = false
			v.Detail = fmt.Sprintf("dtype %s emitted %d logit steps, reference has %d", spec.name, len(rows), len(ref.Logits))
			return v
		}
		for i, row := range rows {
			if len(row) != len(ref.Logits[i]) {
				v.Pass = false
				v.Detail = fmt.Sprintf("dtype %s step %d emitted %d candidates, reference has %d", spec.name, i, len(row), len(ref.Logits[i]))
				return v
			}
			for j, got := range row {
				delta := math.Abs(got - ref.Logits[i][j])
				if delta > worst[si] {
					worst[si] = delta
				}
				if delta > spec.band {
					v.Pass = false
					v.FirstDivergence = &Divergence{
						Index:     i,
						Reference: strconv.FormatFloat(ref.Logits[i][j], 'g', -1, 64),
						Engine:    strconv.FormatFloat(got, 'g', -1, 64),
					}
					v.Detail = fmt.Sprintf("dtype %s diverged beyond its band at token %d (%q): |delta| %.6g > band %.6g (candidate %d)",
						spec.name, i, tokenAt(ref.Tokens, i), delta, spec.band, j)
					return v
				}
			}
		}
	}
	parts := make([]string, len(dtSpecs))
	for si, spec := range dtSpecs {
		parts[si] = fmt.Sprintf("%s worst |delta| %.3g within band %.3g", spec.name, worst[si], spec.band)
	}
	v.Detail = "all dtype lanes within declared budgets: " + strings.Join(parts, "; ")
	return v
}
