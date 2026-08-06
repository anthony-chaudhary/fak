package quality

import (
	"fmt"
	"sort"
	"strings"
)

// nightly_matrix.go is the nightly model-backend-engine matrix (#4577, under epic
// #4509): the tier that runs REPRESENTATIVE MODELS across supported backends and
// engine modes on a set of task slices, rather than one case on one path. A defect
// that only appears on ROCm, or only in graph mode, or only on the long-context
// slice, is invisible to every single-cell gate; the matrix exists to make the
// whole product one verdict.
//
// The hard part of a matrix is not fanning out — it is what comes back. One root
// regression lights up EVERY cell that shares the broken coordinate, so a nightly
// that files one issue per failing cell buries the operator in N copies of the
// same bug and nobody reads the tenth. So this file folds:
//
//   - Failing cells are DEDUPLICATED by defect signature (the first failing
//     oracle, its kind, and the serving stage the bundle's own evidence points
//     at). One injected regression files ONE issue, listing every affected cell.
//   - Each issue is ATTRIBUTED to the axis coordinates every affected cell shares
//     and no passing cell holds — the matrix's whole informational advantage, so
//     "eight cells failed" becomes "backend=rocm regressed".
//   - Each issue is REPLAY-BACKED: it carries one representative scrubbed
//     FailureBundle, which embeds the case and both traces, so the issue reproduces
//     from the artifact alone instead of from a nightly log nobody kept.
//   - Missing evidence is NEVER a pass. A cell whose engine could not be built,
//     whose oracles do not resolve, whose run errored, or which failed without
//     emitting a scrubbed artifact is recorded INCONCLUSIVE and folds into its own
//     issue. An unrun cell is not a green cell.
//
// It is additive: it registers no oracle and edits no core. Routing and cost come
// from the #4574 suite splitter (SplitSuites), so a matrix cell is budgeted by the
// same rule as every other case and the nightly tier's cost envelope is reported
// rather than asserted.

// NightlyMatrixSchema is the versioned tag on a matrix report. Consumers pin the
// major so a schema bump is a conscious migration (the #4519 house rule).
const NightlyMatrixSchema = "fak-quality-nightly-matrix/1"

// MatrixModel is one representative model on the model axis, with the tokenizer
// it is decoded under. The two travel together because a tokenizer swap changes
// the token stream every differential oracle compares — recording the model
// without its tokenizer would make a divergence unattributable.
type MatrixModel struct {
	Model     Revision `json:"model"`
	Tokenizer Revision `json:"tokenizer"`
}

// MatrixSlice is one task slice: the prompt, the decode configuration, the
// reference trace that slice is judged against, and the oracles to apply. Slices
// are the "task" axis — a matrix that varies only the hardware proves nothing
// about whether a model still answers the questions it is deployed to answer.
type MatrixSlice struct {
	Name      string         `json:"name"`
	Prompt    string         `json:"prompt"`
	Params    SamplingParams `json:"params"`
	Reference Trace          `json:"reference"`
	Oracles   []string       `json:"oracles"`
	Rubric    RubricSpec     `json:"rubric,omitempty"`
}

// MatrixSpec declares one nightly matrix: the four axes plus the provenance every
// expanded cell inherits. The provenance fields are spec-level on purpose — a
// matrix qualifies ONE code revision against ONE baseline under ONE tolerance
// policy, so letting a cell carry its own would let a green matrix be assembled
// from cells that were never comparable.
type MatrixSpec struct {
	// Engine is the engine implementation under test; Backends and Modes are the
	// supported backends and engine modes it is exercised across.
	Engine   string        `json:"engine"`
	Models   []MatrixModel `json:"models"`
	Backends []string      `json:"backends"`
	Modes    []string      `json:"modes"`
	Slices   []MatrixSlice `json:"slices"`

	Code      Revision       `json:"code"`
	Oracle    OracleEvidence `json:"oracle"`
	Tolerance ToleranceSpec  `json:"tolerance"`
	Baseline  BaselineSpec   `json:"baseline"`
	Owner     string         `json:"owner"`
	Family    string         `json:"family"`
	// Cost is the per-cell runtime and resource declaration. It is checked against
	// the nightly tier budget by SplitSuites, so a cell too expensive for the
	// nightly wall is refused with the ceiling it broke instead of quietly making
	// the nightly overrun.
	Cost CostSpec `json:"cost"`
}

// MatrixCell is one point in the matrix: the case id it expands to and its four
// axis coordinates. It is the unit an operator reads a failure in.
type MatrixCell struct {
	CaseID  string `json:"case_id"`
	Model   string `json:"model"`
	Backend string `json:"backend"`
	Mode    string `json:"mode"`
	Slice   string `json:"slice"`
}

// coords renders the cell as its axis coordinates, in fixed axis order. It is the
// vocabulary attribution is computed in.
func (c MatrixCell) coords() []string {
	return []string{
		"model=" + c.Model,
		"backend=" + c.Backend,
		"mode=" + c.Mode,
		"slice=" + c.Slice,
	}
}

// MatrixCase is one expanded cell: its coordinates and the canonical quality case
// those coordinates produce.
type MatrixCase struct {
	Cell MatrixCell  `json:"cell"`
	Case QualityCase `json:"case"`
}

// MatrixOutcome is the closed set of states one cell can end in. There is
// deliberately no "skipped": a cell that did not produce evidence is
// Inconclusive, which blocks exactly like a failure.
type MatrixOutcome string

const (
	MatrixPass         MatrixOutcome = "pass"
	MatrixFail         MatrixOutcome = "fail"
	MatrixInconclusive MatrixOutcome = "inconclusive"
)

// Causes recorded on an inconclusive cell's signature. They are the dedup key for
// the inconclusive issue, so "no engine on four cells" is one issue and not four.
const (
	causeEngineUnavailable = "engine-unavailable"
	causeOracleUnresolved  = "oracle-unresolved"
	causeRunError          = "run-error"
	causeNoArtifact        = "no-replay-artifact"
	causeUnscrubbed        = "unscrubbed-artifact"
)

// MatrixSignature is the identity of a defect as the matrix sees it: which oracle
// failed first, of which kind, and which serving stage the bundle's own evidence
// attributes it to. Cells sharing a signature are the SAME defect observed at
// different coordinates, which is what makes the fold a deduplication rather than
// a summary. On an inconclusive cell, Kind is "inconclusive" and Stage is the
// cause the evidence went missing.
type MatrixSignature struct {
	FailingOracle string `json:"failing_oracle,omitempty"`
	FailingKind   string `json:"failing_kind"`
	Stage         string `json:"stage"`
}

func (s MatrixSignature) key() string {
	return s.FailingOracle + "|" + s.FailingKind + "|" + s.Stage
}

// String renders the signature as the one line that names the defect.
func (s MatrixSignature) String() string {
	if s.FailingOracle == "" {
		return fmt.Sprintf("%s (%s)", s.FailingKind, s.Stage)
	}
	return fmt.Sprintf("%s (%s) at stage %s", s.FailingOracle, s.FailingKind, s.Stage)
}

// MatrixCellResult is one cell's outcome plus, when it did not pass, the
// signature it folds under and the detail that localized it.
type MatrixCellResult struct {
	Cell      MatrixCell       `json:"cell"`
	Outcome   MatrixOutcome    `json:"outcome"`
	Signature *MatrixSignature `json:"signature,omitempty"`
	Detail    string           `json:"detail"`
	// bundle is this cell's scrubbed failure artifact, carried unexported so it
	// reaches the fold without republishing one full bundle per failing cell — the
	// deduplicated issue publishes exactly one of them.
	bundle *FailureBundle
}

// MatrixIssue is ONE deduplicated finding: every cell that observed a single
// defect signature, the axis coordinates that defect is attributed to, and one
// representative scrubbed replay artifact. Filing this — rather than one issue per
// failing cell — is the #4577 acceptance criterion.
type MatrixIssue struct {
	ID        string          `json:"id"`
	Kind      string          `json:"kind"` // "regression" or "inconclusive"
	Signature MatrixSignature `json:"signature"`
	// Attribution is the axis coordinates shared by every affected cell that no
	// passing cell holds — the narrowest coordinate the evidence indicts. It is
	// empty when the failures span the matrix with no common exonerated axis, and
	// saying so is the honest answer.
	Attribution []string     `json:"attribution,omitempty"`
	Cells       []MatrixCell `json:"cells"`
	Detail      string       `json:"detail"`
	// Replay is the representative scrubbed failure bundle: it embeds the case and
	// both traces, so this issue reproduces from the artifact alone. It is absent
	// only on an inconclusive issue, where by definition no artifact was produced.
	Replay *FailureBundle `json:"replay,omitempty"`
}

// MatrixReport is the machine-readable outcome of one nightly matrix run: the
// nightly suite the cells were routed into (carrying the tier's summed cost
// envelope), every cell's outcome, the deduplicated issues, and every cell the
// router refused.
type MatrixReport struct {
	Schema   string             `json:"schema"`
	Tier     Tier               `json:"tier"`
	Suite    Suite              `json:"suite"`
	Cells    []MatrixCellResult `json:"cells"`
	Issues   []MatrixIssue      `json:"issues,omitempty"`
	Rejected []SuiteReject      `json:"rejected,omitempty"`
}

// MatrixEngine resolves the engine runner for one cell. A real harness returns an
// adapter bound to that model/backend/mode build; a returned error is not a
// failure of the code under test but an absence of evidence, so RunMatrix records
// it as inconclusive rather than as a pass or a fail.
type MatrixEngine func(cell MatrixCell, c QualityCase) (Runner, error)

// ExpandMatrix expands a spec into one canonical quality case per
// (model, backend, mode, slice) point. Every case is stamped to the NIGHTLY tier
// and inherits the spec's provenance, so each cell records model, tokenizer,
// engine/backend, its deterministic oracle or seed, the code revision, and the
// tolerance/baseline it is judged under.
//
// Order is fixed — models, then backends, then modes, then slices — so a matrix
// replays cell-for-cell. Two points that collide on a case id (a repeated axis
// value in the spec) are refused as ambiguous rather than racing: one id cannot
// carry two coordinates.
func ExpandMatrix(spec MatrixSpec) ([]MatrixCase, []SuiteReject) {
	var (
		out      []MatrixCase
		rejected []SuiteReject
		seen     = map[string]MatrixCell{}
	)
	for _, m := range spec.Models {
		for _, backend := range spec.Backends {
			for _, mode := range spec.Modes {
				for _, slice := range spec.Slices {
					cell := MatrixCell{
						CaseID:  matrixCaseID(m.Model.Name, backend, mode, slice.Name),
						Model:   m.Model.Name,
						Backend: backend,
						Mode:    mode,
						Slice:   slice.Name,
					}
					if first, dup := seen[cell.CaseID]; dup {
						rejected = append(rejected, SuiteReject{
							CaseID: cell.CaseID, Tier: TierNightly,
							Reason: fmt.Sprintf("duplicate matrix coordinate: %s already expands to this case id — one id cannot carry two cells",
								strings.Join(first.coords(), " ")),
						})
						continue
					}
					seen[cell.CaseID] = cell
					out = append(out, MatrixCase{Cell: cell, Case: matrixCase(spec, m, backend, mode, slice, cell.CaseID)})
				}
			}
		}
	}
	return out, rejected
}

func matrixCaseID(model, backend, mode, slice string) string {
	return fmt.Sprintf("nightly-matrix/%s/%s/%s/%s", model, backend, mode, slice)
}

// matrixCase builds one cell's canonical case. The backend and mode are written
// into the engine spec's replay flags rather than into the prompt: they are
// execution coordinates, so a replayer reads them from the provenance it already
// has to carry.
func matrixCase(spec MatrixSpec, m MatrixModel, backend, mode string, slice MatrixSlice, id string) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        id,
		Version:   1,
		Prompt:    slice.Prompt,
		Params:    slice.Params,
		Reference: slice.Reference,
		Oracles:   append([]string(nil), slice.Oracles...),
		Rubric:    slice.Rubric,
		Metadata: CaseMetadata{
			Model:     m.Model,
			Tokenizer: m.Tokenizer,
			Engine: EngineSpec{
				Name:    spec.Engine,
				Backend: backend,
				Flags:   map[string]string{"mode": mode},
			},
			Code:      spec.Code,
			Oracle:    spec.Oracle,
			Tolerance: spec.Tolerance,
			Baseline:  spec.Baseline,
			Tier:      TierSpec{Name: string(TierNightly)},
			Cost:      spec.Cost,
			Owner:     spec.Owner,
			Family:    spec.Family,
		},
		Tags: []string{"nightly-matrix", "backend=" + backend, "mode=" + mode, "slice=" + slice.Name},
	}
}

// RunMatrix expands, routes, runs, and folds one nightly matrix. Budgets nil means
// DefaultBudgets. It never returns an error: every way a matrix can go wrong —
// an unroutable cell, an unavailable engine, a failing oracle — is a state in the
// report a gate can route on, and only MatrixReport.Green is a pass.
func RunMatrix(spec MatrixSpec, engine MatrixEngine, budgets map[Tier]TierBudget) MatrixReport {
	rep := MatrixReport{Schema: NightlyMatrixSchema, Tier: TierNightly}

	expanded, rejected := ExpandMatrix(spec)
	cases := make([]QualityCase, 0, len(expanded))
	for _, e := range expanded {
		cases = append(cases, e.Case)
	}
	// Routing and cost are the #4574 splitter's job, not this file's: a matrix cell
	// is budgeted by exactly the rule every other case is, and the nightly suite it
	// lands in carries the summed runtime/resource envelope the tier costs.
	plan := SplitSuites(cases, budgets)
	rep.Rejected = append(rejected, plan.Rejected...)
	sort.SliceStable(rep.Rejected, func(i, j int) bool { return rep.Rejected[i].CaseID < rep.Rejected[j].CaseID })
	for _, s := range plan.Suites {
		if s.Tier == TierNightly {
			rep.Suite = s
		}
	}

	placed := make(map[string]bool, len(rep.Suite.Cases))
	for _, sc := range rep.Suite.Cases {
		placed[sc.CaseID] = true
	}
	// Cells run in EXPANSION order, not the suite's cheapest-first order, so the
	// readout walks the matrix axes the way an operator reasons about them.
	for _, e := range expanded {
		if !placed[e.Cell.CaseID] {
			continue // refused by the router; already recorded in rep.Rejected
		}
		rep.Cells = append(rep.Cells, runMatrixCell(e, engine))
	}
	rep.Issues = foldMatrixIssues(rep.Cells)
	return rep
}

// runMatrixCell runs one cell to an outcome. Every path that ends without a
// scrubbed, localized artifact ends INCONCLUSIVE with the cause named — never a
// pass, and never a bare failure that cannot be replayed.
func runMatrixCell(e MatrixCase, engine MatrixEngine) MatrixCellResult {
	res := MatrixCellResult{Cell: e.Cell}
	inconclusive := func(cause, detail string) MatrixCellResult {
		res.Outcome = MatrixInconclusive
		res.Signature = &MatrixSignature{FailingKind: string(MatrixInconclusive), Stage: cause}
		res.Detail = detail
		return res
	}

	oracles, err := Lookup(e.Case.Oracles)
	if err != nil {
		return inconclusive(causeOracleUnresolved, "cell names an unrunnable oracle: "+err.Error())
	}
	if engine == nil {
		return inconclusive(causeEngineUnavailable, "no engine resolver was supplied for the matrix")
	}
	eng, err := engine(e.Cell, e.Case)
	if err != nil {
		return inconclusive(causeEngineUnavailable, "engine unavailable for this cell: "+err.Error())
	}
	if eng == nil {
		return inconclusive(causeEngineUnavailable, "engine resolver returned no runner for this cell")
	}
	run, err := RunCase(e.Case, ReferenceRunner{}, eng, oracles)
	if err != nil {
		return inconclusive(causeRunError, "cell did not run to a verdict: "+err.Error())
	}
	if run.Pass {
		res.Outcome = MatrixPass
		res.Detail = fmt.Sprintf("%d oracle(s) agreed with the reference", len(run.Verdicts))
		return res
	}
	fb := run.FailureBundle
	if fb == nil {
		return inconclusive(causeNoArtifact, "cell failed but emitted no replay artifact, so the failure cannot be reproduced")
	}
	if !fb.Scrubbed {
		return inconclusive(causeUnscrubbed, "cell emitted an artifact the spine did not redact; refusing to file an unscrubbed replay")
	}
	stage := Classify(run)
	res.Outcome = MatrixFail
	res.Signature = &MatrixSignature{FailingOracle: fb.FailingOracle, FailingKind: fb.FailingKind, Stage: stage.Stage}
	res.Detail = fb.Detail
	res.bundle = fb
	return res
}

// foldMatrixIssues deduplicates the non-passing cells into one issue per defect
// signature, attributes each issue to the axis coordinates its cells share and no
// passing cell holds, and attaches the first affected cell's scrubbed bundle as
// the issue's replay artifact. Issues are returned in id order so a nightly's
// output is stable across runs.
func foldMatrixIssues(cells []MatrixCellResult) []MatrixIssue {
	var passing []MatrixCell
	groups := map[string]*MatrixIssue{}
	var order []string
	for _, c := range cells {
		if c.Outcome == MatrixPass {
			passing = append(passing, c.Cell)
			continue
		}
		if c.Signature == nil {
			continue
		}
		k := c.Signature.key()
		g, ok := groups[k]
		if !ok {
			kind := "regression"
			if c.Outcome == MatrixInconclusive {
				kind = "inconclusive"
			}
			g = &MatrixIssue{
				ID:        kind + ":" + k,
				Kind:      kind,
				Signature: *c.Signature,
				Detail:    c.Detail,
				Replay:    c.bundle,
			}
			groups[k] = g
			order = append(order, k)
		}
		g.Cells = append(g.Cells, c.Cell)
	}

	issues := make([]MatrixIssue, 0, len(order))
	for _, k := range order {
		g := groups[k]
		g.Attribution = attributeMatrix(g.Cells, passing)
		issues = append(issues, *g)
	}
	sort.SliceStable(issues, func(i, j int) bool { return issues[i].ID < issues[j].ID })
	return issues
}

// attributeMatrix narrows an issue to the axis coordinates EVERY affected cell
// shares and NO passing cell holds. The intersection alone is not enough — in a
// single-model matrix every failing cell trivially shares that model — so a
// coordinate a passing cell also holds is exonerated by that pass and dropped.
// What survives is the narrowest coordinate the matrix's own evidence indicts.
func attributeMatrix(failing, passing []MatrixCell) []string {
	if len(failing) == 0 {
		return nil
	}
	shared := map[string]bool{}
	for _, c := range failing[0].coords() {
		shared[c] = true
	}
	for _, cell := range failing[1:] {
		have := map[string]bool{}
		for _, c := range cell.coords() {
			have[c] = true
		}
		for k := range shared {
			if !have[k] {
				delete(shared, k)
			}
		}
	}
	exonerated := map[string]bool{}
	for _, cell := range passing {
		for _, c := range cell.coords() {
			exonerated[c] = true
		}
	}
	var out []string
	for k := range shared {
		if !exonerated[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// Green is the fail-closed verdict on a matrix report, naming the first blocking
// reason. A refused cell blocks (refusing it is the point), any issue blocks
// (including an inconclusive one — missing evidence is never a pass), and a matrix
// that ran no cell at all blocks, because a matrix that qualified nothing has
// produced no evidence.
func (r MatrixReport) Green() (bool, string) {
	if len(r.Rejected) > 0 {
		j := r.Rejected[0]
		why := fmt.Sprintf("cell %s not routed: %s", j.CaseID, j.Reason)
		if more := len(r.Rejected) - 1; more > 0 {
			why += fmt.Sprintf(" (+%d more refused)", more)
		}
		return false, why
	}
	if len(r.Issues) > 0 {
		i := r.Issues[0]
		why := fmt.Sprintf("%s issue %s across %d cell(s): %s", i.Kind, i.Signature, len(i.Cells), i.Detail)
		if len(i.Attribution) > 0 {
			why += " — attributed to " + strings.Join(i.Attribution, " ")
		}
		if more := len(r.Issues) - 1; more > 0 {
			why += fmt.Sprintf(" (+%d more issue(s))", more)
		}
		return false, why
	}
	if len(r.Cells) == 0 {
		return false, "no matrix cell ran — a matrix that qualifies nothing is not evidence"
	}
	return true, ""
}

// ExplainMatrix renders a report as the operator readout: the nightly tier's cost
// envelope, the cell tally, then each deduplicated issue with the coordinates it
// is attributed to, its replay handle, and every cell it covers. It mirrors
// Explain / ExplainPlan — the bridge from a machine report to "one bug, here is
// where it lives, here is the artifact that reproduces it".
func ExplainMatrix(r MatrixReport) string {
	var b strings.Builder
	green, why := r.Green()
	state := "FAIL"
	if green {
		state = "PASS"
	}
	pass, fail, inc := 0, 0, 0
	for _, c := range r.Cells {
		switch c.Outcome {
		case MatrixPass:
			pass++
		case MatrixFail:
			fail++
		default:
			inc++
		}
	}
	fmt.Fprintf(&b, "%s  nightly matrix — %d cell(s): %d pass, %d fail, %d inconclusive, %d refused\n",
		state, len(r.Cells), pass, fail, inc, len(r.Rejected))
	fmt.Fprintf(&b, "  tier %s cost: runtime=%ds timeout=%ds cpu<=%d mem<=%dMiB accel<=%d\n",
		r.Tier, r.Suite.TotalRuntimeSec, r.Suite.TotalTimeoutSec, r.Suite.MaxCPU, r.Suite.MaxMemoryMiB, r.Suite.MaxAccelerators)
	if !green {
		fmt.Fprintf(&b, "  blocked: %s\n", why)
	}
	for _, i := range r.Issues {
		fmt.Fprintf(&b, "ISSUE %s — %s across %d cell(s)\n", i.ID, i.Signature, len(i.Cells))
		if len(i.Attribution) > 0 {
			fmt.Fprintf(&b, "  attributed to: %s\n", strings.Join(i.Attribution, " "))
		} else {
			fmt.Fprintf(&b, "  attributed to: no axis coordinate is shared by every affected cell and absent from every passing cell\n")
		}
		fmt.Fprintf(&b, "  detail: %s\n", i.Detail)
		if i.Replay != nil {
			fmt.Fprintf(&b, "  replay: scrubbed bundle for case %s @ v%d, oracle %s\n",
				i.Replay.CaseID, i.Replay.Case.Version, i.Replay.FailingOracle)
			if d := i.Replay.FirstDivergence; d != nil {
				fmt.Fprintf(&b, "  first divergence at token %d: reference %q, engine %q\n", d.Index, d.Reference, d.Engine)
			}
		} else {
			fmt.Fprintf(&b, "  replay: none — this issue is an ABSENCE of evidence, which is never a pass\n")
		}
		for _, c := range i.Cells {
			fmt.Fprintf(&b, "    %s\n", strings.Join(c.coords(), " "))
		}
	}
	for _, j := range r.Rejected {
		fmt.Fprintf(&b, "REFUSED %s: %s\n", j.CaseID, j.Reason)
	}
	return b.String()
}
