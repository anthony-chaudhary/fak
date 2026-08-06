package quality

import (
	"sort"
	"strings"
	"testing"
)

// This is the #4577 witness. The matrix fans one corpus across
// (model x backend x mode x slice), and the property under test is what comes
// BACK: a single planted regression must file exactly ONE deduplicated,
// replay-backed, axis-attributed issue — not one per affected cell — and the same
// matrix must go green again once the defect is removed. The replay half is
// exercised independently: the issue's bundle is re-run through fresh runners
// built from the bundle ALONE, with no access to the spec or the engine that
// produced it.

// mtxLogitRow is the deterministic per-step logit capture both paths record. It
// is a pure function of (slice, step), so a matrix replays bit-for-bit and the
// stage classifier has real numeric evidence to attribute a divergence to.
func mtxLogitRow(slice string, step int) []float64 {
	h := 0
	for i := 0; i < len(slice); i++ {
		h = h*31 + int(slice[i])
	}
	row := make([]float64, 3)
	for j := range row {
		row[j] = float64((h+step*7+j*13)%64)/8.0 - float64(j)
	}
	return row
}

// mtxSlice builds one task slice with a reference trace that carries per-step
// logits alongside its tokens.
func mtxSlice(name, prompt string, toks []string) MatrixSlice {
	logits := make([][]float64, len(toks))
	for i := range toks {
		logits[i] = mtxLogitRow(name, i)
	}
	return MatrixSlice{
		Name:   name,
		Prompt: prompt,
		Params: SamplingParams{Temperature: 0, MaxTokens: len(toks)},
		Reference: Trace{
			Tokens: toks,
			Logits: logits,
			Text:   strings.Join(toks, " "),
		},
		Oracles: []string{"greedy-token-diff"},
	}
}

// mtxSpec is the representative nightly matrix: two models, three supported
// backends, two engine modes, two task slices — 24 cells, all routable to the
// nightly tier under the default budgets.
func mtxSpec() MatrixSpec {
	return MatrixSpec{
		Engine: "fak",
		Models: []MatrixModel{
			{Model: Revision{Name: "small-instruct", Revision: "sha256:m1"}, Tokenizer: Revision{Name: "bpe-32k", Revision: "sha256:t1"}},
			{Model: Revision{Name: "mid-instruct", Revision: "sha256:m2"}, Tokenizer: Revision{Name: "bpe-64k", Revision: "sha256:t2"}},
		},
		Backends: []string{"cpu", "cuda", "rocm"},
		Modes:    []string{"eager", "graph"},
		Slices: []MatrixSlice{
			mtxSlice("throughput", "Summarize this week's throughput.", []string{"Throughput", "increased", "12", "%", "."}),
			mtxSlice("latency", "Summarize this week's latency.", []string{"Latency", "fell", "to", "40", "ms", "."}),
		},
		Code:      Revision{Name: "github.com/anthony-chaudhary/fak", Revision: "git:deadbeef"},
		Oracle:    OracleEvidence{Kind: "exact-greedy-trace", Revision: "sha256:o1"},
		Tolerance: ToleranceSpec{Metric: "exact-token", Revision: "policy:v1"},
		Baseline:  BaselineSpec{ID: "nightly-baseline", Revision: "sha256:b1"},
		Owner:     "quality-team",
		Family:    string(FamilyCorpora),
		Cost:      CostSpec{RuntimeSeconds: 30, TimeoutSeconds: 300, CPU: 4, MemoryMiB: 2048},
	}
}

// Planted defect classes for mtxEngine.
const (
	mtxDefectROCmKernel = "rocm-kernel" // rocm flips a token and drifts its logit row
	mtxDefectGraphStop  = "graph-stop"  // graph mode ignores the stop and keeps decoding
	mtxDefectROCmDown   = "rocm-down"   // the rocm build could not be resolved at all
)

// mtxEngine is the matrix's engine resolver with an optional planted regression.
// A faithful cell replays the slice's reference verbatim; a defective cell mutates
// a DEEP COPY, so the shared slice reference is never corrupted.
func mtxEngine(defects ...string) MatrixEngine {
	has := func(d string) bool {
		for _, x := range defects {
			if x == d {
				return true
			}
		}
		return false
	}
	return func(cell MatrixCell, c QualityCase) (Runner, error) {
		if has(mtxDefectROCmDown) && cell.Backend == "rocm" {
			return nil, errNoBuild{cell.Backend}
		}
		toks := append([]string(nil), c.Reference.Tokens...)
		logits := make([][]float64, len(c.Reference.Logits))
		for i, row := range c.Reference.Logits {
			logits[i] = append([]float64(nil), row...)
		}
		label := "engine-clean"
		if has(mtxDefectROCmKernel) && cell.Backend == "rocm" {
			// A backend kernel regression: the distribution at step 1 drifts and the
			// argmax flips with it. Real, localizable, and identical in KIND across
			// every model and slice it touches — which is exactly what must dedupe.
			toks[1] = "REGRESSED"
			logits[1][0] += 0.25
			label = "engine-rocm-kernel-defect"
		}
		if has(mtxDefectGraphStop) && cell.Mode == "graph" {
			// A different defect class: every shared token agrees and the engine
			// simply keeps decoding past the reference's last token.
			toks = append(toks, "and", "more")
			logits = append(logits, mtxLogitRow(cell.Slice, len(logits)), mtxLogitRow(cell.Slice, len(logits)+1))
			label = "engine-graph-stop-defect"
		}
		return ScriptedRunner{Label: label, Trace: Trace{
			Tokens: toks, Logits: logits, Text: strings.Join(toks, " "),
		}}, nil
	}
}

type errNoBuild struct{ backend string }

func (e errNoBuild) Error() string {
	return "no " + e.backend + " build was published for this revision"
}

func mtxIssueIDs(r MatrixReport) []string {
	out := make([]string, 0, len(r.Issues))
	for _, i := range r.Issues {
		out = append(out, i.ID)
	}
	sort.Strings(out)
	return out
}

// TestExpandMatrixStampsNightlyProvenanceOnEveryCell covers the per-case
// provenance criterion: every expanded cell is a canonical case recording model,
// tokenizer, engine/backend, its deterministic oracle, the code revision, and the
// tolerance/baseline it is judged under — routed to an explicit tier with a
// declared cost.
func TestExpandMatrixStampsNightlyProvenanceOnEveryCell(t *testing.T) {
	spec := mtxSpec()
	expanded, rejected := ExpandMatrix(spec)

	if len(rejected) != 0 {
		t.Fatalf("well-formed spec rejected cells: %+v", rejected)
	}
	want := len(spec.Models) * len(spec.Backends) * len(spec.Modes) * len(spec.Slices)
	if len(expanded) != want {
		t.Fatalf("expanded %d cells, want %d (2 models x 3 backends x 2 modes x 2 slices)", len(expanded), want)
	}

	ids := map[string]bool{}
	for _, e := range expanded {
		if err := e.Case.ValidateCanonical(); err != nil {
			t.Fatalf("cell %s is not a canonical case: %v", e.Cell.CaseID, err)
		}
		if ids[e.Case.ID] {
			t.Fatalf("duplicate case id %s", e.Case.ID)
		}
		ids[e.Case.ID] = true
		m := e.Case.Metadata
		if m.Tier.Name != string(TierNightly) {
			t.Errorf("cell %s routed to tier %q, want nightly", e.Cell.CaseID, m.Tier.Name)
		}
		if m.Engine.Backend != e.Cell.Backend {
			t.Errorf("cell %s records backend %q, want %q", e.Cell.CaseID, m.Engine.Backend, e.Cell.Backend)
		}
		if got := m.Engine.Flags["mode"]; got != e.Cell.Mode {
			t.Errorf("cell %s records mode %q, want %q", e.Cell.CaseID, got, e.Cell.Mode)
		}
		if m.Model.Name != e.Cell.Model || m.Tokenizer.Revision == "" {
			t.Errorf("cell %s lost its model/tokenizer provenance: %+v", e.Cell.CaseID, m)
		}
		if m.Code.Revision != spec.Code.Revision || m.Tolerance.Revision != spec.Tolerance.Revision || m.Baseline.Revision != spec.Baseline.Revision {
			t.Errorf("cell %s lost its code/tolerance/baseline provenance: %+v", e.Cell.CaseID, m)
		}
		if m.Cost != spec.Cost {
			t.Errorf("cell %s lost its declared cost: %+v", e.Cell.CaseID, m.Cost)
		}
	}

	// The expansion is a pure function of the spec: same spec, same cells in the
	// same order, so a nightly replays cell-for-cell.
	again, _ := ExpandMatrix(spec)
	for i := range expanded {
		if again[i].Cell != expanded[i].Cell {
			t.Fatalf("expansion is not deterministic at index %d: %+v vs %+v", i, again[i].Cell, expanded[i].Cell)
		}
	}
}

// TestRunMatrixCleanRunIsGreenAndDocumentsCost is the passing half of the
// witness — and the "after the fix" state the failing half must return to. It
// also pins the runtime/resource cost the nightly tier is documented at.
func TestRunMatrixCleanRunIsGreenAndDocumentsCost(t *testing.T) {
	spec := mtxSpec()
	rep := RunMatrix(spec, mtxEngine(), nil)

	if green, why := rep.Green(); !green {
		t.Fatalf("clean matrix is not green: %s\n%s", why, ExplainMatrix(rep))
	}
	if len(rep.Cells) != 24 {
		t.Fatalf("ran %d cells, want 24", len(rep.Cells))
	}
	for _, c := range rep.Cells {
		if c.Outcome != MatrixPass {
			t.Fatalf("cell %s outcome %s, want pass: %s", c.Cell.CaseID, c.Outcome, c.Detail)
		}
	}
	if len(rep.Issues) != 0 {
		t.Fatalf("clean matrix filed issues: %+v", rep.Issues)
	}
	if rep.Suite.Tier != TierNightly {
		t.Fatalf("report suite tier %q, want nightly", rep.Suite.Tier)
	}
	// Cost is summed, not asserted: 24 cells at 30s runtime / 300s timeout each.
	if got, want := rep.Suite.TotalRuntimeSec, int64(24*30); got != want {
		t.Errorf("suite runtime %ds, want %ds", got, want)
	}
	if got, want := rep.Suite.TotalTimeoutSec, int64(24*300); got != want {
		t.Errorf("suite timeout %ds, want %ds", got, want)
	}
	if rep.Suite.MaxCPU != 4 || rep.Suite.MaxMemoryMiB != 2048 || rep.Suite.MaxAccelerators != 0 {
		t.Errorf("suite resource envelope not documented: %+v", rep.Suite)
	}
}

// TestRunMatrixInjectedRegressionFilesOneDeduplicatedIssue is the core #4577
// acceptance criterion. One planted backend regression lights up eight cells
// spanning both models, both modes, and both slices; the matrix must file ONE
// issue that names all eight, attribute it to backend=rocm, and back it with a
// scrubbed replay artifact pinning the first actionable divergence.
func TestRunMatrixInjectedRegressionFilesOneDeduplicatedIssue(t *testing.T) {
	rep := RunMatrix(mtxSpec(), mtxEngine(mtxDefectROCmKernel), nil)

	if green, _ := rep.Green(); green {
		t.Fatalf("matrix with a planted regression reported green:\n%s", ExplainMatrix(rep))
	}
	if len(rep.Issues) != 1 {
		t.Fatalf("planted ONE regression, got %d issue(s) %v — the fold did not deduplicate:\n%s",
			len(rep.Issues), mtxIssueIDs(rep), ExplainMatrix(rep))
	}
	issue := rep.Issues[0]
	if issue.Kind != "regression" {
		t.Errorf("issue kind %q, want regression", issue.Kind)
	}
	if len(issue.Cells) != 8 {
		t.Fatalf("issue covers %d cells, want all 8 rocm cells: %+v", len(issue.Cells), issue.Cells)
	}
	for _, c := range issue.Cells {
		if c.Backend != "rocm" {
			t.Errorf("issue swept in non-rocm cell %s", c.CaseID)
		}
	}
	// The matrix's whole informational advantage: eight failures narrow to one
	// coordinate, because every passing cell exonerates the others.
	if got := strings.Join(issue.Attribution, " "); got != "backend=rocm" {
		t.Errorf("attribution %q, want %q", got, "backend=rocm")
	}
	// Replay-backed: exactly one scrubbed artifact, carrying the embedded case and
	// the first actionable divergence.
	fb := issue.Replay
	if fb == nil {
		t.Fatal("issue carries no replay artifact")
	}
	if !fb.Scrubbed {
		t.Error("issue replay artifact is not scrubbed")
	}
	if fb.Case.ID != fb.CaseID || fb.Case.ID == "" {
		t.Errorf("replay artifact does not embed its own case: case_id=%q embedded=%q", fb.CaseID, fb.Case.ID)
	}
	if fb.FirstDivergence == nil {
		t.Fatal("issue replay artifact pins no first divergence")
	}
	if fb.FirstDivergence.Index != 1 || fb.FirstDivergence.Engine != "REGRESSED" {
		t.Errorf("first divergence %+v, want index 1 engine %q", *fb.FirstDivergence, "REGRESSED")
	}
	// The signature is read from the evidence, not from the cell's coordinates:
	// the drifted logit row at the divergent step is what names the stage.
	if issue.Signature.FailingOracle != "greedy-token-diff" || issue.Signature.Stage != "logits" {
		t.Errorf("signature %+v, want greedy-token-diff at stage logits", issue.Signature)
	}
	// Every non-rocm cell still passed, so the nightly localizes rather than
	// reporting a blanket red.
	pass := 0
	for _, c := range rep.Cells {
		if c.Outcome == MatrixPass {
			pass++
		}
	}
	if pass != 16 {
		t.Errorf("%d cells passed, want the 16 non-rocm cells", pass)
	}
}

// TestRunMatrixIssueReplaysFromItsArtifactAlone is the "independently replayed
// environment" half of the witness: the filed issue's bundle is re-run through
// runners built from the BUNDLE, with no access to the spec, the matrix, or the
// engine that produced it — and it must reproduce the same first divergence.
func TestRunMatrixIssueReplaysFromItsArtifactAlone(t *testing.T) {
	rep := RunMatrix(mtxSpec(), mtxEngine(mtxDefectROCmKernel), nil)
	if len(rep.Issues) != 1 || rep.Issues[0].Replay == nil {
		t.Fatalf("expected one replay-backed issue, got %+v", rep.Issues)
	}
	fb := *rep.Issues[0].Replay

	// Everything below reads ONLY fb.
	oracles, err := Lookup(fb.Case.Oracles)
	if err != nil {
		t.Fatalf("artifact names an unrunnable oracle: %v", err)
	}
	replayed, err := RunCase(fb.Case,
		ScriptedRunner{Label: "replay-reference", Trace: fb.Reference},
		ScriptedRunner{Label: "replay-engine", Trace: fb.Engine},
		oracles)
	if err != nil {
		t.Fatalf("artifact did not replay: %v", err)
	}
	if replayed.Pass {
		t.Fatal("replaying the artifact PASSED; the filed regression did not reproduce")
	}
	got := replayed.FailureBundle
	if got == nil {
		t.Fatal("replay produced no failure bundle")
	}
	if got.FailingOracle != fb.FailingOracle || got.FailingKind != fb.FailingKind {
		t.Errorf("replay blamed %s (%s), artifact recorded %s (%s)",
			got.FailingOracle, got.FailingKind, fb.FailingOracle, fb.FailingKind)
	}
	if got.FirstDivergence == nil || *got.FirstDivergence != *fb.FirstDivergence {
		t.Errorf("replay diverged at %+v, artifact recorded %+v", got.FirstDivergence, fb.FirstDivergence)
	}
}

// TestRunMatrixFixTurnsTheSameMatrixGreen closes the before/after loop on ONE
// spec: it fails with the defect present and passes once it is removed, so the
// gate is proven to be doing work rather than merely being green.
func TestRunMatrixFixTurnsTheSameMatrixGreen(t *testing.T) {
	spec := mtxSpec()

	broken := RunMatrix(spec, mtxEngine(mtxDefectROCmKernel), nil)
	if green, _ := broken.Green(); green {
		t.Fatal("matrix passed WITH the planted defect: the gate proves nothing")
	}
	fixed := RunMatrix(spec, mtxEngine(), nil)
	if green, why := fixed.Green(); !green {
		t.Fatalf("matrix still fails AFTER the fix: %s\n%s", why, ExplainMatrix(fixed))
	}
	if len(fixed.Issues) != 0 {
		t.Fatalf("fixed matrix still files issues: %v", mtxIssueIDs(fixed))
	}
}

// TestRunMatrixKeepsDistinctDefectsApart proves the fold deduplicates without
// over-merging: two planted defects of different classes stay two issues, each
// attributed to its own axis coordinate.
func TestRunMatrixKeepsDistinctDefectsApart(t *testing.T) {
	rep := RunMatrix(mtxSpec(), mtxEngine(mtxDefectROCmKernel, mtxDefectGraphStop), nil)

	if len(rep.Issues) != 2 {
		t.Fatalf("two distinct defects folded to %d issue(s) %v:\n%s",
			len(rep.Issues), mtxIssueIDs(rep), ExplainMatrix(rep))
	}
	byAttribution := map[string]MatrixIssue{}
	for _, i := range rep.Issues {
		byAttribution[strings.Join(i.Attribution, " ")] = i
	}
	rocm, ok := byAttribution["backend=rocm"]
	if !ok {
		t.Fatalf("no issue attributed to backend=rocm: %+v", byAttribution)
	}
	graph, ok := byAttribution["mode=graph"]
	if !ok {
		t.Fatalf("no issue attributed to mode=graph: %+v", byAttribution)
	}
	// The rocm kernel defect diverges at token 1, before the stop ever matters, so
	// the 4 rocm+graph cells fold under the deeper signature and the stop issue
	// covers only the 8 graph cells the rocm defect did not already claim
	// (2 models x cpu/cuda x 2 slices). Membership is asserted, not just the count:
	// a fold that over-merged would keep the tally and lose the partition.
	if len(rocm.Cells) != 8 {
		t.Errorf("rocm issue covers %d cells, want all 8 rocm cells", len(rocm.Cells))
	}
	for _, c := range rocm.Cells {
		if c.Backend != "rocm" {
			t.Errorf("rocm issue swept in cell %s, want only rocm cells", c.CaseID)
		}
	}
	if len(graph.Cells) != 8 {
		t.Errorf("graph issue covers %d cells, want the 8 graph cells that are not rocm", len(graph.Cells))
	}
	for _, c := range graph.Cells {
		if c.Mode != "graph" || c.Backend == "rocm" {
			t.Errorf("graph issue swept in cell %s, want only graph cells the rocm defect did not already claim", c.CaseID)
		}
	}
	if graph.Signature.Stage != "stops" {
		t.Errorf("graph issue stage %q, want stops", graph.Signature.Stage)
	}
	if rocm.Signature.key() == graph.Signature.key() {
		t.Error("two defect classes share one signature key")
	}
}

// TestRunMatrixMissingEvidenceIsNeverPass covers the third acceptance criterion
// from the other side: a backend whose build could not be resolved produces no
// evidence at all, and no evidence is not a pass.
func TestRunMatrixMissingEvidenceIsNeverPass(t *testing.T) {
	rep := RunMatrix(mtxSpec(), mtxEngine(mtxDefectROCmDown), nil)

	if green, _ := rep.Green(); green {
		t.Fatalf("matrix with 8 unrun cells reported green:\n%s", ExplainMatrix(rep))
	}
	if len(rep.Issues) != 1 {
		t.Fatalf("expected one deduplicated inconclusive issue, got %d: %v", len(rep.Issues), mtxIssueIDs(rep))
	}
	issue := rep.Issues[0]
	if issue.Kind != "inconclusive" {
		t.Errorf("issue kind %q, want inconclusive", issue.Kind)
	}
	if issue.Signature.Stage != causeEngineUnavailable {
		t.Errorf("issue stage %q, want %q", issue.Signature.Stage, causeEngineUnavailable)
	}
	if len(issue.Cells) != 8 {
		t.Errorf("issue covers %d cells, want all 8 rocm cells", len(issue.Cells))
	}
	if got := strings.Join(issue.Attribution, " "); got != "backend=rocm" {
		t.Errorf("attribution %q, want backend=rocm", got)
	}
	if issue.Replay != nil {
		t.Error("an absence of evidence must not carry a replay artifact")
	}
	for _, c := range rep.Cells {
		if c.Cell.Backend == "rocm" && c.Outcome != MatrixInconclusive {
			t.Errorf("unrun cell %s reported %s", c.Cell.CaseID, c.Outcome)
		}
	}
}

// TestRunMatrixRefusesCellsTooExpensiveForNightly covers the tier/cost criterion:
// a matrix whose per-cell cost cannot afford the nightly wall is REFUSED with the
// ceiling it broke, and a matrix that ran nothing is never green.
func TestRunMatrixRefusesCellsTooExpensiveForNightly(t *testing.T) {
	spec := mtxSpec()
	spec.Cost.TimeoutSeconds = 7200 // beyond the nightly budget of 3600s
	rep := RunMatrix(spec, mtxEngine(), nil)

	if len(rep.Cells) != 0 {
		t.Fatalf("ran %d cells that the nightly budget cannot afford", len(rep.Cells))
	}
	if len(rep.Rejected) != 24 {
		t.Fatalf("refused %d cells, want all 24", len(rep.Rejected))
	}
	if !strings.Contains(rep.Rejected[0].Reason, "exceeds tier nightly budget") {
		t.Errorf("refusal does not name the broken ceiling: %s", rep.Rejected[0].Reason)
	}
	green, why := rep.Green()
	if green {
		t.Fatal("a matrix that routed nothing reported green")
	}
	if !strings.Contains(why, "not routed") {
		t.Errorf("Green reason %q does not name the routing refusal", why)
	}
}

// TestRunMatrixRefusesAcceleratorCellsOnTheNightlyWall pins the other half of the
// nightly budget: accelerator qualification belongs to the release tier, so a
// matrix that asks the nightly lane for a GPU is refused rather than silently
// queued behind hardware it will never get.
func TestRunMatrixRefusesAcceleratorCellsOnTheNightlyWall(t *testing.T) {
	spec := mtxSpec()
	spec.Cost.Accelerators = 2
	rep := RunMatrix(spec, mtxEngine(), nil)

	if len(rep.Rejected) != 24 {
		t.Fatalf("refused %d cells, want all 24", len(rep.Rejected))
	}
	if !strings.Contains(rep.Rejected[0].Reason, "route to release") {
		t.Errorf("refusal does not route the cell to a tier that can afford it: %s", rep.Rejected[0].Reason)
	}
}

// TestExpandMatrixRefusesDuplicateCoordinates keeps the matrix from quietly
// running one cell twice and counting it as two greens.
func TestExpandMatrixRefusesDuplicateCoordinates(t *testing.T) {
	spec := mtxSpec()
	spec.Backends = []string{"cpu", "cpu"}
	expanded, rejected := ExpandMatrix(spec)

	if len(expanded) != 8 {
		t.Errorf("expanded %d cells, want 8 unique ones", len(expanded))
	}
	if len(rejected) != 8 {
		t.Fatalf("refused %d duplicate cells, want 8", len(rejected))
	}
	if !strings.Contains(rejected[0].Reason, "duplicate matrix coordinate") {
		t.Errorf("refusal reason %q does not name the duplicate", rejected[0].Reason)
	}
}

// TestExplainMatrixRendersOneIssueWithItsAttributionAndReplay is the operator
// readout witness: the captured render names the single issue, the coordinate it
// is attributed to, the replay handle, and the first divergence — the whole point
// being that an operator reads ONE finding, not eight.
func TestExplainMatrixRendersOneIssueWithItsAttributionAndReplay(t *testing.T) {
	rep := RunMatrix(mtxSpec(), mtxEngine(mtxDefectROCmKernel), nil)
	out := ExplainMatrix(rep)

	for _, want := range []string{
		"FAIL  nightly matrix — 24 cell(s): 16 pass, 8 fail, 0 inconclusive, 0 refused",
		"tier nightly cost: runtime=720s timeout=7200s cpu<=4 mem<=2048MiB accel<=0",
		"attributed to: backend=rocm",
		"replay: scrubbed bundle for case nightly-matrix/",
		`first divergence at token 1: reference "increased", engine "REGRESSED"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("readout is missing %q:\n%s", want, out)
		}
	}
	if got := strings.Count(out, "ISSUE "); got != 1 {
		t.Errorf("readout names %d issues, want exactly 1:\n%s", got, out)
	}
}
