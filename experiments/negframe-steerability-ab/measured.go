package main

// measured.go — the #5851 consumer wiring: the A/B's two arms are now PRODUCED BY THE #3568
// LEVER over the real injected stream, not read out of the hand-authored fixture corpus.
//
// #3568 shipped the lever (`FAK_ABLATE=negframe_reframe` routes fak-authored SessionStart prose
// UNREFRAMED; an unset env REFRAMES it) and the per-turn signal (one journal row per turn naming
// the ACTIVE ARM plus applied/residual/verbatim_fallback). Its last refinement said #3546's A/B
// should consume that lever "replacing hand-swapped strings with one env toggle". This file is
// that consumer.
//
// How it measures, and why this shape:
//
//   - The lever lives in `package main` at cmd/fak (guard_negframe_summary.go), so it cannot be
//     imported. The measured path is therefore the REAL BINARY, run twice over the same source
//     prose with the ONLY difference being the arm env. That is stronger than an import would
//     be: it exercises the shipped env parsing, the shipped prose, and the shipped journal
//     writer end to end.
//   - Each arm runs in its OWN scratch workspace, because the journal path is workspace-relative
//     and SessionStart TRUNCATES it (guardNegframeBegin). Separate workspaces keep the two arms'
//     rows from overwriting each other.
//   - Every other FAK_* knob that perturbs the composed prose is STRIPPED from both arms'
//     environments, so the framing lever is the single varying factor the A/B isolates.
//
// What is MEASURED here and what stays MODELED: the arm labels and the negation counts come off
// disk from a real run (measured). The compliance rate they are mapped to is still the same
// stated linear cost model as the fixture half (modeled) — there is no live agent in this
// environment, so a compliance RATE remains a prediction. The report labels both.

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// measuredJournalRel mirrors cmd/fak's guardNegframeJournalRel: the workspace-relative per-turn
// negframe row stream #3568 writes. Duplicated as a constant rather than imported because the
// producer is a `package main`; the DoD names this exact path, and the test reads the file back
// off disk so a drift shows up as a hard failure, not a silent skip.
const measuredJournalRel = ".fak/negframe/journal.jsonl"

// The arm labels #3568 writes into the row. Not aliases of anything importable — pinned here and
// asserted against a real run, which is what makes a drift visible.
const (
	measuredArmOff = "reframe_off" // control: FAK_ABLATE=negframe_reframe
	measuredArmOn  = "reframe_on"  // treatment: env unset
)

// measuredAblateEnv/measuredAblateToken are the coarse human-typed lever; measuredFeatureEnv is
// the canonical per-feature env an `fak ablate --sweep` child carries. Both are cleared on the
// treatment arm so an inherited value from the surrounding session cannot flip it.
const (
	measuredAblateEnv   = "FAK_ABLATE"
	measuredAblateToken = "negframe_reframe"
	measuredFeatureEnv  = "FAK_ABLATE_NEGFRAME_REFRAME"
)

// measuredArgv is the measured path: the SessionStart hook that injects fak-authored prose, in
// its --managed shape (the headless/fleet posture, which composes the richest fak-authored
// stream: the MCP affordance hint + the long-horizon rule + the tool-width hint).
var measuredArgv = []string{"guard-sessionstart", "--managed"}

// measuredScrubEnv are the knobs that would change the COMPOSED PROSE rather than the framing.
// Clearing them on both arms is what makes this a controlled comparison: same source text, one
// lever. CLAUDE_CODE_SESSION_ID is cleared too, so a measured run cannot append a stray identity
// row to the host's durable resume store.
var measuredScrubEnv = []string{
	measuredAblateEnv,
	measuredFeatureEnv,
	"FAK_GUARD_AFFORDANCE_MODE", // "off" suppresses the injection entirely (and the journal row)
	"FAK_TOOL_WIDTH_HINT",       // "off" drops one paragraph from the composed prose
	"CLAUDE_CODE_SESSION_ID",
}

// negframeJournalRow is the on-disk row #3568 writes, field-for-field.
type negframeJournalRow struct {
	Arm              string `json:"arm"`
	Applied          int    `json:"applied"`
	Residual         int    `json:"residual"`
	VerbatimFallback int    `json:"verbatim_fallback"`
}

// MeasuredArm is one arm of the measured A/B: the env that selected it, the journal row the run
// wrote, and the modeled compliance that row's negation load maps to.
type MeasuredArm struct {
	Label            string  `json:"label"` // "control" | "treatment"
	Env              string  `json:"env"`   // the env difference that selected this arm
	Arm              string  `json:"arm_from_journal"`
	Applied          int     `json:"applied"`
	Residual         int     `json:"residual"`
	VerbatimFallback int     `json:"verbatim_fallback"`
	Mechanical       int     `json:"modeled_mechanical_load"`
	Judgement        int     `json:"modeled_judgement_load"`
	Compliance       float64 `json:"modeled_compliance_from_measured_load"`
	JournalPath      string  `json:"journal_path"`
}

// MeasuredAB is the measured half of the report: two arms of the same injected stream, separated
// by one env toggle, with their arm labels read back off the per-turn journal.
type MeasuredAB struct {
	Schema       string      `json:"schema"`
	Provenance   string      `json:"provenance"`
	Binary       string      `json:"binary"`
	BinarySource string      `json:"binary_source"`
	Argv         []string    `json:"argv"`
	Control      MeasuredArm `json:"control_arm"`
	Treatment    MeasuredArm `json:"treatment_arm"`
	// ResidualDelta is the MEASURED quantity: post-pass negatives on the treatment arm minus
	// those the control arm ships raw. Negative means the reframe removed negations.
	ResidualDelta int `json:"measured_residual_delta_treatment_minus_control"`
	// ModeledDelta maps each arm's MEASURED negation load through the same stated cost model the
	// fixture half uses. Still modeled — only the load underneath it is measured.
	ModeledDelta float64 `json:"modeled_compliance_delta_treatment_minus_control"`
	Note         string  `json:"note"`
}

const measuredProvenance = "MEASURED ARM LABELS + NEGATION LOAD (read from a real run's " +
	measuredJournalRel + "); the compliance RATE they map to is still MODELED."

// measuredLoad splits an arm's journal row into the (mechanical, judgement) counts the shared
// complianceProxy takes.
//
// The row does not carry the tier split, so it is reconstructed from what the lever itself
// proves. Both arms run the SAME source text, so the treatment arm's `applied` count IS the
// number of mechanical-tier idioms that text carries (a mechanical finding is exactly one with a
// confident positive rewrite, which is what the pass flips). Therefore:
//
//   - control (reframe off): nothing was flipped, so its `residual` is the whole negation load;
//     `applied` mechanical idioms of it are mechanical, the rest judgement-tier.
//   - treatment (reframe on): the mechanical idioms are gone by construction, so everything left
//     in `residual` is scored at the judgement tier.
//
// The mechanical count is clamped to the arm's own residual so a degraded run (fallbacks
// inflating `applied` past what the raw text carries) cannot manufacture a negative judgement
// count and with it a fictitious delta.
func measuredLoad(row negframeJournalRow, treatmentApplied int) (mechanical, judgement int) {
	if row.Arm == measuredArmOn {
		return 0, row.Residual
	}
	mechanical = treatmentApplied
	if mechanical > row.Residual {
		mechanical = row.Residual
	}
	return mechanical, row.Residual - mechanical
}

// runMeasuredAB runs the measured path twice under root — control with the lever set, treatment
// with it unset — and folds the two journal rows into the measured half of the report.
//
// Nothing in this function reads `fixtures`: every number it returns came off disk from a
// subprocess run. That is the property #5851 asks for, and the property measured_test.go pins.
func runMeasuredAB(binary, binarySource, root string) (*MeasuredAB, error) {
	controlRow, controlJournal, err := runMeasuredArm(binary, filepath.Join(root, "control"), true)
	if err != nil {
		return nil, fmt.Errorf("control arm: %w", err)
	}
	treatmentRow, treatmentJournal, err := runMeasuredArm(binary, filepath.Join(root, "treatment"), false)
	if err != nil {
		return nil, fmt.Errorf("treatment arm: %w", err)
	}

	controlMech, controlJudge := measuredLoad(controlRow, treatmentRow.Applied)
	treatmentMech, treatmentJudge := measuredLoad(treatmentRow, treatmentRow.Applied)
	control := MeasuredArm{
		Label:            "control",
		Env:              measuredAblateEnv + "=" + measuredAblateToken,
		Arm:              controlRow.Arm,
		Applied:          controlRow.Applied,
		Residual:         controlRow.Residual,
		VerbatimFallback: controlRow.VerbatimFallback,
		Mechanical:       controlMech,
		Judgement:        controlJudge,
		Compliance:       complianceProxy(controlMech, controlJudge),
		JournalPath:      controlJournal,
	}
	treatment := MeasuredArm{
		Label:            "treatment",
		Env:              "(" + measuredAblateEnv + " unset)",
		Arm:              treatmentRow.Arm,
		Applied:          treatmentRow.Applied,
		Residual:         treatmentRow.Residual,
		VerbatimFallback: treatmentRow.VerbatimFallback,
		Mechanical:       treatmentMech,
		Judgement:        treatmentJudge,
		Compliance:       complianceProxy(treatmentMech, treatmentJudge),
		JournalPath:      treatmentJournal,
	}
	return &MeasuredAB{
		Schema:        "fak-negframe-steerability-ab-measured/1",
		Provenance:    measuredProvenance,
		Binary:        binary,
		BinarySource:  binarySource,
		Argv:          measuredArgv,
		Control:       control,
		Treatment:     treatment,
		ResidualDelta: treatment.Residual - control.Residual,
		ModeledDelta:  treatment.Compliance - control.Compliance,
		Note:          measuredNote(control, treatment),
	}, nil
}

// measuredNote states the measured finding in one sentence, including the honest zero-delta case:
// a stream whose only negation is judgement-tier gives the lever nothing to flip, so the two arms
// emit identical prose and differ only in the label the journal records.
func measuredNote(control, treatment MeasuredArm) string {
	switch {
	case treatment.Applied == 0 && control.Residual == treatment.Residual:
		return "MEASURED zero delta: the injected stream carries no mechanical-tier negation, so the " +
			"reframe pass had nothing to flip and both arms emit identical prose. The arms differ only " +
			"in the journal's arm label. Its " + fmt.Sprintf("%d", control.Residual) +
			" residual negative(s) are judgement-tier, which negframe deliberately does not auto-rewrite — " +
			"so lifting this stream further is a prose edit, not a lever setting."
	case treatment.VerbatimFallback > treatment.Applied:
		return "MEASURED degradation: the treatment arm refused more reframe candidates (verbatim " +
			"fallbacks) than it applied, so its prose ships largely unreframed while still labelled " +
			"reframe_on. Treat this run's delta as unreliable until the gate is fixed."
	default:
		return fmt.Sprintf("MEASURED: the reframe pass flipped %d mechanical idiom(s), taking the injected "+
			"stream from %d residual negative(s) on the control arm to %d on the treatment arm.",
			treatment.Applied, control.Residual, treatment.Residual)
	}
}

// runMeasuredArm executes one arm of the measured path in its own scratch workspace and returns
// the journal row that run wrote, plus the path it was read from.
func runMeasuredArm(binary, workspace string, ablate bool) (negframeJournalRow, string, error) {
	var row negframeJournalRow
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		return row, "", fmt.Errorf("make workspace: %w", err)
	}
	cmd := exec.Command(binary, measuredArgv...)
	cmd.Dir = workspace
	cmd.Env = measuredEnv(ablate)
	// A nil Stdin gives the child an empty stdin, so the hook's compact look-ahead pickup reads
	// EOF and composes to the plain affordance on both arms.
	var stderr strings.Builder
	cmd.Stdout = nil
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return row, "", fmt.Errorf("run %s %s: %w\n%s", binary, strings.Join(measuredArgv, " "), err, stderr.String())
	}
	journal := filepath.Join(workspace, filepath.FromSlash(measuredJournalRel))
	row, err := readLastJournalRow(journal)
	if err != nil {
		return row, journal, err
	}
	return row, journal, nil
}

// measuredEnv builds one arm's environment: the host environment with every prose-perturbing FAK
// knob stripped, plus the lever on the control arm only.
func measuredEnv(ablate bool) []string {
	drop := make(map[string]bool, len(measuredScrubEnv))
	for _, k := range measuredScrubEnv {
		drop[k] = true
	}
	out := make([]string, 0, len(os.Environ())+1)
	for _, kv := range os.Environ() {
		name, _, ok := strings.Cut(kv, "=")
		if ok && drop[name] {
			continue
		}
		out = append(out, kv)
	}
	if ablate {
		out = append(out, measuredAblateEnv+"="+measuredAblateToken)
	}
	return out
}

// readLastJournalRow reads the LAST parseable row of a negframe journal — the same "last row
// wins" rule guardNegframeSummaryLine uses to name the arm a process actually ran.
func readLastJournalRow(path string) (negframeJournalRow, error) {
	var last negframeJournalRow
	f, err := os.Open(path)
	if err != nil {
		return last, fmt.Errorf("read %s: %w (a pre-#3568 binary writes no negframe journal)", path, err)
	}
	defer f.Close()
	found := false
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row negframeJournalRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		last, found = row, true
	}
	if err := sc.Err(); err != nil {
		return last, fmt.Errorf("scan %s: %w", path, err)
	}
	if !found {
		return last, fmt.Errorf("no parseable negframe row in %s", path)
	}
	return last, nil
}

// resolveFakBinary honors an explicit FAK_BIN override or builds the checked-out source into
// the caller-provided scratch directory. It deliberately does not reuse a module-root or PATH
// binary: either can predate #3568 and would measure unrelated installed state.
func resolveFakBinary(buildDir string) (path, source string, err error) {
	exeName := "fak"
	if runtime.GOOS == "windows" {
		exeName += ".exe"
	}
	if override := strings.TrimSpace(os.Getenv("FAK_BIN")); override != "" {
		if st, statErr := os.Stat(override); statErr == nil && !st.IsDir() {
			return override, "FAK_BIN override", nil
		}
		return "", "", fmt.Errorf("FAK_BIN=%q is not a runnable file", override)
	}
	root, rootErr := moduleRoot()
	if rootErr != nil {
		return "", "", fmt.Errorf("locate module root to build ./cmd/fak: %w", rootErr)
	}
	if err := os.MkdirAll(buildDir, 0o755); err != nil {
		return "", "", fmt.Errorf("create isolated fak build directory: %w", err)
	}
	out := filepath.Join(buildDir, exeName)
	build := windowgate.Command("go", "build", "-o", out, "./cmd/fak")
	build.Dir = root
	if combined, buildErr := build.CombinedOutput(); buildErr != nil {
		return "", "", fmt.Errorf("build ./cmd/fak: %w\n%s\n(set FAK_BIN=<path to a fak binary> to measure "+
			"without building — a shared tree with a peer's half-saved file will not build)", buildErr, combined)
	}
	return out, "isolated go build ./cmd/fak", nil
}

// moduleRoot walks up from the working directory to the directory holding go.mod. The Go module
// IS the repository root in this repo, so that is also the clone root the measured build runs in.
func moduleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if st, statErr := os.Stat(filepath.Join(dir, "go.mod")); statErr == nil && !st.IsDir() {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("no go.mod found walking up from the working directory")
		}
		dir = parent
	}
}
