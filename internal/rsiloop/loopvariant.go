package rsiloop

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

const (
	// LoopVariantArchiveSchema tags kept-variant archive rows. The archive is JSONL
	// so each KEEP becomes one append-only stepping-stone record.
	LoopVariantArchiveSchema = "fak-loopvariant-archive/1"
	// LoopVariantMetricName is the shipgate metric for the metaloop objective. It is
	// a fold over the task-set spec/oracle, not a raw efficiency metric.
	LoopVariantMetricName = "spec_oracle_points"
)

// VariantKind names the loop surface a metaloop candidate changes.
type VariantKind string

const (
	VariantPrompt          VariantKind = "prompt"
	VariantToolPolicy      VariantKind = "tool_policy"
	VariantIterationPolicy VariantKind = "iteration_policy"
)

// LoopVariant is one proposed loop/prompt/tool-policy diff over a baseline loop
// config. Diff is deliberately opaque to this package; the driver owns how it is
// applied in an isolated worktree, while this package owns the keep/revert fold.
type LoopVariant struct {
	ID          string      `json:"id"`
	Kind        VariantKind `json:"kind"`
	Description string      `json:"description,omitempty"`
	Diff        string      `json:"diff,omitempty"`
}

// LoopConfig is the baseline loop surface a metaloop proposer may change. The
// fields are deliberately textual: applying a prompt/tool/iteration-policy diff is
// driver-owned, while this package owns the evidence gate after the driver measures
// the variant.
type LoopConfig struct {
	Prompt          string `json:"prompt,omitempty"`
	ToolPolicy      string `json:"tool_policy,omitempty"`
	IterationPolicy string `json:"iteration_policy,omitempty"`
}

// LoopVariantProposer emits candidate diffs over a baseline loop config. The
// archive argument lets a DGM/SICA-style proposer reuse kept stepping stones without
// trusting them: every new candidate still has to pass EvaluateLoopVariant.
type LoopVariantProposer interface {
	ProposeLoopVariants(baseline LoopConfig, archive []LoopVariantRecord) ([]LoopVariant, error)
}

// StaticLoopVariantProposer is the deterministic proposer used by fixtures and
// simple drivers that already generated candidate diffs elsewhere.
type StaticLoopVariantProposer []LoopVariant

// ProposeLoopVariants returns a defensive copy of the static candidates.
func (p StaticLoopVariantProposer) ProposeLoopVariants(LoopConfig, []LoopVariantRecord) ([]LoopVariant, error) {
	return append([]LoopVariant(nil), p...), nil
}

// DOSEvidence is the non-author read-back attached to a task result. In the live
// driver this is the `dos verify` / `dos improve --observe` receipt for the task.
type DOSEvidence struct {
	ID        string `json:"id"`
	Subject   string `json:"subject,omitempty"`
	Confirmed bool   `json:"confirmed"`
	Summary   string `json:"summary,omitempty"`
}

// TaskWitness is the re-measured outcome of one fixed task under one loop config.
// SpecPassed is the oracle verdict. SpecPoints is credited only when SpecPassed is
// true, so raw metric wins (Turns, LatencyMS, etc.) cannot by themselves move KEEP.
type TaskWitness struct {
	TaskID     string      `json:"task_id"`
	SpecPassed bool        `json:"spec_passed"`
	SpecPoints float64     `json:"spec_points,omitempty"`
	Turns      int         `json:"turns,omitempty"`
	LatencyMS  int         `json:"latency_ms,omitempty"`
	Evidence   DOSEvidence `json:"dos_evidence"`
	Note       string      `json:"note,omitempty"`
}

// TaskSetWitness is the fixed task-set read-back for one baseline or candidate run.
type TaskSetWitness struct {
	Ref   string        `json:"ref,omitempty"`
	Tasks []TaskWitness `json:"tasks"`
}

// SpecFold is the objective fold used for the keep-bit. Raw metrics stay outside
// this structure by design; a shorter run that fails the oracle has zero points.
type SpecFold struct {
	Tasks       int      `json:"tasks"`
	Passed      int      `json:"passed"`
	Points      float64  `json:"points"`
	SpecGreen   bool     `json:"spec_green"`
	TruthClean  bool     `json:"truth_clean"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

// LoopVariantRecord is the durable archive row for a kept stepping-stone variant,
// and also the in-memory verdict returned for a reverted candidate.
type LoopVariantRecord struct {
	Schema       string           `json:"schema"`
	Variant      LoopVariant      `json:"variant"`
	Decision     string           `json:"decision"`
	Kept         bool             `json:"kept"`
	MetricName   string           `json:"metric_name"`
	Before       SpecFold         `json:"before"`
	After        SpecFold         `json:"after"`
	Score        *Scorecard       `json:"score,omitempty"`
	Witness      shipgate.Witness `json:"witness"`
	DOSEvidence  []DOSEvidence    `json:"dos_evidence"`
	Tasks        []TaskWitness    `json:"tasks"`
	BaselineRef  string           `json:"baseline_ref,omitempty"`
	CandidateRef string           `json:"candidate_ref,omitempty"`
}

// FoldTaskSet folds one task-set witness into the spec/oracle objective. Passing
// tasks with no explicit SpecPoints count as one point; failed tasks count as zero.
func FoldTaskSet(ts TaskSetWitness) SpecFold {
	f := SpecFold{Tasks: len(ts.Tasks), SpecGreen: len(ts.Tasks) > 0, TruthClean: len(ts.Tasks) > 0}
	for _, t := range ts.Tasks {
		if !t.SpecPassed {
			f.SpecGreen = false
		} else {
			f.Passed++
			if t.SpecPoints > 0 {
				f.Points += t.SpecPoints
			} else {
				f.Points++
			}
		}
		if !t.Evidence.Confirmed || t.Evidence.ID == "" {
			f.TruthClean = false
			continue
		}
		f.EvidenceIDs = append(f.EvidenceIDs, t.Evidence.ID)
	}
	return f
}

// EvaluateLoopVariant folds a variant through the existing shipgate keep-bit. The
// baseline and candidate must be the SAME ordered task set; a mismatch is a driver
// bug, because #1177's delta is over a fixed task set.
func EvaluateLoopVariant(v LoopVariant, baseline, candidate TaskSetWitness) (LoopVariantRecord, error) {
	if err := sameTaskSet(baseline, candidate); err != nil {
		return LoopVariantRecord{}, err
	}
	before := FoldTaskSet(baseline)
	after := FoldTaskSet(candidate)
	w := shipgate.Witness{
		Metric:      LoopVariantMetricName,
		Before:      before.Points,
		After:       after.Points,
		LowerBetter: false,
		SuiteGreen:  after.SpecGreen,
		TruthClean:  before.TruthClean && after.TruthClean,
	}
	decision, witness := shipgate.Evaluate(w)
	rec := LoopVariantRecord{
		Schema:       LoopVariantArchiveSchema,
		Variant:      v,
		Decision:     decision.String(),
		Kept:         witness.Kept(),
		MetricName:   LoopVariantMetricName,
		Before:       before,
		After:        after,
		Score:        loopVariantScorecard(before, after),
		Witness:      witness,
		DOSEvidence:  taskSetEvidence(baseline, candidate),
		Tasks:        append([]TaskWitness(nil), candidate.Tasks...),
		BaselineRef:  baseline.Ref,
		CandidateRef: candidate.Ref,
	}
	return rec, nil
}

func loopVariantScorecard(before, after SpecFold) *Scorecard {
	specGreen := 0.0
	if after.SpecGreen {
		specGreen = 1
	}
	truthClean := 0.0
	if before.TruthClean && after.TruthClean {
		truthClean = 1
	}
	return &Scorecard{
		Name:  LoopVariantMetricName,
		Value: after.Points,
		Grade: loopVariantScoreGrade(before, after),
		Components: []ScoreComponent{
			{Name: "before_points", Value: before.Points, Unit: "points"},
			{Name: "after_points", Value: after.Points, Unit: "points"},
			{Name: "point_delta", Value: after.Points - before.Points, Unit: "points"},
			{Name: "before_passed", Value: float64(before.Passed), Unit: "tasks"},
			{Name: "after_passed", Value: float64(after.Passed), Unit: "tasks"},
			{Name: "tasks", Value: float64(after.Tasks), Unit: "tasks"},
			{Name: "spec_green", Value: specGreen, Unit: "bool"},
			{Name: "truth_clean", Value: truthClean, Unit: "bool"},
			{Name: "evidence_ids", Value: float64(len(after.EvidenceIDs)), Unit: "ids"},
		},
	}
}

func loopVariantScoreGrade(before, after SpecFold) string {
	switch {
	case !before.TruthClean || !after.TruthClean:
		return "truth-dirty"
	case !after.SpecGreen:
		return "spec-red"
	case after.Points <= before.Points:
		return "no-gain"
	case after.Passed == after.Tasks:
		return "complete"
	default:
		return "improved"
	}
}

// ArchiveLoopVariant appends a record only when shipgate kept the variant. A REVERT
// is returned to the caller but never becomes a stepping stone.
func ArchiveLoopVariant(v LoopVariant, baseline, candidate TaskSetWitness, a *LoopVariantArchive) (LoopVariantRecord, error) {
	rec, err := EvaluateLoopVariant(v, baseline, candidate)
	if err != nil {
		return LoopVariantRecord{}, err
	}
	if rec.Kept && a != nil {
		if err := a.Append(rec); err != nil {
			return rec, err
		}
	}
	return rec, nil
}

// LoopVariantArchive is an append-only JSONL sink for kept stepping-stone variants.
type LoopVariantArchive struct {
	w      io.WriteCloser
	closer bool
}

// NewLoopVariantArchive opens a kept-variant archive at path. A path of "-" or ""
// writes JSONL to stdout and is not closed.
func NewLoopVariantArchive(path string) (*LoopVariantArchive, error) {
	if path == "" || path == "-" {
		return &LoopVariantArchive{w: nopWriteCloser{os.Stdout}}, nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &LoopVariantArchive{w: f, closer: true}, nil
}

// Append writes one kept variant as a single JSON line. The archive is a stepping-
// stone ledger, so a REVERT record is never admitted even if a caller bypasses
// ArchiveLoopVariant.
func (a *LoopVariantArchive) Append(r LoopVariantRecord) error {
	if !r.Kept || r.Decision != shipgate.KEEP.String() || !r.Witness.Kept() {
		return fmt.Errorf("loopvariant archive admits only kept variants, got decision=%s kept=%v", r.Decision, r.Kept)
	}
	if err := validateArchiveEvidence(r); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = a.w.Write(b)
	return err
}

// Close releases the archive file. Stdout archives are left open.
func (a *LoopVariantArchive) Close() error {
	if a.closer {
		return a.w.Close()
	}
	return nil
}

// ReadLoopVariantArchive loads archive records, skipping torn/non-JSON lines so an
// interrupted append does not hide valid stepping stones already written.
func ReadLoopVariantArchive(path string) ([]LoopVariantRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var rows []LoopVariantRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r LoopVariantRecord
		if err := json.Unmarshal(line, &r); err != nil {
			continue
		}
		rows = append(rows, r)
	}
	return rows, nil
}

func sameTaskSet(a, b TaskSetWitness) error {
	if len(a.Tasks) != len(b.Tasks) {
		return fmt.Errorf("task-set mismatch: baseline has %d task(s), candidate has %d", len(a.Tasks), len(b.Tasks))
	}
	for i := range a.Tasks {
		if a.Tasks[i].TaskID == "" {
			return fmt.Errorf("baseline task %d has empty task_id", i)
		}
		if b.Tasks[i].TaskID == "" {
			return fmt.Errorf("candidate task %d has empty task_id", i)
		}
		if a.Tasks[i].TaskID != b.Tasks[i].TaskID {
			return fmt.Errorf("task-set mismatch at %d: baseline %q candidate %q", i, a.Tasks[i].TaskID, b.Tasks[i].TaskID)
		}
	}
	return nil
}

func taskSetEvidence(taskSets ...TaskSetWitness) []DOSEvidence {
	var n int
	for _, ts := range taskSets {
		n += len(ts.Tasks)
	}
	out := make([]DOSEvidence, 0, n)
	for _, ts := range taskSets {
		for _, t := range ts.Tasks {
			out = append(out, t.Evidence)
		}
	}
	return out
}

func validateArchiveEvidence(r LoopVariantRecord) error {
	if len(r.DOSEvidence) == 0 {
		return fmt.Errorf("loopvariant archive requires dos evidence for kept variant %q", r.Variant.ID)
	}
	for i, ev := range r.DOSEvidence {
		if ev.ID == "" || !ev.Confirmed {
			return fmt.Errorf("loopvariant archive evidence %d for kept variant %q is not confirmed", i, r.Variant.ID)
		}
	}
	return nil
}
