package trajctl

import "fmt"

// benchprogress.go — issue #2547, a leaf of the trajectory-control epic (#2533):
// the benchmark progress-rate adapter.
//
// Benchmark harnesses (swebench, the LCB lanes) report only an end-state resolve
// bit, so an AgentBoard-style progress rate f(state, goal) in [0,1] exists
// nowhere and mid-run trajectory value is invisible in benchmark work. This
// adapter maps a benchmark task's ordered, harness-verified sub-goal state into a
// progress rate emitted as W3 ScoreRows at every phase boundary. Benchmarks are
// where scoring methods meet ground truth, so this is the lane the scorer
// calibration meter anchors on: the final row's value is pinned to the harness
// resolve bit, and a task whose resolve bit contradicts its sub-goal witnesses is
// refused rather than written — the ledger can never carry a progress curve that
// disagrees with ground truth.

const (
	// BenchScorerMethod is the stable method id of the benchmark progress-rate
	// adapter. It keys the registry and travels in every row it emits.
	BenchScorerMethod = "benchmark-progress-rate"
	// BenchScorerVersion is this implementation's version.
	BenchScorerVersion = "1"
	// evidenceKindHarness is the evidence kind for a benchmark-harness verdict,
	// distinct from the "commit" pointers the witnessed-commit scorer records.
	evidenceKindHarness = "harness"
)

// BenchSubGoal is one harness-verified sub-goal of a benchmark task, named in the
// order the harness reaches it (e.g. patch-applies → fail-to-pass → pass-to-pass).
// Verified is the benchmark harness verdict — W3, not a model call or a
// self-report: the sub-goal either passed the harness or it did not.
type BenchSubGoal struct {
	// ID is the stable sub-goal identifier, unique within a task.
	ID string
	// Title is an optional human label for the sub-goal.
	Title string
	// Verified is the harness verdict: true iff the harness confirmed this
	// sub-goal.
	Verified bool
	// Evidence optionally points at the harness row that backs the verdict. When
	// its Kind is unset, a "harness" pointer to the task id + sub-goal id is
	// synthesized so every credited sub-goal still traces to a re-resolvable
	// witness.
	Evidence EvidenceRef
}

// BenchTask is one benchmark instance's phase state: its ordered, harness-verified
// sub-goals plus the harness's end-state resolve bit. It is the AgentBoard-style
// f(state, goal) input the adapter folds into a progress curve.
type BenchTask struct {
	// ObjectiveID binds the emitted rows to a trajctl objective.
	ObjectiveID string
	// Benchmark names the harness this task ran under (e.g. "swebench",
	// "livecodebench").
	Benchmark string
	// TaskID is the harness's instance id (e.g. a swebench instance_id). It is
	// recorded on every row's RunID and used to synthesize evidence pointers, so
	// it is required.
	TaskID string
	// SubGoals are the task's sub-goals in the order the harness reaches them.
	SubGoals []BenchSubGoal
	// Resolved is the harness's end-state resolve bit. It must agree with the
	// sub-goal witnesses: a resolved task has every sub-goal verified, and an
	// unresolved task leaves at least one unverified (see ProgressRows).
	Resolved bool
}

// progressRate is the shared f(state, goal): the fraction of a task's sub-goals
// harness-verified so far. total==0 scores 0 rather than dividing by zero.
func progressRate(verified, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(verified) / float64(total)
}

// ProgressRows folds a benchmark task's ordered sub-goal state into cumulative
// progress-rate ScoreRows, one per phase boundary. Row i (1-indexed) carries the
// fraction of the task's sub-goals harness-verified within the first i boundaries
// over the total, so the curve rises only on a witnessed sub-goal and the final
// row's value is the task's overall harness progress rate. Every row is W3 and
// carries a re-resolvable evidence pointer for each credited sub-goal.
//
// The final row is pinned to the harness resolve bit: a task the harness marks
// resolved must have every sub-goal verified (final value 1.0), and an unresolved
// task must leave at least one sub-goal unverified (final value < 1). A BenchTask
// whose Resolved bit contradicts its sub-goal witnesses is a self-inconsistent
// harness report and is refused rather than emitted.
func (t BenchTask) ProgressRows(unixMillis int64) ([]ScoreRow, error) {
	if t.ObjectiveID == "" {
		return nil, fmt.Errorf("trajctl: bench task objective id is required")
	}
	if t.TaskID == "" {
		return nil, fmt.Errorf("trajctl: bench task id is required (it backs every row's evidence pointer)")
	}
	if len(t.SubGoals) == 0 {
		return nil, fmt.Errorf("trajctl: bench task %q has no sub-goals to score", t.TaskID)
	}
	seen := map[string]bool{}
	allVerified := true
	for _, sg := range t.SubGoals {
		if sg.ID == "" {
			return nil, fmt.Errorf("trajctl: bench task %q sub-goal id is required", t.TaskID)
		}
		if seen[sg.ID] {
			return nil, fmt.Errorf("trajctl: bench task %q duplicate sub-goal %q", t.TaskID, sg.ID)
		}
		seen[sg.ID] = true
		if !sg.Verified {
			allVerified = false
		}
	}
	if t.Resolved != allVerified {
		return nil, fmt.Errorf("trajctl: bench task %q resolve bit %t disagrees with sub-goal witnesses (all-verified=%t); the final progress row must agree with the harness resolve bit", t.TaskID, t.Resolved, allVerified)
	}

	n := len(t.SubGoals)
	rows := make([]ScoreRow, 0, n)
	verified := 0
	evidence := make([]EvidenceRef, 0, n)
	for _, sg := range t.SubGoals {
		if sg.Verified {
			verified++
			ref := sg.Evidence
			if ref.Kind == "" {
				ref = EvidenceRef{Kind: evidenceKindHarness, Ref: t.TaskID, Detail: sg.ID}
			}
			evidence = append(evidence, ref)
		}
		rows = append(rows, ScoreRow{
			ObjectiveID: t.ObjectiveID,
			Value:       progressRate(verified, n),
			Method:      BenchScorerMethod,
			Version:     BenchScorerVersion,
			Witness:     W3,
			Evidence:    append([]EvidenceRef(nil), evidence...),
			UnixMillis:  unixMillis,
			RunID:       t.TaskID,
		})
	}
	return rows, nil
}

// BenchProgressScorer folds an objective's plan phases into a single cumulative
// benchmark progress-rate row for the current window, resolving each phase's
// harness verdict through the injected resolver. It is the registry-facing
// (#2536) entry point a benchmark run path registers to score the live window;
// the full phase-boundary series over a completed task is produced by
// BenchTask.ProgressRows. Stateless and pure: the only host-dependent input is
// win.Resolve, so an un-resolvable window scores 0 rather than crediting
// unverified work (fail-closed).
type BenchProgressScorer struct{}

// Method implements Scorer.
func (BenchProgressScorer) Method() string { return BenchScorerMethod }

// Version implements Scorer.
func (BenchProgressScorer) Version() string { return BenchScorerVersion }

// Score returns one W3 progress row for obj: the fraction of its plan phases
// whose harness pointer resolves to EvidenceVerified. An objective with no plan
// has nothing to score against and yields no row.
func (BenchProgressScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	if len(obj.Plan) == 0 {
		return nil
	}
	verified := 0
	var evidence []EvidenceRef
	for _, phase := range obj.Plan {
		ref := EvidenceRef{Kind: evidenceKindHarness, Ref: obj.ID, Detail: phase.ID}
		if win.Resolve == nil || win.Resolve(ref) != EvidenceVerified {
			continue
		}
		verified++
		evidence = append(evidence, ref)
	}
	return []ScoreRow{{
		ObjectiveID: obj.ID,
		Value:       progressRate(verified, len(obj.Plan)),
		Method:      BenchScorerMethod,
		Version:     BenchScorerVersion,
		Witness:     W3,
		Evidence:    evidence,
		UnixMillis:  win.UnixMillis,
	}}
}
