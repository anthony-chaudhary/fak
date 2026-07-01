package frontierswe

// This file is the describe-surface projection layer over geometry.go's C4
// TTSModel: it turns a loaded Task (task.go) into a deterministic, model-free
// time-to-solution PROJECTION — the per-task value-stack floor `fak frontierswe
// describe --tts` prints. geometry.go owns the arithmetic (A/B/C arms, the reuse
// curve); this file only CONSUMES it, deriving the per-turn geometry a task
// implies from its committed facts and folding in the cross-trial reuse arm that
// FrontierSWE's n_concurrent_trials adds. Nothing here edits geometry.go or
// task.go; it is a pure overlay so the two value-stack floors (swebench's
// AggregatePrefill worker sweep and this one) are discovered and used the same way.
//
// WITNESSED vs PROJECTION (same honesty boundary geometry.go states).
//   - WITNESSED-by-construction: every arithmetic step below — the A/B work
//     integrals, the C(r) reuse interpolation, and the cross-trial shared-prefix
//     arm. They are exact over the derived geometry and reproduced bit-for-bit by
//     the test.
//   - PROJECTION: the DERIVED geometry itself (turns/decode/result inferred from
//     the task's budget, not measured from a live trajectory) and the realized
//     reuse rate r. Both are floors/dials here; pinning them to a measured run is
//     the C8/C14 measurement. Until then every number this file emits is labeled a
//     deterministic floor, never a measurement.

// Default per-turn token shape for the FrontierSWE long-horizon regime. These are
// honest order-of-magnitude constants for a coding agent's trajectory (the same
// spirit as swebench.DefaultGeometryModel's BasePrefix/Decode/Result), NOT a
// measurement — they fix the SHAPE of the re-prefill integral so the projection is
// deterministic and reproducible. The realized geometry of a live run is the
// C8/C14 measurement that replaces these.
const (
	// defaultPrefixTokens is the initial resident context a FrontierSWE trajectory
	// carries: system prompt + tool schemas + the repo/instruction snapshot.
	defaultPrefixTokens = 8000
	// defaultDecodeTokens is the assistant tokens decoded per turn.
	defaultDecodeTokens = 64
	// defaultResultTokens is the tool-result tokens ingested per turn.
	defaultResultTokens = 32
	// turnsPerHour projects the long-horizon turn count from the agent's
	// wall-clock budget: a deterministic ~100 model round-trips per budgeted hour.
	// It is the single dial that maps the one fact every task carries (its [agent]
	// timeout_sec budget) onto the trajectory length the re-prefill integral runs
	// over. It is a PROJECTION constant, replaced by the measured turn count in C14.
	turnsPerHour = 100
	// defaultConcurrentTrials is the cross-trial fan-out used when job.yaml does not
	// declare n_concurrent_trials (FrontierSWE runs several concurrent trials of the
	// same task that share the identical prefix). A single trial means no cross-trial
	// reuse arm; the default keeps the sweep meaningful for fixtures with no job.yaml.
	defaultConcurrentTrials = 1
)

// DefaultReuseRate is the sensible default cross-turn reuse rate the describe
// surface projects at when --reuse is not given: a conservative-but-real 0.85,
// the value-stack reuse rate the epic's value story is told at. It is a PROJECTION
// dial (the realized rate is the C8/C14 measurement), exposed as a named constant
// so the CLI default and the test agree on one number.
const DefaultReuseRate = 0.85

// GeometryForTask derives the deterministic long-horizon TaskGeometry a task
// implies from its committed facts — the per-turn shape geometry.go's TTSModel
// consumes. Turns scale with the task's [agent] timeout_sec budget (turnsPerHour
// round-trips per budgeted hour); the per-turn token shape is the default regime
// constants. A task with no agent budget falls back to a single-turn floor so the
// arithmetic stays defined. This is a PROJECTION (the turn count is inferred, not
// measured); the geometry's Name carries the task name for the describe row.
func GeometryForTask(t *Task) TaskGeometry {
	turns := ProjectedTurns(t.AgentTimeoutSec())
	name := ""
	if t != nil {
		name = t.Name
	}
	return TaskGeometry{
		Name:   name,
		Prefix: defaultPrefixTokens,
		Turns:  turns,
		Decode: defaultDecodeTokens,
		Result: defaultResultTokens,
	}
}

// ProjectedTurns maps an agent wall-clock budget (seconds) onto the projected
// long-horizon turn count: turnsPerHour round-trips per budgeted hour, floored at
// 1 so a budget-carrying task always has a defined trajectory. Pure arithmetic.
func ProjectedTurns(agentTimeoutSec float64) int {
	if agentTimeoutSec <= 0 {
		return 1
	}
	turns := int((agentTimeoutSec / 3600.0) * float64(turnsPerHour))
	if turns < 1 {
		turns = 1
	}
	return turns
}

// TrialsForTask is the cross-trial fan-out for a task: job.yaml's
// n_concurrent_trials when declared (concurrent trials of the same task share the
// identical prefix), else defaultConcurrentTrials. Floored at 1.
func TrialsForTask(t *Task) int {
	if t != nil && t.Job.NConcurrentTrial > 0 {
		return t.Job.NConcurrentTrial
	}
	return defaultConcurrentTrials
}

// TrialArms is the cross-trial extension of geometry.go's WorkArms: the same
// single-trial A/B/C re-prefill work, plus the cross-trial reuse arm D that N
// concurrent trials of the SAME task imply. FrontierSWE runs n_concurrent_trials
// trials that share the byte-identical prefix, so the shared prefix is prefilled
// ONCE total across trials while only the incremental per-turn ingest multiplies —
// exactly swebench's c = P + C·(T-1)·R cross-worker arm, lifted to the TTS regime.
type TrialArms struct {
	WorkArms         // the embedded single-trial A/B/C arms + ratios at reuse rate r
	Trials   int     `json:"trials"`                // N concurrent trials of the same task (from n_concurrent_trials)
	ATrials  float64 `json:"a_naive_all_trials"`    // naive A ×N: every trial re-prefills its whole context
	DShared  float64 `json:"d_cross_trial_reuse"`   // shared prefix once + N·incremental result ingest
	ATrialsD float64 `json:"a_trials_over_d"`       // cross-trial work-elimination: naive-all-trials vs shared-prefix
	TTSTrial float64 `json:"tts_ratio_cross_trial"` // projected T_fak/T_raw with cross-trial prefix sharing (floor)
}

// CrossTrialArms computes the cross-trial work arms for a task geometry at reuse
// rate r and a trial fan-out. The single-trial arms come straight from the C4
// TTSModel (Derive); the cross-trial arms add the shared-prefix lever:
//
//	A_trials = N · A                    — every trial re-prefills its whole context
//	D_shared = P + N · ((A − P) · ratio)
//	           where ratio = C(r)/A folds the cross-turn reuse INTO the per-trial
//	           incremental work, and the prefix P is paid ONCE total across trials.
//
// At N=1 the cross-trial arm collapses to the single-trial C(r) (no fan-out, no
// cross-trial reuse) so the two surfaces agree. Pure arithmetic over geometry.go's
// WorkArms — no timing, deterministic, machine-load-independent.
func CrossTrialArms(g TaskGeometry, r float64, trials int) TrialArms {
	if trials < 1 {
		trials = 1
	}
	w := DefaultTTSModel().Derive(g, r)
	out := TrialArms{WorkArms: w, Trials: trials}

	P := float64(g.Prefix)
	A := float64(w.A)
	// The single-trial reused work C(r) splits into the prefix (paid once total
	// across trials) and the incremental remainder (paid per trial). N concurrent
	// trials share the byte-identical prefix, so the prefix term does NOT multiply.
	perTrialIncremental := w.C - P
	if perTrialIncremental < 0 {
		perTrialIncremental = 0
	}
	out.ATrials = A * float64(trials)
	out.DShared = P + float64(trials)*perTrialIncremental
	if out.DShared > 0 {
		out.ATrialsD = out.ATrials / out.DShared
		out.TTSTrial = out.DShared / out.ATrials
	}
	return out
}

// TTSProjection is the full per-task describe payload: the derived geometry, the
// single-trial work arms at the chosen reuse rate, and the cross-trial arms across
// a trial sweep. It is the FrontierSWE analogue of swebench's per-instance Summary
// row — a deterministic, offline-computable preview of the value stack on one task,
// labeled a PROJECTION (floor) throughout.
type TTSProjection struct {
	Name       string       `json:"name"`
	HasFixture bool         `json:"has_fixture"`      // a committed task.toml supplied the budget the geometry is derived from
	BudgetSec  float64      `json:"agent_budget_sec"` // the [agent] timeout_sec the turn count is projected from
	Reuse      float64      `json:"reuse_rate"`       // r used for the single-trial C arm and the TTS ratio (PROJECTION dial)
	Geometry   TaskGeometry `json:"geometry"`         // the derived (projected) per-turn geometry the model consumes
	Arms       WorkArms     `json:"arms"`             // single-trial A/B/C arms + ratios at reuse rate r
	TrialSweep []TrialArms  `json:"trial_sweep"`      // cross-trial arms, one per trial count in the sweep
}

// ProjectTTS builds the per-task projection at reuse rate r over a trial sweep.
// The geometry is derived from the task's budget (GeometryForTask); the single-
// trial arms and each trial-sweep entry come from the C4 TTSModel. An empty
// trialSweep defaults to {1, n_concurrent_trials} (or {1} when the task declares no
// concurrent trials), so the sweep always shows the single-trial floor plus the
// task's real fan-out. Deterministic and offline.
func ProjectTTS(t *Task, r float64, trialSweep []int) TTSProjection {
	g := GeometryForTask(t)
	m := DefaultTTSModel()
	p := TTSProjection{
		Name:     g.Name,
		Reuse:    clampReuse(r),
		Geometry: g,
		Arms:     m.Derive(g, r),
	}
	if t != nil {
		p.BudgetSec = t.AgentTimeoutSec()
	}
	if len(trialSweep) == 0 {
		trialSweep = defaultTrialSweep(TrialsForTask(t))
	}
	for _, n := range trialSweep {
		p.TrialSweep = append(p.TrialSweep, CrossTrialArms(g, r, n))
	}
	return p
}

// defaultTrialSweep returns the trial counts to sweep when none is supplied: the
// single-trial floor (1) plus the task's real fan-out, de-duplicated and ordered.
func defaultTrialSweep(trials int) []int {
	if trials <= 1 {
		return []int{1}
	}
	return []int{1, trials}
}

// clampReuse clamps r to [0,1], matching geometry.go's Derive so a projection's
// recorded Reuse never lies about a rate the model would have clamped.
func clampReuse(r float64) float64 {
	if r < 0 {
		return 0
	}
	if r > 1 {
		return 1
	}
	return r
}
