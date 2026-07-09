package frontierswe

// This file is the C9 run driver for FrontierSWE (epic #1706, issue #1715): the
// FrontierSWE analogue of `fak swebench run`. It drives one task end-to-end
// through the fak-routed agent shape and captures the two artifacts the later
// grade (C13) and compare (C12) consume: the per-task SUBMISSION and the per-turn
// TIME-TO-SOLUTION trace (turn count, cumulative wall-clock, the C8 reuse series).
//
// Honesty boundary. A real end-to-end run stands the C7 environment up under
// Docker/Modal and execs the harbor harness; on a box without Docker/Modal (or
// the GHCR task image) that cannot happen. The acceptance for #1715 explicitly
// permits a MOCKED environment, so this driver runs a deterministic mock agent
// loop that reproduces the SHAPE a real fak-routed run emits — the fak-routed
// job.yaml (C6), the per-turn cache-witness fold (C8), the submission tree, and
// the run meta — and always prints the exact remote command (from the C7
// env-adapter plan) so the real run is one copy-paste away on a capable host.
// Every mocked run stamps Mocked=true; nothing here claims a live serving win.
//
// Determinism. The mock drive is pure arithmetic over the task's committed facts
// (the [agent] timeout_sec budget and the projected long-horizon geometry) — no
// wall-clock, no network, no model — so the trace is reproduced bit-for-bit by
// the test, exactly like geometry.go's TTS floor and cachewitness_series.go's fold.

// RunSchema is the versioned schema id stamped on the run meta emitted by
// `fak frontierswe run`, so a produced run is inspectable and machine-joinable by
// the later grade (C13) and cross-run compare (C12).
const RunSchema = "fak.frontierswe.run.v1"

// RunConfig is the operator-facing shape of one FrontierSWE run.
type RunConfig struct {
	Task           *Task  // the loaded task (budget, job.yaml artifacts, no-internet boundary)
	Agent          string // the harness short name to route (claude-code, codex, ...)
	Model          string // model id forwarded unchanged to the upstream
	GatewayBaseURL string // the OpenAI-compatible fak gateway base URL the C6 shim sees
	UpstreamBase   string // the co-resident/pinned upstream fak serve fronts
	Trials         int    // n_concurrent_trials to record (>=1)
	Turns          int    // how many turns the mock loop is asked to drive (>=1)
	PredsOnly      bool   // stop before grading (grading is C13)
	Reuse          float64
}

// RunMeta is the fak.frontierswe.run.v1 meta: the identity + budget facts a later
// grade/compare needs to bind the run to its task, agent, model, and budget.
type RunMeta struct {
	Schema      string  `json:"schema"`
	Task        string  `json:"task"`
	Agent       string  `json:"agent"`              // harness short name actually routed
	Model       string  `json:"model"`              // model id forwarded to the upstream
	BudgetSec   int64   `json:"agent_budget_sec"`   // the [agent] timeout_sec enforced (the 20h headline)
	BudgetHours float64 `json:"agent_budget_hours"` // the same budget in hours, for the headline
	Trials      int     `json:"trials"`             // n_concurrent_trials recorded
	Turns       int     `json:"turns"`              // turns actually driven (after the budget cap)
	ElapsedSec  float64 `json:"elapsed_sec"`        // cumulative projected wall-clock of the driven turns
	Mocked      bool    `json:"mocked"`             // true when driven against a mocked env (no Docker/Modal)
	PredsOnly   bool    `json:"preds_only"`         // stopped before grading (grading is C13)
}

// TTSTracePoint is one per-turn point of the run's time-to-solution trace: the
// turn ordinal, the cumulative projected wall-clock reached at that turn, and the
// C8 cache-witness point folded from the turn's /metrics scrape.
type TTSTracePoint struct {
	Turn         int               `json:"turn"`
	CumWallSec   float64           `json:"cum_wall_sec"`
	CacheWitness CacheWitnessPoint `json:"cache_witness"`
}

// TTSTrace is the per-turn time-to-solution trace #1715 requires: the turn count,
// the cumulative wall-clock, and the C8 reuse series (the folded cache-witness
// trajectory + realized reuse rate). On a mocked run the wall-clock is projected
// from the budget, not measured — TotalWallSec is the driven trajectory's projected
// wall-clock, always <= the [agent] timeout_sec (budget-respecting by construction).
type TTSTrace struct {
	Schema       string             `json:"schema"`
	Turns        int                `json:"turns"`
	TotalWallSec float64            `json:"total_wall_sec"`
	BudgetSec    int64              `json:"agent_budget_sec"`
	Points       []TTSTracePoint    `json:"points"`
	CacheSeries  CacheWitnessSeries `json:"cache_series"` // the folded C8 reuse series + realized reuse rate
}

// RunArtifact is one job.yaml artifact and whether this run collected it. On a
// mocked run a placeholder is written so the collection SHAPE is exercised; on a
// real run the collected boolean reflects the actual harbor artifact copy.
type RunArtifact struct {
	Name      string `json:"name"`      // the job.yaml artifact path/name (e.g. solution.patch)
	Collected bool   `json:"collected"` // whether this run produced/collected it
	Mocked    bool   `json:"mocked"`    // the collected content is a mock placeholder
}

// RunGate is the honest local capability gate: whether the real C7 environment
// could be stood up on this host, and the exact remote command to run it where it
// cannot. It is the run's copy of the env-adapter plan's gate, never a silent fail.
type RunGate struct {
	Runnable      bool   `json:"runnable"`       // Docker present AND no-internet boundary satisfied
	DockerPresent bool   `json:"docker_present"` // Docker found on this host
	IntegrityOK   bool   `json:"integrity_ok"`   // the no-internet boundary held
	Reason        string `json:"reason,omitempty"`
	RemoteCommand string `json:"remote_command"` // the exact command to stand the env up on a capable host
}

// RunResult is the full run payload the cmd layer writes out: the meta, the
// per-turn TTS trace, the collected job.yaml artifact list, the fak-routed C6
// job.yaml, the submission contract, and the capability gate.
type RunResult struct {
	Meta         RunMeta       `json:"meta"`
	Trace        TTSTrace      `json:"tts_trace"`
	Artifacts    []RunArtifact `json:"artifacts"`
	JobYAML      string        `json:"job_yaml"`
	Submission   RunSubmission `json:"submission"`
	Gate         RunGate       `json:"gate"`
	WrappedAgent string        `json:"wrapped_agent"`
}

// RunSubmission is the per-task submission contract (#1715): the explicit target
// the task grades (cranelift's instruction.md: "the modified source tree at
// /app/wasmtime/"), and the files this run wrote to represent it. On a mocked run
// Files carries placeholder names derived from the job.yaml artifact list.
type RunSubmission struct {
	// Target is the per-task submission target the task's instruction.md fixes
	// (the source tree / files the verifier reads). Derived from the artifact list
	// when a submission-shaped artifact is declared, else a task-named default.
	Target string   `json:"target"`
	Files  []string `json:"files"`
	Mocked bool     `json:"mocked"`
}

// mockPerTurnGrowthTokens is the resident-context growth per mock turn (decode +
// result), reused verbatim from the describe-surface regime constants so the
// mocked reuse series and the TTS projection share one shape.
const mockPerTurnGrowthTokens = defaultDecodeTokens + defaultResultTokens

// BuildRun drives one FrontierSWE run and returns the full payload. It never does
// I/O (the cmd layer owns file writes) and never starts Docker — the mock loop is
// deterministic arithmetic over the task's budget + projected geometry, and the
// gate reports whether the real environment could be stood up here. Turns is
// capped so the projected cumulative wall-clock never exceeds the [agent]
// timeout_sec: the run respects the 20h budget by construction.
func BuildRun(cfg RunConfig) RunResult {
	task := cfg.Task
	if task == nil {
		task = &Task{Name: "unknown"}
	}
	agent := defaultString(cfg.Agent, "claude-code")
	model := defaultString(cfg.Model, DefaultModelEnv)
	gatewayBase := defaultString(cfg.GatewayBaseURL, DefaultGatewayBaseURL)
	upstreamBase := defaultString(cfg.UpstreamBase, DefaultUpstreamBase)
	trials := cfg.Trials
	if trials < 1 {
		trials = TrialsForTask(task)
	}
	reuse := cfg.Reuse
	if reuse <= 0 {
		reuse = DefaultReuseRate
	}

	// The capability gate + remote command come straight from the C7 env-adapter
	// plan so the run's honest gate and the env-adapter's are the same witness.
	wrapped, ok := WrappedAgentForHarness(agent)
	if !ok {
		wrapped = DefaultWrappedAgent
	}
	plan := BuildEnvAdapterPlan(EnvAdapterConfig{
		Task: task, GatewayBaseURL: gatewayBase, UpstreamBaseURL: upstreamBase,
		Model: model, WrappedAgent: wrapped,
	})
	gate := RunGate{
		Runnable:      plan.Capability.Runnable,
		DockerPresent: plan.Capability.DockerPresent,
		IntegrityOK:   plan.Integrity.OK,
		Reason:        plan.Capability.Reason,
		RemoteCommand: plan.Command,
	}

	// Budget cap: the mock projects turnsPerHour round-trips over the budgeted
	// hours, so the full trajectory is ProjectedTurns(budget). A request beyond
	// that would overrun the [agent] timeout_sec, so cap the driven turns there —
	// this is where the run "respects the 20h budget".
	budgetSec := task.AgentTimeoutSec()
	projectedTurns := ProjectedTurns(budgetSec)
	turns := cfg.Turns
	if turns < 1 {
		turns = 1
	}
	if turns > projectedTurns {
		turns = projectedTurns
	}
	// Projected per-turn wall-clock: the budget spread across the full trajectory,
	// so cumulative wall after `turns` turns is always <= the budget.
	perTurnWall := 0.0
	if projectedTurns > 0 {
		perTurnWall = budgetSec / float64(projectedTurns)
	}

	series, points := driveMockTrace(turns, perTurnWall)
	totalWall := 0.0
	if len(points) > 0 {
		totalWall = points[len(points)-1].CumWallSec
	}

	artifacts := collectArtifacts(task)
	submission := buildSubmission(task, artifacts)

	return RunResult{
		Meta: RunMeta{
			Schema:      RunSchema,
			Task:        task.Name,
			Agent:       harnessNameForWrapped(wrapped),
			Model:       model,
			BudgetSec:   int64(budgetSec),
			BudgetHours: budgetSec / 3600.0,
			Trials:      trials,
			Turns:       turns,
			ElapsedSec:  totalWall,
			Mocked:      true, // this increment always drives the mock loop; a live run is the C7-gated path
			PredsOnly:   cfg.PredsOnly,
		},
		Trace: TTSTrace{
			Schema:       CacheWitnessSchema,
			Turns:        turns,
			TotalWallSec: totalWall,
			BudgetSec:    int64(budgetSec),
			Points:       points,
			CacheSeries:  series,
		},
		Artifacts:    artifacts,
		JobYAML:      routedJobYAML(harnessNameForWrapped(wrapped), wrapped, gatewayBase, task.Environment.AllowInternet),
		Submission:   submission,
		Gate:         gate,
		WrappedAgent: wrapped,
	}
}

// driveMockTrace runs the deterministic mock agent loop for n turns and folds the
// per-turn cache scrapes into the C8 reuse series + per-turn TTS points. Each turn
// re-prefills a growing resident context; from turn 2 on, fak serves the prior
// resident context from the persistent KV (the reuse the thesis is about), so the
// synthesized cumulative counters yield a non-zero realized reuse rate. Pure.
func driveMockTrace(n int, perTurnWall float64) (CacheWitnessSeries, []TTSTracePoint) {
	samples := make([]CacheSample, 0, n)
	var cumPrompt, cumReused uint64
	for t := 1; t <= n; t++ {
		// Resident context re-prefilled on turn t (grows linearly in t).
		context := uint64(defaultPrefixTokens + (t-1)*mockPerTurnGrowthTokens)
		cumPrompt += context
		if t >= 2 {
			// The prior turn's resident context is served from the persistent KV.
			reused := uint64(defaultPrefixTokens + (t-2)*mockPerTurnGrowthTokens)
			cumReused += reused
		}
		samples = append(samples, CacheSample{
			Turn:         t,
			PromptTokens: cumPrompt,
			ReusedTokens: cumReused,
		})
	}
	series := FoldCacheWitness(samples)
	points := make([]TTSTracePoint, 0, len(series.Points))
	for i, p := range series.Points {
		points = append(points, TTSTracePoint{
			Turn:         p.Turn,
			CumWallSec:   float64(i+1) * perTurnWall,
			CacheWitness: p,
		})
	}
	return series, points
}

// collectArtifacts records the job.yaml artifact list and marks each as collected
// (a mock placeholder in this increment). A task with no job.yaml artifact list
// falls back to the canonical FrontierSWE trio (submission tree, agent log,
// verifier log) so the collection shape is always exercised.
func collectArtifacts(task *Task) []RunArtifact {
	names := task.Job.Artifacts
	if len(names) == 0 {
		names = []string{"solution.patch", "agent.log", "reward.json"}
	}
	out := make([]RunArtifact, 0, len(names))
	for _, n := range names {
		out = append(out, RunArtifact{Name: n, Collected: true, Mocked: true})
	}
	return out
}

// buildSubmission derives the per-task submission contract: the explicit target
// the task grades and the files this run wrote to represent it. The target is the
// modified source tree the instruction.md fixes; absent a per-task override it is a
// task-named default (mirroring cranelift's "/app/wasmtime/"). The files are the
// collected artifacts, so the submission always names concrete outputs.
func buildSubmission(task *Task, artifacts []RunArtifact) RunSubmission {
	files := make([]string, 0, len(artifacts))
	for _, a := range artifacts {
		files = append(files, a.Name)
	}
	return RunSubmission{
		Target: submissionTarget(task),
		Files:  files,
		Mocked: true,
	}
}

// submissionTarget is the per-task submission target path the verifier reads. The
// canonical FrontierSWE contract points at the modified source tree under /app; a
// task-named default keeps it explicit for any of the 17 tasks.
func submissionTarget(task *Task) string {
	if task == nil || task.Name == "" {
		return "/app"
	}
	return "/app/" + task.Name
}
