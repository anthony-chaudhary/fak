package ops

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// Engine coordinates the autonomous operations background tasks.
type Engine struct {
	mu        sync.Mutex
	RepoRoot  string
	Config    Config
	Ledger    *Ledger
	Storage   *StorageManager
	Process   *ProcessManager
	Workspace *WorkspaceManager
	// LoadLedger accounts every tick's own cost (#14474). It is the surface that
	// makes a healthy-but-quiet daemon distinguishable from a dead one.
	LoadLedger *LoadLedger
	// PauseStore persists the durable fak-ops-load-pause/1 state (#14476).
	PauseStore *PauseStore
	pause      PauseState

	lastStorageTick time.Time
	lastQuickTick   time.Time

	// selfCost reads this process's cumulative CPU time. Injectable so a test can
	// drive the accounting deterministically instead of racing a real clock.
	selfCost SelfCostFunc
	// hostSampler reads point-in-time host pressure once per tick (#14475).
	hostSampler HostSampler
	// pauseGate decides per-tick whether deferrable junkwork should skip (#14477).
	pauseGate PauseEvaluator
	// lastSelfCost is the previous tick's reading, used to derive per-tick CPU.
	lastSelfCost     time.Duration
	haveLastSelfCost bool
	// loadTick is the monotonic tick ordinal, continued across restarts by
	// seeding it from the rows already in the load ledger.
	loadTick uint64
	// clock is injectable for tests that must not sleep.
	clock func() time.Time
}

// NewEngine constructs an operations Engine confined to repoRoot: its Tick
// never sweeps the host Go cache or temp dumps, never scans host processes, and
// never kills. Use NewEngineWithHostReach to grant that reach.
func NewEngine(repoRoot string, cfg Config) (*Engine, error) {
	return NewEngineWithHostReach(repoRoot, cfg, HostReach{})
}

// NewEngineWithHostReach constructs an Engine whose Tick may act outside
// repoRoot exactly as far as host allows.
func NewEngineWithHostReach(repoRoot string, cfg Config, host HostReach) (*Engine, error) {
	ledPath := DefaultLedgerPath(repoRoot)
	led, err := OpenLedger(ledPath)
	if err != nil {
		return nil, err
	}

	loadLed, err := OpenLoadLedger(DefaultLoadLedgerPath(repoRoot))
	if err != nil {
		return nil, err
	}

	pauseStore, err := OpenPauseStore(DefaultPauseStatePath(repoRoot))
	if err != nil {
		return nil, err
	}

	e := &Engine{
		RepoRoot:    repoRoot,
		Config:      cfg,
		Ledger:      led,
		Storage:     NewStorageManager(repoRoot, cfg),
		Process:     NewProcessManager(cfg),
		Workspace:   NewWorkspaceManager(repoRoot),
		LoadLedger:  loadLed,
		PauseStore:  pauseStore,
		pause:       pauseStore.ReadFailSafe(),
		selfCost:    DefaultSelfCost,
		hostSampler: HarnessHostSampler{},
		pauseGate:   DefaultPauseEvaluator{},
		clock:       time.Now,
	}
	e.Storage.GoCacheRoot = host.GoCacheRoot
	e.Storage.TempRoots = host.TempRoots
	e.Process.Collect = host.CollectProcesses
	e.Process.EnactAuthorized = host.EnactProcessReap
	// Continue the tick ordinal across restarts so a gap in the load ledger is
	// never mistaken for a tick that simply was not recorded.
	if n, err := loadLed.Count(); err == nil {
		e.loadTick = n
	}
	return e, nil
}

// SetSelfCostFunc installs a deterministic CPU reader (tests, and hosts whose
// platform reader is unavailable).
func (e *Engine) SetSelfCostFunc(fn SelfCostFunc) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.selfCost = fn
	e.lastSelfCost = 0
	e.haveLastSelfCost = false
}

// SetHostSampler installs an injectable host-load sampler (tests, or hosts with
// custom pressure readers). Passing nil yields UNKNOWN host samples.
func (e *Engine) SetHostSampler(s HostSampler) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.hostSampler = s
}

// SetPauseEvaluator installs an injectable pause-gate evaluator (tests, or custom rules).
// Passing nil restores DefaultPauseEvaluator{}.
func (e *Engine) SetPauseEvaluator(g PauseEvaluator) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if g == nil {
		e.pauseGate = DefaultPauseEvaluator{}
		return
	}
	e.pauseGate = g
}

// SetClock installs an injectable clock.
func (e *Engine) SetClock(fn func() time.Time) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if fn != nil {
		e.clock = fn
	}
}

func (e *Engine) now() time.Time {
	if e.clock != nil {
		return e.clock()
	}
	return time.Now()
}

// Paused reports whether the engine is currently in a durable pause state.
func (e *Engine) Paused() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pause.Paused
}

// PauseState returns a copy of the engine's current durable pause record.
func (e *Engine) PauseState() PauseState {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pause
}

// Pause transitions the engine into a durable paused state and writes the
// fak-ops-load-pause/1 record once on transition.
func (e *Engine) Pause(reason PauseReasonClass, detail string, cond ReactivateCondition) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pauseLocked(reason, detail, cond)
}

func (e *Engine) pauseLocked(reason PauseReasonClass, detail string, cond ReactivateCondition) error {
	next := PauseState{
		Schema:         PauseSchemaV1,
		Paused:         true,
		ReasonClass:    reason,
		ReasonDetail:   detail,
		PausedAt:       e.now().UTC(),
		PausedTicks:    0,
		ReactivateWhen: cond,
	}
	if err := next.Validate(); err != nil {
		return err
	}
	if e.PauseStore != nil {
		if err := e.PauseStore.Write(next); err != nil {
			return err
		}
	}
	e.pause = next
	return nil
}

// Resume exits the durable paused state, preserving transition history, and
// writes the fak-ops-load-pause/1 record once on transition.
func (e *Engine) Resume() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.resumeLocked()
}

func (e *Engine) resumeLocked() error {
	return e.resumeWithReasonLocked("")
}

func (e *Engine) resumeWithReasonLocked(reason PauseReasonClass) error {
	if !e.pause.Paused {
		return nil
	}
	next := e.pause
	next.Schema = PauseSchemaV1
	next.Paused = false
	next.ResumedAt = e.now().UTC()
	if reason != "" {
		next.ResumeReasonClass = reason
	}
	if e.PauseStore != nil {
		if err := e.PauseStore.Write(next); err != nil {
			return err
		}
	}
	e.pause = next
	return nil
}

// RecordProbe updates the last probe timestamp and result on the pause record.
func (e *Engine) RecordProbe(result string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	next := e.pause
	next.Schema = PauseSchemaV1
	next.LastProbeAt = e.now().UTC()
	next.LastProbeResult = result
	if e.PauseStore != nil {
		if err := e.PauseStore.Write(next); err != nil {
			return err
		}
	}
	e.pause = next
	return nil
}

// selfCPUDelta returns the CPU time consumed since the previous tick. ok is
// false when the axis is unreadable or there is no prior reading to subtract,
// which the row records as unmeasured rather than as free.
func (e *Engine) selfCPUDelta() (time.Duration, bool) {
	if e.selfCost == nil {
		return 0, false
	}
	total, ok := e.selfCost()
	if !ok {
		return 0, false
	}
	if !e.haveLastSelfCost {
		e.lastSelfCost = total
		e.haveLastSelfCost = true
		return 0, false
	}
	delta := total - e.lastSelfCost
	e.lastSelfCost = total
	if delta < 0 {
		// The counter went backwards: treat it as unreadable rather than
		// emitting a negative cost.
		return 0, false
	}
	return delta, true
}

func (e *Engine) sampleHost() *HostLoad {
	if e.hostSampler == nil {
		return &HostLoad{State: HostSampleUnknown}
	}
	sample, err := e.hostSampler.SampleHost()
	return classifyHostSample(sample, err)
}

// recordLoadCycle appends exactly one fak-ops-load/1 row per tick. A failure to
// account is returned to the caller but never aborts the maintenance the tick
// already performed.
func (e *Engine) recordLoadCycle(started, finished time.Time, phases []PhaseLoad, mutations int, failed bool, host *HostLoad, decision PauseDirective, junkworkSkipped bool) error {
	e.loadTick++
	verdict := LoadIdle
	switch {
	case failed:
		verdict = LoadError
	case mutations > 0:
		verdict = LoadProductive
	case junkworkSkipped:
		if anyReliefRan(phases) {
			verdict = LoadPressureReliefOnly
		} else {
			verdict = LoadSkipped
		}
	}

	cpu, haveCPU := e.selfCPUDelta()
	cycle := LoadCycle{
		Schema:       LoadSchemaV1,
		Tick:         e.loadTick,
		Started:      started.UTC(),
		Finished:     finished.UTC(),
		WallMS:       finished.Sub(started).Milliseconds(),
		Phases:       phases,
		Mutations:    mutations,
		Verdict:      verdict,
		PauseAction:  string(decision.JunkworkAction),
		ReasonClass:  decision.ReasonClass,
		TriggerValue: decision.TriggerValue,
		Host:         host,
	}
	if haveCPU {
		cycle.HaveCPU = true
	}
	if cpu > 0 {
		cycle.HaveSelfCPU = true
		cycle.SelfCPUMillis = cpu.Milliseconds()
		cycle.CPUSeconds = cpu.Seconds()
	}
	if e.LoadLedger == nil {
		return nil
	}
	return e.LoadLedger.Record(cycle)
}

// Tick executes one operations pass. If forceAll is true, all tiers run.
// Otherwise, quick checks (locks, processes) run every 60s and storage sweeps run every 15m.
//
// Every tick is accounted as one fak-ops-load/1 row (#14474), including a tick
// that finds nothing to do: the accounting is the load-bearing part, and a
// mutation-only ledger cannot distinguish "quiet" from "dead".
func (e *Engine) Tick(ctx context.Context, forceAll bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	started := e.now()
	now := started

	gate := e.pauseGate
	if gate == nil {
		gate = DefaultPauseEvaluator{}
	}

	_, _, maxPause, maxSkips := effectivePauseThresholds(e.Config)
	policy := DefaultReactivationPolicy()
	policy.MaxPause = maxPause
	policy.Now = started

	var host *HostLoad
	var decision PauseDirective
	if e.pause.Paused && !ShouldProbe(started, e.pause, policy) && !deadlineReached(started, e.pause, maxPause, maxSkips) {
		decision = PauseDirective{
			JunkworkAction: ActionSkip,
			ReliefAction:   ActionRunPressureReliefOnly,
			ReasonClass:    e.pause.ReasonClass,
			TriggerValue:   "probe_deferred",
		}
	} else {
		host = e.sampleHost()
		decision = gate.Decide(LoadCycle{Started: started, Host: host}, e.pause, e.Config)
		if e.pause.Paused && !decision.ShouldResume {
			nextProbeState, _, _ := ApplyProbeSample(e.pause, started, host, policy)
			e.pause = nextProbeState
		}
	}

	if decision.ShouldPause && !e.pause.Paused {
		_, resumeRatio, maxPause, _ := effectivePauseThresholds(e.Config)
		_ = e.pauseLocked(decision.ReasonClass, decision.TriggerValue, ReactivateCondition{
			PressureRatioBelow: resumeRatio,
			MaxPause:           maxPause.String(),
		})
	} else if decision.ShouldResume && e.pause.Paused {
		_ = e.resumeWithReasonLocked(decision.ReasonClass)
	} else if e.pause.Paused && decision.JunkworkAction == ActionSkip {
		// Update skipped-tick count in memory only; never rewrite the durable
		// pause file on every skipped tick (#14476).
		e.pause.PausedTicks++
	}

	reliefAction := decision.ForPhase(PhasePressureRelief)
	junkworkAction := decision.ForPhase(PhaseDeferrable)
	junkworkSkipped := junkworkAction == ActionSkip

	mutations := 0
	failed := false
	// phases records every maintenance phase this tick CONSIDERED, including one
	// that was not due or was gated. "The expensive tier was skipped" is itself
	// a load fact. Each phase declares its class at its call site (#14477).
	quickDue := forceAll || now.Sub(e.lastQuickTick) >= 60*time.Second
	storageDue := forceAll || now.Sub(e.lastStorageTick) >= 15*time.Minute
	assetDue := quickDue && junkworkAction == ActionRun
	phases := []PhaseLoad{
		{Phase: "locks_worktrees", Ran: quickDue, PhaseAction: string(reliefAction)},
		{Phase: "process_runaways", Ran: quickDue, PhaseAction: string(reliefAction)},
		{Phase: "asset_sync", Ran: assetDue, Deferrable: true, PhaseAction: string(junkworkAction)},
		{Phase: "storage_reclaim", Ran: storageDue, PhaseAction: string(reliefAction)},
	}
	if quickDue {
		start := time.Now()
		healRes, healErr := e.Workspace.SweepLocksAndWorktrees(ctx, false)
		dur := time.Since(start).Milliseconds()
		if healErr != nil {
			failed = true
		}

		if len(healRes.LocksEvicted) > 0 {
			mutations++
			phaseMutated(&phases, "locks_worktrees", dur)
			_ = e.Ledger.Record(Event{
				ActionType: ActionLockEvict,
				Details:    fmt.Sprintf("evicted dead locks: %s", strings.Join(healRes.LocksEvicted, ", ")),
				DurationMS: dur,
			})
		}
		if len(healRes.WorktreesPruned) > 0 {
			mutations++
			phaseMutated(&phases, "locks_worktrees", dur)
			_ = e.Ledger.Record(Event{
				ActionType: ActionWorktreePrune,
				Details:    fmt.Sprintf("pruned cold worktrees: %s", strings.Join(healRes.WorktreesPruned, ", ")),
				DurationMS: dur,
			})
		}

		// Process runaway & orphan checks
		startProc := time.Now()
		procRes, procErr := e.Process.SweepProcessRunaways(ctx, false)
		durProc := time.Since(startProc).Milliseconds()
		if procErr != nil {
			failed = true
		}
		phaseWall(&phases, "process_runaways", durProc)
		phaseSkipped(&phases, "process_runaways", procRes.Skipped)

		if len(procRes.PIDsReaped) > 0 {
			mutations++
			phaseMutated(&phases, "process_runaways", durProc)
			_ = e.Ledger.Record(Event{
				ActionType:   ActionProcessReap,
				Details:      fmt.Sprintf("reaped %d processes: %s", len(procRes.PIDsReaped), strings.Join(procRes.Names, ", ")),
				PIDsAffected: procRes.PIDsReaped,
				DurationMS:   durProc,
			})
		}

		// Background asset synchronization for portable skill adapters (#11156, #11158).
		// Declared deferrable at its call site (#14477): skipped under host pressure.
		if assetDue {
			startAsset := time.Now()
			assetReceipt, synced, assetErr := projectassets.EnsureSync(e.RepoRoot)
			durAsset := time.Since(startAsset).Milliseconds()
			phaseWall(&phases, "asset_sync", durAsset)
			if assetErr != nil {
				failed = true
			}
			if assetErr == nil && synced {
				mutations++
				phaseMutated(&phases, "asset_sync", durAsset)
				_ = e.Ledger.Record(Event{
					ActionType: ActionAssetSync,
					Details:    fmt.Sprintf("synchronized portable skill adapters for opencode and codex workers (%d skills)", len(assetReceipt.Harnesses["opencode"].Canonical)),
					DurationMS: durAsset,
				})
			}
		}

		e.lastQuickTick = now
	}

	// 2. Storage & Cache Lifecycle (every 15m or forceAll)
	if storageDue {
		start := time.Now()
		reclaimRes, reclaimErr := e.Storage.ReclaimCascade(ctx, false)
		dur := time.Since(start).Milliseconds()
		phaseWall(&phases, "storage_reclaim", dur)
		phaseSkipped(&phases, "storage_reclaim", reclaimRes.Skipped)
		if reclaimErr != nil {
			failed = true
		}

		if reclaimRes.TotalBytes > 0 {
			mutations++
			phaseMutated(&phases, "storage_reclaim", dur)
			_ = e.Ledger.Record(Event{
				ActionType:     ActionStorageReclaim,
				Details:        fmt.Sprintf("reclaimed %d bytes across %d files (%s)", reclaimRes.TotalBytes, reclaimRes.FilesCount, strings.Join(reclaimRes.Actions, "; ")),
				BytesReclaimed: reclaimRes.TotalBytes,
				DurationMS:     dur,
			})
		}

		e.lastStorageTick = now
	}

	return e.recordLoadCycle(started, e.now(), phases, mutations, failed, host, decision, junkworkSkipped)
}

func anyReliefRan(phases []PhaseLoad) bool {
	for _, p := range phases {
		if !p.Deferrable && p.Ran {
			return true
		}
	}
	return false
}

// phaseWall records how long a considered phase took on this tick.
func phaseWall(phases *[]PhaseLoad, name string, wallMS int64) {
	for i := range *phases {
		if (*phases)[i].Phase == name {
			(*phases)[i].WallMS = wallMS
			return
		}
	}
}

// phaseSkipped records the host-reaching steps a phase declined.
func phaseSkipped(phases *[]PhaseLoad, name string, skips []ScopeSkip) {
	if len(skips) == 0 {
		return
	}
	for i := range *phases {
		if (*phases)[i].Phase == name {
			(*phases)[i].Skipped = append((*phases)[i].Skipped, skips...)
			return
		}
	}
}

// phaseMutated marks a considered phase as having changed state, and stamps the
// wall time observed for the mutation path.
func phaseMutated(phases *[]PhaseLoad, name string, wallMS int64) {
	for i := range *phases {
		if (*phases)[i].Phase == name {
			(*phases)[i].Mutated = true
			(*phases)[i].WallMS = wallMS
			return
		}
	}
}
