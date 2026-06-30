// Package procguard is the native Go port of the standalone operator/control-pane
// modes of tools/proc_resource_guard.py — catch a runaway process before it pins
// the host. The dispatch-preflight core (thread/handle/working-set level breach) is
// already a pure classifier in internal/dispatchtick.EvaluateProcGuard; this package
// reuses that classifier for the resource-level dimension and adds the richer modes
// the Python tool still owns: the sustained per-core CPU-pin dimension, orphaned
// helper / idle launcher-shell reaping, the cross-tick CPU-pin streak ledger, the
// opt-in --enact reaper, and the full machine-readable JSON contract the control
// pane folds (schema "fleet-proc-resource-guard/1").
//
// Root cause this exists for: a single process can leak OS threads without bound
// (the witnessed incident was an external llama-cli invoked CPU-only with no -t
// thread bound that climbed to ~129,427 threads on ONE process, pinning the
// machine). No existing fleet watchdog watches per-process resource *level*. This
// is that missing guard, promoted to a first-class native fak command.
//
// It is a control-pane loop first (read-only status the pane folds: ok==false ==
// ACTION, a runaway is live) and an opt-in reaper second (Enact kills the flagged
// runaways, never a protected OS-critical process or this tool's own process tree).
package procguard

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// Schema is the JSON contract identity the Python tool emits and the control pane
// folds; kept byte-identical so the native command is a drop-in replacement.
const Schema = "fleet-proc-resource-guard/1"

// Defaults mirror proc_resource_guard.py's module-level constants exactly.
const (
	DefaultMaxThreads      = dispatchtick.ProcGuardDefaultMaxThreads // 2000
	DefaultMaxHandles      = 0                                       // 0 == dimension disabled
	DefaultMaxWorkingSetMB = 0                                       // 0 == dimension disabled

	DefaultMaxCPUPct      = 0.0 // 0 == CPU dimension disabled (default)
	DefaultCPUWindowSec   = 3.0 // seconds between consecutive CPU samples
	DefaultCPUSamples     = 2   // 2 == one window; >2 requires the pin to hold every window
	DefaultCPUReapConfirm = 1   // 1 == reap a CPU-only pin on first detection (manual run)

	DefaultIdleShellAgeSec = 1800 // 30 min: well past any session-launch transient

	// CPUStreakLedger is the ledger file name the standing reaper persists between
	// scheduled ticks so a CPU-only pin is reaped only after it has been confirmed
	// across CPUReapConfirm consecutive runs.
	CPUStreakLedger = "cpu_pin_streak.json"
)

// DefaultOrphanPatterns matches an ephemeral stdio helper whose owning session has
// exited (a per-session MCP server still resident after its client died).
var DefaultOrphanPatterns = []string{"dos_mcp.server"}

// DefaultIdleShellNames are launcher shells that legitimately wrap a session; one
// with zero live children aged past the floor is a stray launcher.
var DefaultIdleShellNames = map[string]bool{"pwsh": true, "powershell": true, "bash": true}

// DefaultOrphanConsoleShellNames are shell processes that can remain after their
// owner exits while keeping only a console-host child alive.
var DefaultOrphanConsoleShellNames = map[string]bool{"cmd": true}

// DefaultConsoleHostChildNames are console-host processes that do not represent
// useful child work on their own.
var DefaultConsoleHostChildNames = map[string]bool{"conhost": true, "openconsole": true}

// DefaultInteractiveParentNames are attended terminal/session parents. Idle
// shells under these are reported as visible topology, not reaped as abandoned
// background launcher shells.
var DefaultInteractiveParentNames = map[string]bool{
	"windowsterminal": true, "terminal": true, "conhost": true, "openconsole": true,
	"explorer": true, "cmd": true, "powershell": true, "pwsh": true,
}

// DefaultConsoleProneNames are background helpers that have historically been
// associated with user-visible console windows when launched without a no-window
// creation flag.
var DefaultConsoleProneNames = map[string]bool{
	"cmd": true, "powershell": true, "pwsh": true, "bash": true, "sh": true, "zsh": true,
	"gh": true, "git": true, "go": true, "python": true, "node": true,
	"conhost": true, "openconsole": true,
}

// ProtectedNames are OS-critical processes that must NEVER be reaped even with
// Enact. Reuses the same set the dispatchtick classifier already defines.
var ProtectedNames = dispatchtick.ProtectedProcessNames

// Proc is one process snapshot row fed to the classifier. Pointer fields that are
// nil mean the collector could not read that dimension on this platform; a missing
// dimension is skipped, never treated as a breach (the no-false-reap contract).
type Proc struct {
	PID     int
	Name    string
	Threads *int
	Handles *int
	WSMB    *int
	CPUPct  *float64 // sustained per-core CPU% (top-style: 100 == one full core)
	PPID    *int
	Cmdline string
	AgeSec  *int
	Start   string // process start time, for the reuse-safe streak key
}

// Thresholds are the configured per-dimension ceilings (0 disables a dimension).
type Thresholds struct {
	MaxThreads int     `json:"max_threads"`
	MaxHandles int     `json:"max_handles"`
	MaxWSMB    int     `json:"max_ws_mb"`
	MaxCPUPct  float64 `json:"max_cpu_pct"`
}

// DefaultThresholds returns the Python tool's default ceilings.
func DefaultThresholds() Thresholds {
	return Thresholds{
		MaxThreads: DefaultMaxThreads,
		MaxHandles: DefaultMaxHandles,
		MaxWSMB:    DefaultMaxWorkingSetMB,
		MaxCPUPct:  DefaultMaxCPUPct,
	}
}

// Finding is one flagged process row, shape-compatible with the Python contract.
type Finding struct {
	PID        int      `json:"pid"`
	Name       string   `json:"name"`
	Threads    *int     `json:"threads"`
	Handles    *int     `json:"handles"`
	WSMB       *int     `json:"ws_mb"`
	CPUPct     *float64 `json:"cpu_pct,omitempty"`
	PPID       *int     `json:"ppid,omitempty"`
	ParentName string   `json:"parent_name,omitempty"`
	Start      string   `json:"start,omitempty"`
	Reasons    []string `json:"reasons"`
	Protected  bool     `json:"protected"`
	Kind       string   `json:"kind,omitempty"`
	Action     string   `json:"action,omitempty"`
	CPUStreak  *int     `json:"cpu_streak,omitempty"`
}

// Enacted is one reap outcome row in the JSON contract's "enacted" list.
type Enacted struct {
	PID    int    `json:"pid"`
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// ObservationSummary is a non-actionable topology signal. It keeps recurring
// console/process sprawl visible even when the conservative reaper has nothing
// old enough or definite enough to kill.
type ObservationSummary struct {
	RelationScanned                  int            `json:"relation_scanned"`
	ConsoleProneCount                int            `json:"console_prone_count"`
	ConsoleProneByName               map[string]int `json:"console_prone_by_name,omitempty"`
	ConsoleProneByParent             map[string]int `json:"console_prone_by_parent,omitempty"`
	BackgroundConsoleProneCount      int            `json:"background_console_prone_count"`
	ConsoleHostCount                 int            `json:"console_host_count"`
	IdleShellCandidates              int            `json:"idle_shell_candidates"`
	AgedIdleShellCandidates          int            `json:"aged_idle_shell_candidates"`
	OrphanConsoleShellCandidates     int            `json:"orphan_console_shell_candidates"`
	AgedOrphanConsoleShellCandidates int            `json:"aged_orphan_console_shell_candidates"`
}

// Payload is the full machine-readable status the control pane folds — the Go
// mirror of the Python build_payload() dict, field-for-field.
type Payload struct {
	Schema                 string              `json:"schema"`
	OK                     bool                `json:"ok"`
	Platform               string              `json:"platform"`
	Thresholds             Thresholds          `json:"thresholds"`
	CPUReapConfirm         int                 `json:"cpu_reap_confirm"`
	CPUStreaks             map[string]int      `json:"cpu_streaks"`
	Scanned                int                 `json:"scanned"`
	FlaggedCount           int                 `json:"flagged_count"`
	ActionableFlaggedCount int                 `json:"actionable_flagged_count"`
	Flagged                []Finding           `json:"flagged"`
	Enacted                []Enacted           `json:"enacted"`
	Enact                  bool                `json:"enact"`
	Observations           *ObservationSummary `json:"observations,omitempty"`
	CollectError           string              `json:"collect_error,omitempty"`
	NextAction             string              `json:"next_action"`
}

// Options bundles the classifier knobs build accepts.
type Options struct {
	Thresholds     Thresholds
	ProtectedPIDs  []int
	AllowNames     []string
	Enact          bool
	CPUReapConfirm int
	CPUStreaksPrev map[string]int
	OrphanRows     []Finding
	Observations   *ObservationSummary
	Platform       string
	CollectError   string
	// Killer is the destructive reaper, injected so tests never spawn anything.
	// nil => no kill is ever attempted (report-only), even with Enact.
	Killer func(pid int) (bool, string)
}

// Classify returns the subset of procs that breach a configured resource-LEVEL
// threshold (thread/handle/working-set/CPU). The thread/handle/ws core is delegated
// to dispatchtick.EvaluateProcGuard (no duplication); the CPU-pin dimension is added
// here because it is the one dimension dispatch preflight does not carry.
func Classify(procs []Proc, th Thresholds, protectedPIDs []int, allowNames []string) []Finding {
	core := dispatchtick.EvaluateProcGuard(dispatchtick.ProcGuardInput{
		Processes:     toDispatchProcs(procs),
		Thresholds:    dispatchtick.ProcGuardThresholds{MaxThreads: th.MaxThreads, MaxHandles: th.MaxHandles, MaxWorkingSetMB: th.MaxWSMB},
		ProtectedPIDs: protectedPIDs,
		AllowNames:    allowNames,
	})
	// Index the auxiliary fields the dispatch classifier does not carry (cpu/start/ppid).
	aux := map[int]Proc{}
	for _, p := range procs {
		aux[p.PID] = p
	}
	allow := lowerSet(allowNames)
	protSet := intSet(protectedPIDs)

	out := make([]Finding, 0, len(core.Flagged)+len(procs))
	seen := map[int]bool{}
	for _, f := range core.Flagged {
		p := aux[f.PID]
		fnd := Finding{
			PID: f.PID, Name: f.Name, Threads: f.Threads, Handles: f.Handles, WSMB: f.WorkingSetMB,
			CPUPct: p.CPUPct, PPID: p.PPID, Start: p.Start,
			Reasons: append([]string{}, f.Reasons...), Protected: f.Protected,
		}
		out = append(out, fnd)
		seen[f.PID] = true
	}
	// Add the CPU-pin dimension the dispatch classifier omits: a single-threaded
	// runaway pins one core (normal thread/handle count) and is only visible as a
	// sustained per-core CPU%, so it merges into (or creates) a finding here.
	if th.MaxCPUPct > 0 {
		for _, p := range procs {
			name := strings.TrimSpace(p.Name)
			if allow[strings.ToLower(name)] {
				continue
			}
			if p.CPUPct == nil || *p.CPUPct <= th.MaxCPUPct {
				continue
			}
			reason := fmt.Sprintf("cpu %.0f%%/core > %.0f%% sustained", *p.CPUPct, th.MaxCPUPct)
			if seen[p.PID] {
				for i := range out {
					if out[i].PID == p.PID {
						out[i].Reasons = append(out[i].Reasons, reason)
						break
					}
				}
				continue
			}
			out = append(out, Finding{
				PID: p.PID, Name: name, Threads: p.Threads, Handles: p.Handles, WSMB: p.WSMB,
				CPUPct: p.CPUPct, PPID: p.PPID, Start: p.Start,
				Reasons:   []string{reason},
				Protected: protSet[p.PID] || ProtectedNames[strings.ToLower(name)],
			})
		}
	}
	sortFindings(out)
	return out
}

// RelationTopology is the process-tree view derived from one relation scan.
type RelationTopology struct {
	LivePIDs    map[int]bool
	ChildCounts map[int]int
	ChildNames  map[int][]string
	ParentNames map[int]string
}

// OrphanOptions bundles the orphan-sprawl detector knobs.
type OrphanOptions struct {
	Patterns                []string
	IdleShellNames          map[string]bool
	OrphanConsoleShellNames map[string]bool
	ConsoleHostChildNames   map[string]bool
	InteractiveParentNames  map[string]bool
	ConsoleProneNames       map[string]bool
	MinAgeSec               int
	ReapIdleShells          bool
	ProtectedPIDs           []int
	AllowNames              []string
}

// DefaultOrphanOptions returns the default orphan-sprawl detector settings.
func DefaultOrphanOptions() OrphanOptions {
	return OrphanOptions{
		Patterns:                DefaultOrphanPatterns,
		IdleShellNames:          DefaultIdleShellNames,
		OrphanConsoleShellNames: DefaultOrphanConsoleShellNames,
		ConsoleHostChildNames:   DefaultConsoleHostChildNames,
		InteractiveParentNames:  DefaultInteractiveParentNames,
		ConsoleProneNames:       DefaultConsoleProneNames,
		MinAgeSec:               DefaultIdleShellAgeSec,
	}
}

// NewRelationTopology builds the ppid/name indexes shared by the reaper and the
// non-actionable observation layer.
func NewRelationTopology(procs []Proc) RelationTopology {
	live := map[int]bool{}
	parents := map[int]string{}
	for _, p := range procs {
		if p.PID > 0 {
			live[p.PID] = true
			parents[p.PID] = p.Name
		}
	}
	return RelationTopology{
		LivePIDs:    live,
		ChildCounts: ChildCounts(procs),
		ChildNames:  ChildNames(procs),
		ParentNames: parents,
	}
}

// ClassifyOrphans flags orphaned sprawl: ephemeral helpers whose owner is gone, and
// idle launcher shells with no live children. Pure: livePIDs and childCounts are
// derived from the same relation scan by the caller.
func ClassifyOrphans(procs []Proc, livePIDs map[int]bool, childCounts map[int]int, orphanPatterns []string, idleShellNames map[string]bool, minAgeSec int, reapIdleShells bool, protectedPIDs []int, allowNames []string) []Finding {
	top := RelationTopology{LivePIDs: livePIDs, ChildCounts: childCounts}
	opt := DefaultOrphanOptions()
	opt.Patterns = orphanPatterns
	opt.IdleShellNames = idleShellNames
	opt.MinAgeSec = minAgeSec
	opt.ReapIdleShells = reapIdleShells
	opt.ProtectedPIDs = protectedPIDs
	opt.AllowNames = allowNames
	return ClassifyOrphanSprawl(procs, top, opt)
}

// ClassifyOrphanSprawl is the topology-aware detector used by the native command.
// It keeps the kill criteria conservative: attended terminal parents are spared,
// and cmd.exe is only flagged when its owner is gone and its remaining children are
// console hosts.
func ClassifyOrphanSprawl(procs []Proc, top RelationTopology, opt OrphanOptions) []Finding {
	opt = normalizeOrphanOptions(opt)
	patterns := nonEmpty(opt.Patterns)
	allow := lowerSet(opt.AllowNames)
	protSet := intSet(opt.ProtectedPIDs)
	flagged := []Finding{}
	for _, p := range procs {
		name := strings.TrimSpace(p.Name)
		stem := stemLower(name)
		if allow[stem] {
			continue
		}
		reasons := []string{}
		kind := ""

		// Orphaned ephemeral helper: matches a known pattern AND its owner is gone.
		if len(patterns) > 0 && !ownerAlive(p.PPID, top.LivePIDs) {
			hay := name + " " + p.Cmdline
			for _, pat := range patterns {
				if strings.Contains(hay, pat) {
					ppid := 0
					if p.PPID != nil {
						ppid = *p.PPID
					}
					reasons = append(reasons, fmt.Sprintf("orphaned helper: owner pid %d not alive", ppid))
					kind = "orphan-helper"
					break
				}
			}
		}

		// Idle launcher shell: a wrapper shell with no live children, aged out.
		if opt.ReapIdleShells && opt.IdleShellNames[stem] {
			kids := top.ChildCounts[p.PID]
			aged := opt.MinAgeSec <= 0 || (p.AgeSec != nil && *p.AgeSec >= opt.MinAgeSec)
			if kids == 0 && aged && !attendedParent(p, top, opt.InteractiveParentNames) {
				note := ""
				if p.AgeSec != nil {
					note = fmt.Sprintf(", age %ds", *p.AgeSec)
				}
				reasons = append(reasons, "idle launcher shell: 0 live children"+note)
				if kind == "" {
					kind = "idle-shell"
				}
			}
		}

		// Orphaned console shell: cmd.exe can outlive the parent with only its
		// conhost/openconsole child, so the generic zero-child idle-shell rule
		// cannot see it. This fires only when the owner is gone and every remaining
		// child is just the console host.
		if opt.ReapIdleShells && opt.OrphanConsoleShellNames[stem] {
			aged := opt.MinAgeSec <= 0 || (p.AgeSec != nil && *p.AgeSec >= opt.MinAgeSec)
			children := childNameStems(top.ChildNames[p.PID])
			if !ownerAlive(p.PPID, top.LivePIDs) && aged && onlyConsoleChildren(children, opt.ConsoleHostChildNames) {
				ppid := 0
				if p.PPID != nil {
					ppid = *p.PPID
				}
				note := ""
				if p.AgeSec != nil {
					note = fmt.Sprintf(", age %ds", *p.AgeSec)
				}
				childNote := ", children=none"
				if len(children) > 0 {
					childNote = ", children=" + strings.Join(children, ",")
				}
				reasons = append(reasons, fmt.Sprintf("orphaned console shell: owner pid %d not alive%s%s", ppid, childNote, note))
				if kind == "" {
					kind = "orphan-console-shell"
				}
			}
		}

		if len(reasons) == 0 {
			continue
		}
		flagged = append(flagged, Finding{
			PID: p.PID, Name: name, PPID: p.PPID, ParentName: parentName(p, top), Threads: p.Threads, WSMB: p.WSMB,
			Reasons:   reasons,
			Protected: protSet[p.PID] || ProtectedNames[stem],
			Kind:      kind,
		})
	}
	sort.Slice(flagged, func(i, j int) bool { return flagged[i].PID < flagged[j].PID })
	return flagged
}

// ChildCounts builds the ppid -> live-child-count map an idle-shell scan needs.
func ChildCounts(procs []Proc) map[int]int {
	counts := map[int]int{}
	for _, p := range procs {
		if p.PPID != nil {
			counts[*p.PPID]++
		}
	}
	return counts
}

// ChildNames builds the ppid -> live child process-name map for console-host-only
// orphan-shell detection.
func ChildNames(procs []Proc) map[int][]string {
	out := map[int][]string{}
	for _, p := range procs {
		if p.PPID != nil {
			out[*p.PPID] = append(out[*p.PPID], p.Name)
		}
	}
	return out
}

// ObserveSprawl summarizes console-prone topology without deciding to kill
// anything. This is the trend signal for recurring window spam.
func ObserveSprawl(procs []Proc, top RelationTopology, opt OrphanOptions) ObservationSummary {
	opt = normalizeOrphanOptions(opt)
	out := ObservationSummary{
		RelationScanned:      len(procs),
		ConsoleProneByName:   map[string]int{},
		ConsoleProneByParent: map[string]int{},
	}
	for _, p := range procs {
		stem := stemLower(p.Name)
		aged := opt.MinAgeSec <= 0 || (p.AgeSec != nil && *p.AgeSec >= opt.MinAgeSec)
		if opt.ConsoleProneNames[stem] {
			out.ConsoleProneCount++
			out.ConsoleProneByName[stem]++
			parent := parentName(p, top)
			if parent == "" {
				parent = "unknown"
			}
			out.ConsoleProneByParent[parent]++
			if !attendedParent(p, top, opt.InteractiveParentNames) {
				out.BackgroundConsoleProneCount++
			}
		}
		if opt.ConsoleHostChildNames[stem] {
			out.ConsoleHostCount++
		}
		if opt.IdleShellNames[stem] && top.ChildCounts[p.PID] == 0 && !attendedParent(p, top, opt.InteractiveParentNames) {
			out.IdleShellCandidates++
			if aged {
				out.AgedIdleShellCandidates++
			}
		}
		children := childNameStems(top.ChildNames[p.PID])
		if opt.OrphanConsoleShellNames[stem] && !ownerAlive(p.PPID, top.LivePIDs) && onlyConsoleChildren(children, opt.ConsoleHostChildNames) {
			out.OrphanConsoleShellCandidates++
			if aged {
				out.AgedOrphanConsoleShellCandidates++
			}
		}
	}
	if len(out.ConsoleProneByName) == 0 {
		out.ConsoleProneByName = nil
	}
	if len(out.ConsoleProneByParent) == 0 {
		out.ConsoleProneByParent = nil
	}
	return out
}

// Build merges the resource-level findings with the orphan rows, applies the
// cross-tick CPU streak ledger and the Enact reaper gates, and returns the full
// control-pane Payload — the Go mirror of Python build_payload().
func Build(procs []Proc, opt Options) Payload {
	th := opt.Thresholds
	if th == (Thresholds{}) {
		th = DefaultThresholds()
	}
	confirm := opt.CPUReapConfirm
	if confirm < 1 {
		confirm = 1
	}
	resource := Classify(procs, th, opt.ProtectedPIDs, opt.AllowNames)
	flagged := mergeFlagged(resource, opt.OrphanRows)

	// Cross-tick streak ledger: bump every (pid+start) key CPU-flagged THIS run,
	// drop the rest. Keyed by start time so a recycled pid cannot inherit a streak.
	cpuKeys := []string{}
	for _, r := range flagged {
		if anyCPUReason(r.Reasons) {
			cpuKeys = append(cpuKeys, StreakKey(r.PID, r.Start))
		}
	}
	cpuStreaks := bumpStreaks(opt.CPUStreaksPrev, cpuKeys)

	enacted := []Enacted{}
	for i := range flagged {
		row := &flagged[i]
		isCPU := anyCPUReason(row.Reasons)
		if isCPU {
			s := cpuStreaks[StreakKey(row.PID, row.Start)]
			row.CPUStreak = &s
		}
		if !(opt.Enact && opt.Killer != nil) {
			row.Action = "report"
			continue
		}
		if row.Protected {
			row.Action = "protected-skip"
			continue
		}
		if cpuOnly(*row) {
			streak := 0
			if row.CPUStreak != nil {
				streak = *row.CPUStreak
			}
			if streak < confirm {
				// A core-pin not yet confirmed across enough runs — surfaced (still
				// ACTION) but NOT killed: the gate that keeps a legitimate minutes-
				// long CPU job from being reaped as if it were a wedged loop.
				row.Action = "cpu-unconfirmed"
				continue
			}
		}
		ok, detail := opt.Killer(row.PID)
		if ok {
			row.Action = "killed"
		} else {
			row.Action = "kill-failed"
		}
		enacted = append(enacted, Enacted{PID: row.PID, Name: row.Name, OK: ok, Detail: detail})
	}

	// ACTION (ok:false) iff a collector failed OR a NON-PROTECTED process is flagged.
	// A protected breach is still listed (and logged) but never flips the ok bit.
	actionable := 0
	for _, r := range flagged {
		if !r.Protected {
			actionable++
		}
	}
	collectError := strings.TrimSpace(opt.CollectError)
	ok := collectError == "" && actionable == 0
	return Payload{
		Schema:                 Schema,
		OK:                     ok,
		Platform:               opt.Platform,
		Thresholds:             th,
		CPUReapConfirm:         confirm,
		CPUStreaks:             cpuStreaks,
		Scanned:                len(procs),
		FlaggedCount:           len(flagged),
		ActionableFlaggedCount: actionable,
		Flagged:                flagged,
		Enacted:                enacted,
		Enact:                  opt.Enact,
		Observations:           opt.Observations,
		CollectError:           collectError,
		NextAction:             NextAction(flagged, opt.Enact, collectError),
	}
}

// NextAction renders the operator one-liner the control pane shows, matching the
// Python _next_action() branch set.
func NextAction(flagged []Finding, enact bool, collectError string) string {
	if strings.TrimSpace(collectError) != "" {
		return "process scan failed; rerun the guard and inspect collect_error"
	}
	if len(flagged) == 0 {
		return "no runaway or orphaned process; no action"
	}
	names := dedupNames(flagged, func(Finding) bool { return true })
	if enact {
		killed := dedupNames(flagged, func(r Finding) bool { return r.Action == "killed" })
		deferred := dedupNames(flagged, func(r Finding) bool { return r.Action == "cpu-unconfirmed" })
		parts := []string{}
		if len(killed) > 0 {
			parts = append(parts, "reaped: "+strings.Join(killed, ", "))
		}
		if len(deferred) > 0 {
			parts = append(parts, "CPU pin watched (not yet confirmed across runs, NOT reaped): "+strings.Join(deferred, ", "))
		}
		if len(parts) == 0 {
			return "flagged: " + strings.Join(names, ", ") + "; nothing reaped (protected or unconfirmed)"
		}
		return strings.Join(parts, "; ") + " (protected ones skipped)"
	}
	onlySprawl := true
	for _, r := range flagged {
		if r.Kind != "orphan-helper" && r.Kind != "idle-shell" && r.Kind != "orphan-console-shell" {
			onlySprawl = false
			break
		}
	}
	hint := "Inspect, then re-run with --enact to kill, or fix the launcher (bound -t/--threads on inference binaries)."
	if onlySprawl {
		hint = "orphaned sprawl serving no live session; re-run with --enact to reap."
	}
	return "flagged: " + strings.Join(names, ", ") + ". " + hint
}

// StreakKey is the reuse-safe cross-run identity for a process: pid PLUS its start
// time, so a recycled pid (different start) gets a FRESH streak.
func StreakKey(pid int, start string) string {
	return fmt.Sprintf("%d:%s", pid, start)
}

// CPUPctDelta is the per-core (top-style) CPU% over one window: 100 == one full
// core saturated. Returns (0,false) when a sample is missing or the counter went
// backwards (a backward delta means the PID was reused — a missed flag, never a
// wrong kill).
func CPUPctDelta(before, after, dt float64, haveBefore, haveAfter bool) (float64, bool) {
	if !haveBefore || !haveAfter || dt <= 0 {
		return 0, false
	}
	delta := after - before
	if delta < 0 {
		return 0, false
	}
	return (delta / dt) * 100.0, true
}

// CPUPctSustained returns the sustained per-core CPU% per pid: the MINIMUM window-%
// across consecutive samples. Taking the minimum (not the mean) is what makes the
// signal a *pin* and not a *burst* — a process must stay over the threshold in
// EVERY window. Each sample maps pid -> cumulative CPU-seconds at one instant, dt
// apart. A pid missing from any sample, or whose counter went backwards, is omitted.
func CPUPctSustained(samples []map[int]float64, dt float64) map[int]float64 {
	if len(samples) < 2 || dt <= 0 {
		return map[int]float64{}
	}
	pids := map[int]bool{}
	for _, s := range samples {
		for pid := range s {
			pids[pid] = true
		}
	}
	out := map[int]float64{}
	for pid := range pids {
		windows := []float64{}
		ok := true
		for i := 0; i+1 < len(samples); i++ {
			b, hb := samples[i][pid]
			a, ha := samples[i+1][pid]
			pct, valid := CPUPctDelta(b, a, dt, hb, ha)
			if !valid {
				ok = false
				break
			}
			windows = append(windows, pct)
		}
		if ok && len(windows) > 0 {
			min := windows[0]
			for _, w := range windows[1:] {
				if w < min {
					min = w
				}
			}
			out[pid] = min
		}
	}
	return out
}

// --------------------------------------------------------------------------- //
// internal helpers
// --------------------------------------------------------------------------- //

func toDispatchProcs(procs []Proc) []dispatchtick.ProcInfo {
	out := make([]dispatchtick.ProcInfo, 0, len(procs))
	for _, p := range procs {
		out = append(out, dispatchtick.ProcInfo{
			PID: p.PID, Name: p.Name, Threads: p.Threads, Handles: p.Handles, WorkingSetMB: p.WSMB,
		})
	}
	return out
}

// mergeFlagged unions resource and orphan rows by pid: a process can breach a
// resource threshold AND be orphaned — concat reasons, OR the protected bit, keep
// one row. Re-sorts CPU-first then thread-count to match the Python ordering.
func mergeFlagged(resourceRows, orphanRows []Finding) []Finding {
	byPID := map[int]int{} // pid -> index in out
	out := []Finding{}
	for _, row := range append(append([]Finding{}, resourceRows...), orphanRows...) {
		if idx, ok := byPID[row.PID]; ok {
			tgt := &out[idx]
			for _, r := range row.Reasons {
				if !contains(tgt.Reasons, r) {
					tgt.Reasons = append(tgt.Reasons, r)
				}
			}
			tgt.Protected = tgt.Protected || row.Protected
			if tgt.Kind == "" {
				tgt.Kind = row.Kind
			}
			continue
		}
		byPID[row.PID] = len(out)
		out = append(out, row)
	}
	sortFindings(out)
	return out
}

// sortFindings surfaces the loudest signal first: a CPU pin (a live core-burner)
// outranks a high static thread count for operator attention, then thread count
// breaks ties.
func sortFindings(rows []Finding) {
	sort.SliceStable(rows, func(i, j int) bool {
		ci, cj := f64(rows[i].CPUPct), f64(rows[j].CPUPct)
		if ci != cj {
			return ci > cj
		}
		return iv(rows[i].Threads) > iv(rows[j].Threads)
	})
}

func cpuOnly(row Finding) bool {
	if len(row.Reasons) == 0 || row.Kind != "" {
		return false
	}
	for _, r := range row.Reasons {
		if !isCPUReason(r) {
			return false
		}
	}
	return true
}

func bumpStreaks(prev map[string]int, cpuKeys []string) map[string]int {
	out := map[string]int{}
	for _, key := range cpuKeys {
		out[key] = prev[key] + 1
	}
	return out
}

func ownerAlive(ppid *int, livePIDs map[int]bool) bool {
	return ppid != nil && *ppid > 1 && livePIDs[*ppid]
}

func normalizeOrphanOptions(opt OrphanOptions) OrphanOptions {
	if opt.Patterns == nil {
		opt.Patterns = DefaultOrphanPatterns
	}
	if opt.IdleShellNames == nil {
		opt.IdleShellNames = DefaultIdleShellNames
	}
	if opt.OrphanConsoleShellNames == nil {
		opt.OrphanConsoleShellNames = DefaultOrphanConsoleShellNames
	}
	if opt.ConsoleHostChildNames == nil {
		opt.ConsoleHostChildNames = DefaultConsoleHostChildNames
	}
	if opt.InteractiveParentNames == nil {
		opt.InteractiveParentNames = DefaultInteractiveParentNames
	}
	if opt.ConsoleProneNames == nil {
		opt.ConsoleProneNames = DefaultConsoleProneNames
	}
	return opt
}

func attendedParent(p Proc, top RelationTopology, interactive map[string]bool) bool {
	parent := parentName(p, top)
	return parent != "" && interactive[stemLower(parent)]
}

func parentName(p Proc, top RelationTopology) string {
	if p.PPID == nil {
		return ""
	}
	return top.ParentNames[*p.PPID]
}

func childNameStems(names []string) []string {
	out := make([]string, 0, len(names))
	for _, n := range names {
		if stem := stemLower(n); stem != "" {
			out = append(out, stem)
		}
	}
	sort.Strings(out)
	return out
}

func onlyConsoleChildren(children []string, consoleHosts map[string]bool) bool {
	for _, child := range children {
		if !consoleHosts[child] {
			return false
		}
	}
	return true
}

func stemLower(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	if strings.HasSuffix(name, ".exe") {
		name = strings.TrimSuffix(name, ".exe")
	}
	return name
}

func anyCPUReason(reasons []string) bool {
	for _, r := range reasons {
		if isCPUReason(r) {
			return true
		}
	}
	return false
}

func isCPUReason(r string) bool { return strings.HasPrefix(r, "cpu ") }

func dedupNames(rows []Finding, keep func(Finding) bool) []string {
	set := map[string]bool{}
	for _, r := range rows {
		if !keep(r) {
			continue
		}
		set[fmt.Sprintf("%s(pid %d)", r.Name, r.PID)] = true
	}
	names := make([]string, 0, len(set))
	for n := range set {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func lowerSet(names []string) map[string]bool {
	out := map[string]bool{}
	for _, n := range names {
		n = strings.ToLower(strings.TrimSpace(n))
		if n != "" {
			out[n] = true
		}
	}
	return out
}

func intSet(pids []int) map[int]bool {
	out := map[int]bool{}
	for _, p := range pids {
		if p > 0 {
			out[p] = true
		}
	}
	return out
}

func nonEmpty(in []string) []string {
	out := []string{}
	for _, s := range in {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if h == needle {
			return true
		}
	}
	return false
}

func iv(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func f64(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}
