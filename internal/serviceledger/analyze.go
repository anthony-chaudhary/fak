package serviceledger

import (
	"fmt"
	"io"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// StaleOwner names an owner (manager invocation / PID) that kept reporting
// under a generation an EventLeaseFence had already fenced — the
// "stale-but-still-running" hazard the issue calls out.
type StaleOwner struct {
	ManagerInvocation string `json:"manager_invocation,omitempty"`
	PID               int    `json:"pid,omitempty"`
	Generation        int64  `json:"generation"`
	FencedGeneration  int64  `json:"fenced_generation"`
	LastAtUnixMS      int64  `json:"last_at_unix_ms"`
}

// WorkloadStatus is the correlated rollup for one (node, service, workload).
type WorkloadStatus struct {
	Identity         servicespec.Identity     `json:"identity"`
	Phase            servicespec.Phase        `json:"phase"`
	Desired          servicespec.DesiredState `json:"desired,omitempty"`
	Generation       int64                    `json:"generation,omitempty"`
	BootID           string                   `json:"boot_id,omitempty"`
	LastExit         *servicespec.ExitRecord  `json:"last_exit,omitempty"`
	LastEventUnixMS  int64                    `json:"last_event_unix_ms"`
	Events           int                      `json:"events"`
	RestartsInWindow int                      `json:"restarts_in_window"`
	RestartStorm     bool                     `json:"restart_storm"`
	StaleOwners      []StaleOwner             `json:"stale_owners,omitempty"`
}

// StatusOptions bounds the restart-storm detector. Zero values take the
// servicespec restart-policy defaults; NowUnixMS defaults to the newest event.
type StatusOptions struct {
	WindowMS    int64
	MaxRestarts int
	NowUnixMS   int64
}

// phaseOf derives the observed phase an event implies, or "" when the event
// does not move the phase axis (desired/checkpoint/resume/lease bookkeeping).
func phaseOf(e Event) servicespec.Phase {
	switch e.Type {
	case EventReadiness:
		return e.Phase
	case EventProcessExit:
		if e.Exit != nil && (e.Exit.Class == servicespec.ExitOperatorStop || e.Exit.Class == servicespec.ExitClean) {
			return servicespec.PhaseStopped
		}
		return servicespec.PhaseFailed
	case EventWatchdogTimeout:
		return servicespec.PhaseDegraded
	case EventManagerRestart:
		return servicespec.PhaseStarting
	case EventBootChange:
		return servicespec.PhaseUnknown
	case EventCircuitOpen:
		return servicespec.PhaseFenced
	}
	return ""
}

// restartWorthy reports whether an event counts toward restart-storm detection.
func restartWorthy(e Event) bool {
	if e.Type == EventWatchdogTimeout {
		return true
	}
	return e.Type == EventProcessExit && e.Exit != nil &&
		(e.Exit.Class == servicespec.ExitCrash || e.Exit.Class == servicespec.ExitWatchdog)
}

func workloadKey(id servicespec.Identity) string {
	w := id.Workload
	if w == "" {
		w = id.Service
	}
	return id.Node + "\x00" + id.Service + "\x00" + w
}

// Status folds a timeline into per-workload rollups: last observed phase,
// correlation high-water marks, restart-storm verdict, and stale owners.
func Status(events []Event, opt StatusOptions) []WorkloadStatus {
	if opt.WindowMS <= 0 {
		opt.WindowMS = servicespec.DefaultWindowMS
	}
	if opt.MaxRestarts <= 0 {
		opt.MaxRestarts = servicespec.DefaultWindowMaxRestarts
	}
	sorted := sortedByTime(events)
	if opt.NowUnixMS <= 0 && len(sorted) > 0 {
		opt.NowUnixMS = sorted[len(sorted)-1].AtUnixMS
	}
	type acc struct {
		st        WorkloadStatus
		fencedGen int64
		stale     map[string]*StaleOwner
		restarts  []int64
	}
	byKey := map[string]*acc{}
	var order []string
	for _, e := range sorted {
		key := workloadKey(e.Identity)
		a := byKey[key]
		if a == nil {
			id := e.Identity
			if id.Workload == "" {
				id.Workload = id.Service
			}
			a = &acc{st: WorkloadStatus{Identity: id, Phase: servicespec.PhaseUnknown}, stale: map[string]*StaleOwner{}}
			byKey[key] = a
			order = append(order, key)
		}
		a.st.Events++
		a.st.LastEventUnixMS = e.AtUnixMS
		if p := phaseOf(e); p != "" {
			a.st.Phase = p
		}
		if e.Type == EventDesiredChange {
			a.st.Desired = e.Desired
		}
		if e.Type == EventProcessExit && e.Exit != nil {
			cp := *e.Exit
			a.st.LastExit = &cp
		}
		if e.Correlation.BootID != "" {
			a.st.BootID = e.Correlation.BootID
		}
		if g := e.Correlation.Generation; g > a.st.Generation {
			a.st.Generation = g
		}
		if e.Type == EventLeaseFence && e.Correlation.Generation > a.fencedGen {
			a.fencedGen = e.Correlation.Generation
		} else if g := e.Correlation.Generation; a.fencedGen > 0 && g > 0 && g < a.fencedGen {
			// An already-fenced generation is still emitting events: a stale
			// owner survived its fence.
			sk := fmt.Sprintf("%s\x00%d\x00%d", e.Correlation.ManagerInvocation, e.Correlation.PID, g)
			so := a.stale[sk]
			if so == nil {
				so = &StaleOwner{ManagerInvocation: e.Correlation.ManagerInvocation, PID: e.Correlation.PID, Generation: g}
				a.stale[sk] = so
			}
			so.FencedGeneration = a.fencedGen
			if e.AtUnixMS > so.LastAtUnixMS {
				so.LastAtUnixMS = e.AtUnixMS
			}
		}
		if restartWorthy(e) {
			a.restarts = append(a.restarts, e.AtUnixMS)
		}
	}
	out := make([]WorkloadStatus, 0, len(order))
	for _, key := range order {
		a := byKey[key]
		for _, ts := range a.restarts {
			if ts > opt.NowUnixMS-opt.WindowMS && ts <= opt.NowUnixMS {
				a.st.RestartsInWindow++
			}
		}
		a.st.RestartStorm = a.st.RestartsInWindow >= opt.MaxRestarts
		for _, so := range a.stale {
			a.st.StaleOwners = append(a.st.StaleOwners, *so)
		}
		sort.Slice(a.st.StaleOwners, func(i, j int) bool {
			si, sj := a.st.StaleOwners[i], a.st.StaleOwners[j]
			if si.Generation != sj.Generation {
				return si.Generation < sj.Generation
			}
			return si.PID < sj.PID
		})
		out = append(out, a.st)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i].Identity, out[j].Identity
		if a.Node != b.Node {
			return a.Node < b.Node
		}
		if a.Service != b.Service {
			return a.Service < b.Service
		}
		return a.Workload < b.Workload
	})
	return out
}

func sortedByTime(events []Event) []Event {
	out := make([]Event, len(events))
	copy(out, events)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].AtUnixMS != out[j].AtUnixMS {
			return out[i].AtUnixMS < out[j].AtUnixMS
		}
		return out[i].Seq < out[j].Seq
	})
	return out
}

func stampUTC(ms int64) string {
	return time.UnixMilli(ms).UTC().Format("2006-01-02T15:04:05.000Z")
}

// WriteTimeline writes the concise human timeline: one deterministic line per
// event, time-ordered, causality readable top to bottom.
func WriteTimeline(w io.Writer, events []Event) {
	for _, e := range sortedByTime(events) {
		id := e.Identity.Node + "/" + e.Identity.Service
		if e.Identity.Workload != "" && e.Identity.Workload != e.Identity.Service {
			id += "/" + e.Identity.Workload
		}
		line := fmt.Sprintf("%s %-22s %s", stampUTC(e.AtUnixMS), id, e.Type)
		switch e.Type {
		case EventDesiredChange:
			line += " desired=" + string(e.Desired)
		case EventReadiness:
			line += " phase=" + string(e.Phase)
		case EventProcessExit:
			if e.Exit != nil {
				line += fmt.Sprintf(" class=%s code=%d", e.Exit.Class, e.Exit.Code)
			}
		}
		if c := e.Correlation; true {
			if c.Generation > 0 {
				line += fmt.Sprintf(" gen=%d", c.Generation)
			}
			if c.BootID != "" {
				line += " boot=" + c.BootID
			}
			if c.PID > 0 {
				line += fmt.Sprintf(" pid=%d", c.PID)
			}
			if c.Checkpoint != "" {
				line += " checkpoint=" + c.Checkpoint
			}
			if c.Receipt != "" {
				line += " receipt=" + c.Receipt
			}
			if c.Session != "" {
				line += " session=" + c.Session
			}
		}
		line += " [" + e.Source + "]"
		if e.Detail != "" {
			line += " " + fmt.Sprintf("%q", e.Detail)
		}
		fmt.Fprintln(w, line)
	}
}

// WriteStatus writes the concise human per-workload rollup.
func WriteStatus(w io.Writer, sts []WorkloadStatus) {
	for _, st := range sts {
		id := st.Identity.Node + "/" + st.Identity.Service
		if st.Identity.Workload != "" && st.Identity.Workload != st.Identity.Service {
			id += "/" + st.Identity.Workload
		}
		line := fmt.Sprintf("%-30s phase=%s", id, st.Phase)
		if st.Desired != "" {
			line += " " + string(st.Desired)
		}
		if st.Generation > 0 {
			line += fmt.Sprintf(" gen=%d", st.Generation)
		}
		if st.BootID != "" {
			line += " boot=" + st.BootID
		}
		if st.LastExit != nil {
			line += fmt.Sprintf(" last-exit=%s", st.LastExit.Class)
		}
		line += fmt.Sprintf(" events=%d restarts-in-window=%d", st.Events, st.RestartsInWindow)
		if st.RestartStorm {
			line += " RESTART-STORM"
		}
		for _, so := range st.StaleOwners {
			line += fmt.Sprintf(" STALE-OWNER(gen=%d<%d pid=%d)", so.Generation, so.FencedGeneration, so.PID)
		}
		fmt.Fprintln(w, line)
	}
}
