package toolprocgate

// Blast-radius containment (#2170 protection, sibling to the console-fault
// OBSERVABILITY): ExitConsoleFault already scopes a Windows console/shell/PTY
// crash to the DEAD call — sibling procs are untouched. What it does NOT do is
// stop the NEXT spawn from walking into the same crashing surface, cap how many
// agents one console crash can take down, or trip a fleet-wide hold when faults
// storm across sessions. Those are the cascade paths a single terminal crash
// uses to bring down multiple agents or the whole host.
//
// This is the admission-side guard that closes those paths. It is a PURE fold
// over the console-fault history the supervisor already records, producing a
// closed containment verdict for a proposed spawn:
//
//   - REFUSE_COLOCATION  — the target surface already hosts the per-surface cap
//     of live agents; place the new agent elsewhere so ONE console crash can
//     take down at most MaxAgentsPerSurface agents, not the whole fleet.
//   - QUARANTINE_SURFACE — that surface has faulted repeatedly inside the
//     window; it is in a re-crash loop, so respawning onto it just crashes
//     again. Allocate a fresh console host instead.
//   - BREAKER_OPEN       — faults are storming across MULTIPLE sessions: the
//     host itself is unstable (a GPU reset, an update reboot, a ConPTY storm).
//     Hold ALL new spawns until it clears rather than feeding more agents into
//     a dying host.
//   - ADMIT              — none of the above; the spawn is safe.
//
// Protection-first: when several conditions hold, the MOST severe (widest
// blast radius) verdict wins. The fold reads no clock and holds no state, so
// the same history + request always yields the same verdict — offline-provable,
// and identical whether run live in Supervisor.AdmitSpawn or from the CLI gate.

import "sort"

// ContainmentVerdict is the CLOSED admission vocabulary. A spawn is admitted
// only on ContainAdmit; every other verdict is a protective refusal/hold.
type ContainmentVerdict string

const (
	// ContainAdmit: the spawn is safe to launch.
	ContainAdmit ContainmentVerdict = "ADMIT"
	// ContainRefuseColocation: the surface is at its live-agent cap — place the
	// new agent on a different surface to bound the blast radius of a crash.
	ContainRefuseColocation ContainmentVerdict = "REFUSE_COLOCATION"
	// ContainQuarantineSurface: the surface is in a re-crash loop — allocate a
	// fresh console host instead of respawning onto the broken one.
	ContainQuarantineSurface ContainmentVerdict = "QUARANTINE_SURFACE"
	// ContainBreakerOpen: a cross-session fault storm — hold all new spawns
	// until the host stabilizes.
	ContainBreakerOpen ContainmentVerdict = "BREAKER_OPEN"
)

// ContainmentPolicy bounds the blast radius of a console/terminal crash. The
// zero value is inert (nothing trips); DefaultContainmentPolicy has the wired
// defaults. An embedder tunes these — its fleet shape is its own policy choice,
// the same doctrine as the supervisor's tick cadence.
type ContainmentPolicy struct {
	// WindowMS is the lookback for counting recent faults. A fault older than
	// this is treated as cleared. NowMS<=0 disables the window (all faults
	// count) — the conservative, more-protective reading when the clock is
	// unknown.
	WindowMS int64
	// MaxAgentsPerSurface caps live agents co-located on one console surface, so
	// a single crash can take down at most this many. 0 disables the cap.
	MaxAgentsPerSurface int
	// SurfaceQuarantineFaults is how many faults on ONE surface inside the
	// window mark it a re-crash loop. 0 disables surface quarantine.
	SurfaceQuarantineFaults int
	// BreakerFaults is the total fault count inside the window that, combined
	// with BreakerSessions, opens the fleet breaker. 0 disables the breaker.
	BreakerFaults int
	// BreakerSessions is how many DISTINCT sessions must have faulted inside the
	// window for the storm to count as a cross-session cascade (a single noisy
	// session is not a fleet emergency). <=1 means any session count qualifies.
	BreakerSessions int
}

// DefaultContainmentPolicy is the wired protection: a crash contains to at most
// 3 agents per surface, a surface that faults twice in 5 minutes is quarantined,
// and 5+ faults across 3+ sessions in 5 minutes opens the fleet breaker.
func DefaultContainmentPolicy() ContainmentPolicy {
	return ContainmentPolicy{
		WindowMS:                5 * 60 * 1000,
		MaxAgentsPerSurface:     3,
		SurfaceQuarantineFaults: 2,
		BreakerFaults:           5,
		BreakerSessions:         3,
	}
}

// ContainmentRequest is the proposed spawn the guard adjudicates.
type ContainmentRequest struct {
	// Surface is the console host the new agent would run on.
	Surface string
	// LiveOnSurface is how many agents are already live on Surface (the fan-in
	// the caller knows from its placement table).
	LiveOnSurface int
	// NowMS anchors the fault window.
	NowMS int64
}

// ContainmentDecision is the closed verdict plus the evidence that produced it,
// so a refusal is auditable, not opaque.
type ContainmentDecision struct {
	Verdict ContainmentVerdict `json:"verdict"`
	// Admit is the one-bit gate: true only on ContainAdmit.
	Admit  bool   `json:"admit"`
	Reason string `json:"reason"`
	Advice string `json:"advice"`
	// Evidence.
	SurfaceFaults  int `json:"surface_faults"`  // faults on the requested surface, in window
	WindowFaults   int `json:"window_faults"`   // total faults in window
	WindowSessions int `json:"window_sessions"` // distinct sessions that faulted, in window
	LiveOnSurface  int `json:"live_on_surface"` // echoed fan-in
}

// inWindow reports whether a fault at atMS falls inside [now-window, now].
// A non-positive now or window disables the filter (everything counts) — the
// conservative reading when the clock is unavailable.
func inWindow(atMS, nowMS, windowMS int64) bool {
	if nowMS <= 0 || windowMS <= 0 {
		return true
	}
	return atMS >= nowMS-windowMS && atMS <= nowMS
}

// DecideContainment folds the recorded console faults into a containment verdict
// for req. Protection-first: the widest-blast-radius condition that holds wins
// (breaker > surface quarantine > co-location cap > admit).
func DecideContainment(pol ContainmentPolicy, faults []ConsoleFaultEvent, req ContainmentRequest) ContainmentDecision {
	sessions := map[string]struct{}{}
	windowFaults, surfaceFaults := 0, 0
	for _, f := range faults {
		if !inWindow(f.AtMS, req.NowMS, pol.WindowMS) {
			continue
		}
		windowFaults++
		if f.Session != "" {
			sessions[f.Session] = struct{}{}
		}
		if req.Surface != "" && f.Surface == req.Surface {
			surfaceFaults++
		}
	}
	dec := ContainmentDecision{
		SurfaceFaults:  surfaceFaults,
		WindowFaults:   windowFaults,
		WindowSessions: len(sessions),
		LiveOnSurface:  req.LiveOnSurface,
	}

	// 1) Fleet breaker: a cross-session fault storm means the host is unstable.
	if pol.BreakerFaults > 0 && windowFaults >= pol.BreakerFaults && len(sessions) >= max1(pol.BreakerSessions) {
		dec.Verdict = ContainBreakerOpen
		dec.Reason = "cross-session console-fault storm: the host is unstable"
		dec.Advice = "hold all new spawns until the fault window clears; do not feed more agents into a crashing host"
		return dec
	}
	// 2) Surface quarantine: this console is in a re-crash loop.
	if pol.SurfaceQuarantineFaults > 0 && req.Surface != "" && surfaceFaults >= pol.SurfaceQuarantineFaults {
		dec.Verdict = ContainQuarantineSurface
		dec.Reason = "surface faulted repeatedly in the window: re-crash loop"
		dec.Advice = "allocate a fresh console host for this agent; do not respawn onto the quarantined surface"
		return dec
	}
	// 3) Co-location cap: bound how many agents one crash can take down.
	if pol.MaxAgentsPerSurface > 0 && req.LiveOnSurface >= pol.MaxAgentsPerSurface {
		dec.Verdict = ContainRefuseColocation
		dec.Reason = "surface at its live-agent cap: a crash here would exceed the blast-radius bound"
		dec.Advice = "place this agent on a different console surface so one crash cannot cascade past the cap"
		return dec
	}
	dec.Verdict = ContainAdmit
	dec.Admit = true
	dec.Reason = "no active fault storm, quarantine, or co-location pressure"
	return dec
}

// max1 floors a breaker-session threshold at 1: a zero/negative BreakerSessions
// means "any session count qualifies" rather than "never".
func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// ContainmentSurfaceLoad is a per-surface fan-in count, sorted for a stable
// operator view of where agents are concentrated (the blast-radius map).
type ContainmentSurfaceLoad struct {
	Surface string `json:"surface"`
	Live    int    `json:"live"`
}

// SortSurfaceLoads returns loads sorted by descending fan-in then surface, so
// the most-concentrated (highest-risk) surface reads first.
func SortSurfaceLoads(loads []ContainmentSurfaceLoad) []ContainmentSurfaceLoad {
	out := append([]ContainmentSurfaceLoad(nil), loads...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Live != out[j].Live {
			return out[i].Live > out[j].Live
		}
		return out[i].Surface < out[j].Surface
	})
	return out
}
