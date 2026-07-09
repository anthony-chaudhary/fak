package looporphan

import (
	"sort"
	"strings"
)

// ParentState is the tri-state liveness of a supervisor's owning session/parent,
// mirrored from looprecover.Probe's verdict. The distinction between Dead and
// Unknown is load-bearing: an orphan (parent Dead) is reap-eligible, but an
// unconfirmed parent (Unknown) is not - the core fails closed rather than reap on
// a guess.
type ParentState string

const (
	ParentAlive   ParentState = "alive"
	ParentDead    ParentState = "dead"
	ParentUnknown ParentState = "unknown"
)

// Supervisor is one candidate loop/drainer engine process, already matched by
// command-line marker by the shell. The core folds a census of these; it holds no
// process handle and performs no I/O.
type Supervisor struct {
	PID         int         // the supervisor process id
	PPID        int         // its parent (the owning session/loop driver), 0 if unknown
	Start       string      // start-time fence (procguard.Proc.Start); "" = unfenced
	Cmdline     string      // raw command line (identity fallback when Lane is "")
	Lane        string      // parsed loop identity (--lane / goal id); "" if unknown
	Parent      ParentState // liveness of the owning session/parent (looprecover.Probe)
	LiveWorkers int         // count of live `fak c` / `claude -p` workers in the subtree
}

// Action is the closed verdict vocabulary. REAP is a RECOMMENDATION only - the
// core never kills; the shell acts on REAP behind an explicit opt-in.
type Action string

const (
	KEEP      Action = "KEEP"      // canonical / live / attached - do not touch
	REAP      Action = "REAP"      // orphaned-or-duplicate AND idle - safe to tree-kill
	COLLISION Action = "COLLISION" // 2+ live supervisors for one lane - operator decides
	UNKNOWN   Action = "UNKNOWN"   // insufficient evidence - fail closed, never reap
)

// Reason tokens - a closed set, one per (action, cause). The shell surfaces these
// verbatim; they are the refusal/decision vocabulary for `fak loop reap`.
const (
	ReasonKeepLiveWork  = "LOOP_KEEP_LIVE_WORK" // KEEP: parents the live worker
	ReasonKeepAttached  = "LOOP_KEEP_ATTACHED"  // KEEP: parent/session alive, idle between workers
	ReasonOrphanIdle    = "LOOP_ORPHAN_IDLE"    // REAP: parent gone, no live work
	ReasonDupIdle       = "LOOP_DUP_IDLE"       // REAP: idle duplicate of a lane that has a keeper
	ReasonCollisionLive = "LOOP_COLLISION_LIVE" // COLLISION: 2+ live supervisors, one lane
	ReasonNoFence       = "LOOP_NO_FENCE"       // UNKNOWN: reap-eligible but no start-time fence
	ReasonEvidenceThin  = "LOOP_EVIDENCE_THIN"  // UNKNOWN: no lane/cmdline identity, or parent unknown
)

// Verdict is the per-supervisor decision.
type Verdict struct {
	PID       int
	Lane      string
	Group     string // the group key the supervisor was folded under
	GroupSize int    // supervisors sharing that key
	Action    Action
	Reason    string // one closed reason token
	Detail    string // human-readable one-liner
}

// Report is the fleet-level fold: every verdict plus per-action counts.
type Report struct {
	Verdicts  []Verdict
	Keep      int
	Reap      int
	Collision int
	Unknown   int
}

// Config carries the (few) policy knobs. The zero value is the safe default.
type Config struct {
	// AllowUnfencedReap lifts the start-time-fence guard: when true, a
	// reap-eligible supervisor with an empty Start is still REAP rather than
	// downgraded to UNKNOWN. Off by default - an unfenced PID cannot be checked
	// for reuse, so the core refuses to recommend killing it.
	AllowUnfencedReap bool
}

// DefaultConfig is the safe, fail-closed configuration.
func DefaultConfig() Config { return Config{} }

// groupKey is a supervisor's dedup identity: its Lane if known, else its trimmed
// command line. Empty means no identity at all.
func groupKey(s Supervisor) string {
	if k := strings.TrimSpace(s.Lane); k != "" {
		return k
	}
	return strings.TrimSpace(s.Cmdline)
}

// fenced reports whether a supervisor may be recommended for REAP given the
// PID-reuse guard: a non-empty Start, or an explicit operator override.
func fenced(s Supervisor, cfg Config) bool {
	return cfg.AllowUnfencedReap || strings.TrimSpace(s.Start) != ""
}

// reapOr returns a REAP verdict for a reap-eligible supervisor, downgraded to
// UNKNOWN/LOOP_NO_FENCE when it lacks a start-time fence.
func reapOr(s Supervisor, key string, size int, reason, detail string, cfg Config) Verdict {
	if !fenced(s, cfg) {
		return Verdict{
			PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
			Action: UNKNOWN, Reason: ReasonNoFence,
			Detail: "reap-eligible (" + reason + ") but no start-time fence; refusing to recommend a kill it cannot check for PID reuse",
		}
	}
	return Verdict{
		PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
		Action: REAP, Reason: reason, Detail: detail,
	}
}

// idleDupVerdict decides one idle (no-live-worker) member of a lane group that
// already has a live keeper. Crucially it consults Parent rather than reaping on
// the group membership alone: a parsed lane is a fallback identity and CAN collide
// (two different loops both launched --region billing, or empty-lane cmdline
// twins), so an idle member whose own owning parent is still alive is an attached
// idle loop - NOT dead weight - and is KEPT. Only a confirmed orphan (parent dead)
// is a reapable duplicate; an unknown parent fails closed. Without this, a
// live-but-idle loop that merely shares a lane string with an unrelated live loop
// would be wrongly reaped.
func idleDupVerdict(s Supervisor, key string, size int, dupDetail string, cfg Config) Verdict {
	switch s.Parent {
	case ParentAlive:
		return Verdict{
			PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
			Action: KEEP, Reason: ReasonKeepAttached,
			Detail: "attached idle loop (owning parent alive) sharing a lane with live work - keeping; not dead weight",
		}
	case ParentDead:
		return reapOr(s, key, size, ReasonDupIdle, dupDetail, cfg)
	default:
		return Verdict{
			PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
			Action: UNKNOWN, Reason: ReasonEvidenceThin,
			Detail: "idle duplicate of a kept lane, but parent liveness is unknown - cannot confirm orphan, so not reaping",
		}
	}
}

// Plan folds a supervisor census into a keep/reap Report. It is pure and
// deterministic: verdicts are ordered by group key, then PID.
func Plan(census []Supervisor, cfg Config) Report {
	// Bucket by dedup identity; empty-identity supervisors are decided in place
	// (fail closed) since they cannot be grouped.
	groups := map[string][]Supervisor{}
	var order []string
	var verdicts []Verdict

	for _, s := range census {
		key := groupKey(s)
		if key == "" {
			verdicts = append(verdicts, Verdict{
				PID: s.PID, Lane: s.Lane, Group: "", GroupSize: 1,
				Action: UNKNOWN, Reason: ReasonEvidenceThin,
				Detail: "no lane or command-line identity; cannot group or reap",
			})
			continue
		}
		if _, seen := groups[key]; !seen {
			order = append(order, key)
		}
		groups[key] = append(groups[key], s)
	}

	for _, key := range order {
		g := groups[key]
		size := len(g)
		var live, idle []Supervisor
		for _, s := range g {
			if s.LiveWorkers > 0 {
				live = append(live, s)
			} else {
				idle = append(idle, s)
			}
		}

		switch {
		case len(live) >= 2:
			// Genuine duplication over live work: reaping either strands a live
			// worker, so the live ones are an operator decision. Idle members are
			// unambiguous dead weight.
			for _, s := range live {
				verdicts = append(verdicts, Verdict{
					PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
					Action: COLLISION, Reason: ReasonCollisionLive,
					Detail: "two or more live supervisors share this lane; keeping all - reaping either would strand live work",
				})
			}
			for _, s := range idle {
				verdicts = append(verdicts, idleDupVerdict(s, key, size,
					"idle duplicate of a lane with live supervisors", cfg))
			}
		case len(live) == 1:
			s := live[0]
			verdicts = append(verdicts, Verdict{
				PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
				Action: KEEP, Reason: ReasonKeepLiveWork,
				Detail: "parents the live worker for this lane - the canonical keeper",
			})
			for _, s := range idle {
				verdicts = append(verdicts, idleDupVerdict(s, key, size,
					"idle duplicate of a lane whose live supervisor is kept", cfg))
			}
		default:
			// No live work anywhere in the group: decide each member by parent
			// liveness. Attached-idle loops are kept; confirmed orphans are reaped;
			// unconfirmed parents fail closed.
			for _, s := range g {
				switch s.Parent {
				case ParentAlive:
					verdicts = append(verdicts, Verdict{
						PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
						Action: KEEP, Reason: ReasonKeepAttached,
						Detail: "no live work now, but the owning session/parent is alive - an attached idle loop between workers",
					})
				case ParentDead:
					verdicts = append(verdicts, reapOr(s, key, size, ReasonOrphanIdle,
						"owning session/parent is gone and no live work in its subtree - an idle orphan", cfg))
				default:
					verdicts = append(verdicts, Verdict{
						PID: s.PID, Lane: s.Lane, Group: key, GroupSize: size,
						Action: UNKNOWN, Reason: ReasonEvidenceThin,
						Detail: "no live work, but parent liveness is unknown - cannot confirm orphan, so not reaping",
					})
				}
			}
		}
	}

	sort.SliceStable(verdicts, func(i, j int) bool {
		if verdicts[i].Group != verdicts[j].Group {
			return verdicts[i].Group < verdicts[j].Group
		}
		return verdicts[i].PID < verdicts[j].PID
	})

	rep := Report{Verdicts: verdicts}
	for _, v := range verdicts {
		switch v.Action {
		case KEEP:
			rep.Keep++
		case REAP:
			rep.Reap++
		case COLLISION:
			rep.Collision++
		case UNKNOWN:
			rep.Unknown++
		}
	}
	return rep
}

// ReapPIDs is a convenience: the PIDs the report recommends tree-killing, in
// verdict order. The shell passes these to procguard.KillPID behind an explicit
// operator opt-in.
func (r Report) ReapPIDs() []int {
	var out []int
	for _, v := range r.Verdicts {
		if v.Action == REAP {
			out = append(out, v.PID)
		}
	}
	return out
}
