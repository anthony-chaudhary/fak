package toolprocgate

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Supervisor is the seam-1 engine: the live, in-process side of the tool
// process table. An embedder (the gateway proxy, `fak guard`, the agent loop,
// the MCP server) reports lifecycle observations as they cross the wire —
// Spawn when an adjudicated call is dispatched (registering the cancel lever
// for the in-flight work), Pulse on any liveness signal, Exit on completion —
// and calls Tick on its cadence. Tick folds the journal through the SAME pure
// toolproc.Fold the offline CLI uses and then ACTS on the advice:
//
//   - kill / reap  → invoke the registered cancel lever (once), enter the call
//     into the revocation table (so a late completion is quarantined by the
//     rank-2 Gate), and append the kill event to the journal — the table and
//     the enforcement stay in one causal record;
//   - probe        → reported to the caller (a liveness probe is the
//     embedder's move: poll the job, nudge the stream), never destructive;
//   - quarantine_result → already enforced by the Gate on the admission path.
//
// The Supervisor holds no goroutine and reads no clock: the embedder supplies
// nowMS on every entry point, so behavior under test is deterministic and an
// embedder's tick cadence is its own policy choice.
type Supervisor struct {
	mu      sync.Mutex
	cfg     toolproc.Config
	events  []toolproc.Event
	cancels map[string]func() // callID -> cancel lever, cleared once fired or terminal
	spawned map[string]bool   // callIDs ever spawned (journal identity guard)
	pids    map[string]int    // callID -> bound OS process-tree root (seam 6), cleared with cancels
	reaper  OSReaper          // OS lever for bound pids; nil = advice-only (no teeth)
}

// TickAction is one enforcement act Tick performed or advised.
type TickAction struct {
	CallID string          `json:"call_id"`
	Reason string          `json:"reason"` // closed toolproc verdict token
	Advice toolproc.Advice `json:"advice"`
	// Cancelled reports the registered cancel lever was invoked this tick.
	// False for advisory findings (probe) and for kills whose call had no
	// registered lever (nothing in flight to cancel; revocation still applies).
	Cancelled bool `json:"cancelled"`
	// Reaped reports the bound OS process tree was terminated this tick (the
	// reaper ran AND reported success). False when no pid was bound, no reaper
	// is set, or the reaper failed — ReapDetail carries the reaper's message
	// either way it ran.
	Reaped     bool   `json:"reaped,omitempty"`
	ReapDetail string `json:"reap_detail,omitempty"`
}

// TickReport is what one Tick did: the folded table plus the acts.
type TickReport struct {
	Table   toolproc.Table `json:"table"`
	Actions []TickAction   `json:"actions,omitempty"`
}

// NewSupervisor builds a Supervisor with the given fold config (zero value =
// toolproc defaults: no default deadline, stall multiplier 3).
func NewSupervisor(cfg toolproc.Config) *Supervisor {
	return &Supervisor{
		cfg:     cfg,
		cancels: map[string]func(){},
		spawned: map[string]bool{},
		pids:    map[string]int{},
	}
}

// Spawn reports an adjudicated call went long / went live. cancel is the
// lever Tick pulls to abort the in-flight work (a context.CancelFunc, an
// http request abort, an MCP $/cancel emitter); nil means "observable but not
// cancellable" — advice still lands in the revocation table.
func (s *Supervisor) Spawn(callID, tool, session string, deadlineMS, heartbeatEveryMS, nowMS int64, cancel func()) error {
	ev := toolproc.Event{Kind: toolproc.EvSpawn, CallID: callID, Tool: tool, Session: session,
		AtMS: nowMS, DeadlineMS: deadlineMS, HeartbeatEveryMS: heartbeatEveryMS}
	if err := toolproc.ValidateEvent(ev); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.spawned[callID] {
		return fmt.Errorf("toolprocgate: duplicate spawn for call %s", callID)
	}
	s.spawned[callID] = true
	s.events = append(s.events, ev)
	if cancel != nil {
		s.cancels[callID] = cancel
	}
	return nil
}

// Pulse reports a liveness signal (heartbeat, output chunk, progress, poll).
// via optionally names the polling call's TraceID (launch↔poll correlation).
func (s *Supervisor) Pulse(callID string, nowMS int64, via string) error {
	return s.append(toolproc.Event{Kind: toolproc.EvPulse, CallID: callID, AtMS: nowMS, Via: via})
}

// Exit reports the call completed; status is "ok" or "error".
func (s *Supervisor) Exit(callID string, nowMS int64, status string) error {
	if err := s.append(toolproc.Event{Kind: toolproc.EvExit, CallID: callID, AtMS: nowMS, Status: status}); err != nil {
		return err
	}
	s.mu.Lock()
	delete(s.cancels, callID)
	delete(s.pids, callID)
	s.mu.Unlock()
	return nil
}

// SessionEnd reports the owning session ended (the orphan boundary).
func (s *Supervisor) SessionEnd(session string, nowMS int64) error {
	return s.append(toolproc.Event{Kind: toolproc.EvSessionEnd, Session: session, AtMS: nowMS})
}

func (s *Supervisor) append(ev toolproc.Event) error {
	if err := toolproc.ValidateEvent(ev); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if ev.CallID != "" && !s.spawned[ev.CallID] {
		return fmt.Errorf("toolprocgate: %s for unknown call %s", ev.Kind, ev.CallID)
	}
	s.events = append(s.events, ev)
	return nil
}

// Table folds the journal at nowMS without acting — the observability read.
func (s *Supervisor) Table(nowMS int64) (toolproc.Table, error) {
	s.mu.Lock()
	events := make([]toolproc.Event, len(s.events))
	copy(events, s.events)
	cfg := s.cfg
	s.mu.Unlock()
	return toolproc.Fold(events, nowMS, cfg)
}

// Tick folds the journal at nowMS and ENFORCES: kill and reap advice cancel
// the in-flight work (once) and enter the call into the process-wide
// revocation table, so the rank-2 Gate quarantines any late completion. The
// kill is appended to the journal, making the next fold's state KILLED — Tick
// is idempotent per call (a killed proc yields no further kill advice, and a
// fired cancel lever is dropped).
func (s *Supervisor) Tick(nowMS int64) (TickReport, error) {
	tab, err := s.Table(nowMS)
	if err != nil {
		return TickReport{}, err
	}
	var report TickReport
	killedNow := map[string]bool{}
	for _, p := range tab.Procs {
		for _, f := range p.Findings {
			act := TickAction{CallID: p.CallID, Reason: f.Reason, Advice: f.Advice}
			switch f.Advice {
			case toolproc.AdviceKill, toolproc.AdviceReap:
				// A proc can carry both kill (overdue) and reap (orphaned)
				// advice in one fold; the first act revokes, the rest report.
				if killedNow[p.CallID] {
					break
				}
				killedNow[p.CallID] = true
				Kill(p.CallID, f.Reason)
				s.mu.Lock()
				cancel := s.cancels[p.CallID]
				delete(s.cancels, p.CallID)
				pid, bound := s.pids[p.CallID]
				delete(s.pids, p.CallID)
				reaper := s.reaper
				s.events = append(s.events, toolproc.Event{
					Kind: toolproc.EvKill, CallID: p.CallID, AtMS: nowMS, Reason: f.Reason})
				s.mu.Unlock()
				if cancel != nil {
					cancel()
					act.Cancelled = true
				}
				// The OS lever runs outside the lock: reapers shell out (taskkill)
				// and must not stall other observers.
				if bound && reaper != nil {
					act.Reaped, act.ReapDetail = reaper(pid)
				}
			case toolproc.AdviceProbe, toolproc.AdviceQuarantineResult, toolproc.AdviceObserve:
				// probe: the embedder's move; quarantine_result: the Gate's;
				// observe: nothing to do. All reported, none destructive here.
			}
			report.Actions = append(report.Actions, act)
		}
	}
	// Re-fold when we enforced, so the report's table reflects the kills this
	// tick applied rather than the pre-enforcement view.
	if len(report.Actions) > 0 {
		tab, err = s.Table(nowMS)
		if err != nil {
			return TickReport{}, err
		}
	}
	report.Table = tab
	return report, nil
}

// PruneTerminal drops journal events for procs that reached a terminal state
// (DONE/KILLED) at or before cutoffMS, bounding the journal on long-lived
// embedders. Running procs and the session_end markers are always kept.
func (s *Supervisor) PruneTerminal(nowMS, cutoffMS int64) error {
	tab, err := s.Table(nowMS)
	if err != nil {
		return err
	}
	drop := map[string]bool{}
	for _, p := range tab.Procs {
		if p.State != toolproc.StateRunning && p.EndMS > 0 && p.EndMS <= cutoffMS {
			drop[p.CallID] = true
		}
	}
	if len(drop) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.events[:0]
	for _, ev := range s.events {
		if ev.CallID != "" && drop[ev.CallID] {
			continue
		}
		kept = append(kept, ev)
	}
	s.events = kept
	for id := range drop {
		delete(s.spawned, id) // identity retired with its events
		delete(s.cancels, id)
		delete(s.pids, id)
	}
	return nil
}
