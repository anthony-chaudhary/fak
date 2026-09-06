package toolprocgate

import (
	"fmt"
	"sync"
	"time"

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
	mu           sync.Mutex
	cfg          toolproc.Config
	events       []toolproc.Event
	cancels      map[string]func() // callID -> cancel lever, cleared once fired or terminal
	spawned      map[string]bool   // callIDs ever spawned (journal identity guard)
	pids         map[string]int    // callID -> bound OS process-tree root (seam 6), cleared with cancels
	reaper       OSReaper
	terminalSink func(toolproc.Proc)
	terminalSent map[string]bool // OS lever for bound pids; nil = advice-only (no teeth)
	// recentFaults is a bounded ring of the last console faults ExitConsoleFault
	// recorded, kept so AdmitSpawn can fold them into a blast-radius containment
	// verdict — the memory that lets a crash contain the NEXT spawn instead of
	// only the dead call. Bounded by recentFaultRingCap so a fault storm cannot
	// grow it without limit.
	recentFaults []ConsoleFaultEvent

	// SettlementGrace is the optional bounded settlement grace period allowing
	// protocol-aware tool processes to answer a cancel request and exit cleanly
	// before forceful OS process tree reap.
	SettlementGrace time.Duration
	processGrace    map[string]time.Duration
	settlements     map[string]*settlementRecord
}

// ToolProcSupervisor is an alias for Supervisor for embedders and callers.
type ToolProcSupervisor = Supervisor

// ProcessMonitor is an alias for Supervisor for embedders and callers.
type ProcessMonitor = Supervisor

// TerminalClassification represents the distinct terminal outcome of a cancelled tool call.
type TerminalClassification string

const (
	// TerminalPreDispatchAbort indicates the tool call was cancelled or aborted before OS process dispatch.
	TerminalPreDispatchAbort TerminalClassification = "pre-dispatch-abort"
	// TerminalDeliveredAndSettled indicates the tool process received cancellation and settled cleanly within grace.
	TerminalDeliveredAndSettled TerminalClassification = "delivered-and-settled"
	// TerminalDeliveredButUnknown indicates grace expired without clean settlement and the process was forcefully reaped.
	TerminalDeliveredButUnknown TerminalClassification = "delivered-but-unknown"
)

type settlementState string

const (
	settlementPending settlementState = "settlement_pending"
	settlementSettled settlementState = "settled"
	settlementReaped  settlementState = "reaped"
)

type settlementRecord struct {
	CallID                 string
	Reason                 string
	Advice                 toolproc.Advice
	CancelRequestedAt      time.Time
	RequestedAtMS          int64
	DeadlineMS             int64
	SettledAt              time.Time
	SettledAtMS            int64
	ReapedAt               time.Time
	ReapedAtMS             int64
	State                  settlementState
	Cancelled              bool
	Reaped                 bool
	ReapDetail             string
	Settled                bool
	ReportedSettled        bool
	ReportedReaped         bool
	ReportedPending        bool
	WasBound               bool
	TerminalClassification TerminalClassification
}

// SettlementInfo contains the settlement status, timestamps, and terminal classification for a call.
type SettlementInfo struct {
	CallID                 string                 `json:"call_id"`
	Reason                 string                 `json:"reason,omitempty"`
	State                  string                 `json:"state,omitempty"`
	Cancelled              bool                   `json:"cancelled,omitempty"`
	Settled                bool                   `json:"settled,omitempty"`
	Reaped                 bool                   `json:"reaped,omitempty"`
	Pending                bool                   `json:"pending,omitempty"`
	CancelRequestedAt      time.Time              `json:"cancel_requested_at,omitempty"`
	RequestedAtMS          int64                  `json:"requested_at_ms,omitempty"`
	DeadlineMS             int64                  `json:"deadline_ms,omitempty"`
	SettledAt              time.Time              `json:"settled_at,omitempty"`
	SettledAtMS            int64                  `json:"settled_at_ms,omitempty"`
	ReapedAt               time.Time              `json:"reaped_at,omitempty"`
	ReapedAtMS             int64                  `json:"reaped_at_ms,omitempty"`
	TerminalClassification TerminalClassification `json:"terminal_classification,omitempty"`
}

// recentFaultRingCap bounds the in-memory fault history AdmitSpawn folds. The
// containment policy only ever looks inside a minutes-wide window, so a few
// hundred rows is far more than any live decision reads; the cap is the
// storm-proofing backstop, not the working set.
const recentFaultRingCap = 256

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
	// Settled reports the process cleanly terminated/settled within the
	// settlement grace window without requiring a forceful OS process tree reap.
	Settled bool `json:"settled,omitempty"`
	// SettlementPending reports that cancellation has been requested and the process
	// is currently within its bounded settlement grace window awaiting clean exit before reap.
	SettlementPending bool `json:"settlement_pending,omitempty"`
	// TerminalClassification reports the final fate classification (pre-dispatch-abort,
	// delivered-and-settled, delivered-but-unknown) when a terminal outcome is reached.
	TerminalClassification TerminalClassification `json:"terminal_classification,omitempty"`
}

// TickReport is what one Tick did: the folded table plus the acts.
type TickReport struct {
	Table   toolproc.Table `json:"table"`
	Actions []TickAction   `json:"actions,omitempty"`
}

// SetTerminalSink registers the loop-wake seam. Tick invokes sink once for
// each newly observed terminal transition, after folding the authoritative table.
func (s *Supervisor) SetTerminalSink(sink func(toolproc.Proc)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.terminalSink = sink
}

// NewSupervisor builds a Supervisor with the given fold config (zero value =
// toolproc defaults: no default deadline, stall multiplier 3).
func NewSupervisor(cfg toolproc.Config) *Supervisor {
	return &Supervisor{
		cfg:          cfg,
		cancels:      map[string]func(){},
		spawned:      map[string]bool{},
		pids:         map[string]int{},
		terminalSent: map[string]bool{},
		processGrace: map[string]time.Duration{},
		settlements:  map[string]*settlementRecord{},
	}
}

// SetSettlementGrace sets the default bounded settlement grace period for all processes.
func (s *Supervisor) SetSettlementGrace(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.SettlementGrace = d
}

// SetProcessSettlementGrace sets an optional per-process settlement grace override.
func (s *Supervisor) SetProcessSettlementGrace(callID string, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.processGrace == nil {
		s.processGrace = make(map[string]time.Duration)
	}
	s.processGrace[callID] = d
}

// EffectiveGrace returns the effective settlement grace period for callID.
func (s *Supervisor) EffectiveGrace(callID string) time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.effectiveGraceLocked(callID)
}

func (s *Supervisor) effectiveGraceLocked(callID string) time.Duration {
	if s.processGrace != nil {
		if d, ok := s.processGrace[callID]; ok {
			return d
		}
	}
	return s.SettlementGrace
}

// Cancel requests cancellation for a running call. If SettlementGrace > 0
// (or a per-process grace override is configured), it triggers the process's
// cancel lever and begins the bounded settlement grace window. If no grace
// is configured (SettlementGrace <= 0), it immediately revokes and reaps.
func (s *Supervisor) Cancel(callID string, nowMS int64, reason string) error {
	if reason == "" {
		reason = "CANCEL_REQUESTED"
	}
	s.mu.Lock()
	if !s.spawned[callID] {
		s.mu.Unlock()
		return fmt.Errorf("toolprocgate: cancel for unknown call %s", callID)
	}
	for _, e := range s.events {
		if e.CallID == callID && (e.Kind == toolproc.EvExit || e.Kind == toolproc.EvKill) {
			s.mu.Unlock()
			return nil
		}
	}
	if rec := s.settlements[callID]; rec != nil && rec.State != settlementPending {
		s.mu.Unlock()
		return nil
	}
	grace := s.effectiveGraceLocked(callID)
	cancel := s.cancels[callID]
	delete(s.cancels, callID)
	_, bound := s.pids[callID]

	if grace > 0 {
		rec := s.settlements[callID]
		firstRequest := false
		if rec == nil {
			firstRequest = true
			rec = &settlementRecord{
				CallID:            callID,
				Reason:            reason,
				Advice:            toolproc.AdviceKill,
				CancelRequestedAt: time.UnixMilli(nowMS),
				RequestedAtMS:     nowMS,
				DeadlineMS:        nowMS + graceToMS(grace),
				State:             settlementPending,
				WasBound:          bound,
			}
			if s.settlements == nil {
				s.settlements = make(map[string]*settlementRecord)
			}
			s.settlements[callID] = rec
		} else if bound {
			rec.WasBound = true
		}
		s.mu.Unlock()
		if firstRequest && cancel != nil {
			cancel()
			s.mu.Lock()
			rec.Cancelled = true
			s.mu.Unlock()
		}
		return nil
	}

	pid, bound := s.pids[callID]
	delete(s.pids, callID)
	reaper := s.reaper
	s.events = append(s.events, toolproc.Event{
		Kind: toolproc.EvKill, CallID: callID, AtMS: nowMS, Reason: reason})
	term := TerminalPreDispatchAbort
	if bound {
		term = TerminalDeliveredButUnknown
	}
	rec := &settlementRecord{
		CallID:                 callID,
		Reason:                 reason,
		Advice:                 toolproc.AdviceKill,
		CancelRequestedAt:      time.UnixMilli(nowMS),
		RequestedAtMS:          nowMS,
		DeadlineMS:             nowMS,
		ReapedAt:               time.UnixMilli(nowMS),
		ReapedAtMS:             nowMS,
		State:                  settlementReaped,
		Cancelled:              cancel != nil,
		ReportedReaped:         true,
		WasBound:               bound,
		TerminalClassification: term,
	}
	if s.settlements == nil {
		s.settlements = make(map[string]*settlementRecord)
	}
	s.settlements[callID] = rec
	s.mu.Unlock()

	Kill(callID, reason)
	if cancel != nil {
		cancel()
	}
	if bound && reaper != nil {
		ok, detail := reaper(pid)
		s.mu.Lock()
		rec.Reaped = ok
		rec.ReapDetail = detail
		s.mu.Unlock()
	}
	return nil
}

// Settle marks a call ID as settled cleanly before forceful reap.
func (s *Supervisor) Settle(callID string, nowMS int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.spawned[callID] {
		return fmt.Errorf("toolprocgate: settle for unknown call %s", callID)
	}
	_, bound := s.pids[callID]
	delete(s.cancels, callID)
	delete(s.pids, callID)
	rec, ok := s.settlements[callID]
	if ok {
		if rec.State != settlementReaped {
			rec.State = settlementSettled
			rec.Settled = true
			rec.SettledAt = time.UnixMilli(nowMS)
			rec.SettledAtMS = nowMS
			if rec.WasBound || bound {
				rec.TerminalClassification = TerminalDeliveredAndSettled
			} else {
				rec.TerminalClassification = TerminalPreDispatchAbort
			}
		}
	} else {
		if s.settlements == nil {
			s.settlements = make(map[string]*settlementRecord)
		}
		term := TerminalPreDispatchAbort
		if bound {
			term = TerminalDeliveredAndSettled
		}
		s.settlements[callID] = &settlementRecord{
			CallID:                 callID,
			State:                  settlementSettled,
			Settled:                true,
			SettledAt:              time.UnixMilli(nowMS),
			SettledAtMS:            nowMS,
			WasBound:               bound,
			TerminalClassification: term,
		}
	}
	hasExit := false
	for _, ev := range s.events {
		if ev.CallID == callID && (ev.Kind == toolproc.EvExit || ev.Kind == toolproc.EvKill) {
			hasExit = true
			break
		}
	}
	if !hasExit {
		atMS := nowMS
		if atMS <= 0 {
			atMS = 1
		}
		s.events = append(s.events, toolproc.Event{
			Kind:   toolproc.EvExit,
			CallID: callID,
			AtMS:   atMS,
			Status: "ok",
		})
	}
	return nil
}

// CancelRequestedAt returns the time cancellation was requested for callID,
// or a zero time if no cancellation was requested.
func (s *Supervisor) CancelRequestedAt(callID string) time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.CancelRequestedAt
	}
	return time.Time{}
}

// RequestedAtMS returns the timestamp in ms when cancellation was requested for callID,
// or 0 if no cancellation was requested.
func (s *Supervisor) RequestedAtMS(callID string) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.RequestedAtMS
	}
	return 0
}

// SettledAt returns the time callID settled cleanly, or a zero time if not settled.
func (s *Supervisor) SettledAt(callID string) time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.SettledAt
	}
	return time.Time{}
}

// SettledAtMS returns the timestamp in ms when callID settled cleanly, or 0 if not settled.
func (s *Supervisor) SettledAtMS(callID string) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.SettledAtMS
	}
	return 0
}

// ReapedAt returns the time callID was forcefully reaped, or a zero time if not reaped.
func (s *Supervisor) ReapedAt(callID string) time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.ReapedAt
	}
	return time.Time{}
}

// ReapedAtMS returns the timestamp in ms when callID was forcefully reaped, or 0 if not reaped.
func (s *Supervisor) ReapedAtMS(callID string) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.ReapedAtMS
	}
	return 0
}

// SettlementPending reports whether callID is currently within its settlement grace window.
func (s *Supervisor) SettlementPending(callID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.State == settlementPending
	}
	return false
}

// IsSettled reports whether callID settled cleanly within its grace window.
func (s *Supervisor) IsSettled(callID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.Settled
	}
	return false
}

// SettlementStatus returns (settled, reaped, pending) for callID.
func (s *Supervisor) SettlementStatus(callID string) (settled, reaped, pending bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.settlements[callID]
	if !ok {
		return false, false, false
	}
	return rec.Settled, rec.Reaped, rec.State == settlementPending
}

// TerminalClassification returns the terminal fate classification for callID,
// or empty string if callID is not in a terminal state.
func (s *Supervisor) TerminalClassification(callID string) TerminalClassification {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rec, ok := s.settlements[callID]; ok {
		return rec.TerminalClassification
	}
	return ""
}

// SettlementInfo returns a snapshot of the settlement record for callID, if any.
func (s *Supervisor) SettlementInfo(callID string) (SettlementInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.settlements[callID]
	if !ok {
		return SettlementInfo{}, false
	}
	return SettlementInfo{
		CallID:                 rec.CallID,
		Reason:                 rec.Reason,
		State:                  string(rec.State),
		Cancelled:              rec.Cancelled,
		Settled:                rec.Settled,
		Reaped:                 rec.Reaped,
		Pending:                rec.State == settlementPending,
		CancelRequestedAt:      rec.CancelRequestedAt,
		RequestedAtMS:          rec.RequestedAtMS,
		DeadlineMS:             rec.DeadlineMS,
		SettledAt:              rec.SettledAt,
		SettledAtMS:            rec.SettledAtMS,
		ReapedAt:               rec.ReapedAt,
		ReapedAtMS:             rec.ReapedAtMS,
		TerminalClassification: rec.TerminalClassification,
	}, true
}

func graceToMS(d time.Duration) int64 {
	ms := d.Milliseconds()
	if ms <= 0 && d > 0 {
		return 1
	}
	return ms
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

// ArmMonitor arms an event-stream monitor as a liveness process. It enforces
// the observed-coverage doctrine at the seam: toolproc.ArmMonitor REFUSES a
// filter that covers no failure-signature class (MONITOR_NO_FAILURE_COVERAGE)
// BEFORE anything is journaled, so a progress-only monitor never enters the
// table. An armed monitor carries its declared heartbeat cadence, so a stream
// that goes quiet folds to TOOL_HEARTBEAT_STALLED with KILL advice and the next
// Tick revokes it (unlike a generic long-runner's stall, which is only probed).
// cancel is the lever Tick pulls on that kill; nil means observable-but-not-
// cancellable (the revocation still quarantines a late completion).
func (s *Supervisor) ArmMonitor(spec toolproc.MonitorSpec, cancel func()) error {
	ev, err := toolproc.ArmMonitor(spec)
	if err != nil {
		return err
	}
	if err := toolproc.ValidateEvent(ev); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.spawned[ev.CallID] {
		return fmt.Errorf("toolprocgate: duplicate spawn for call %s", ev.CallID)
	}
	s.spawned[ev.CallID] = true
	s.events = append(s.events, ev)
	if cancel != nil {
		s.cancels[ev.CallID] = cancel
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
	ev := toolproc.Event{Kind: toolproc.EvExit, CallID: callID, AtMS: nowMS, Status: status}
	if err := toolproc.ValidateEvent(ev); err != nil {
		return err
	}
	s.mu.Lock()
	if !s.spawned[callID] {
		s.mu.Unlock()
		return fmt.Errorf("toolprocgate: exit for unknown call %s", callID)
	}
	hasExit := false
	for _, e := range s.events {
		if e.CallID == callID && (e.Kind == toolproc.EvExit || e.Kind == toolproc.EvKill) {
			hasExit = true
			break
		}
	}
	_, bound := s.pids[callID]
	delete(s.cancels, callID)
	delete(s.pids, callID)
	if rec, ok := s.settlements[callID]; ok && rec.State == settlementPending {
		rec.State = settlementSettled
		rec.Settled = true
		rec.SettledAt = time.UnixMilli(nowMS)
		rec.SettledAtMS = nowMS
		if rec.WasBound || bound {
			rec.TerminalClassification = TerminalDeliveredAndSettled
		} else {
			rec.TerminalClassification = TerminalPreDispatchAbort
		}
	}
	if hasExit {
		s.mu.Unlock()
		return nil
	}
	s.events = append(s.events, ev)
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
				s.mu.Lock()
				grace := s.effectiveGraceLocked(p.CallID)
				if grace <= 0 {
					cancel := s.cancels[p.CallID]
					delete(s.cancels, p.CallID)
					pid, bound := s.pids[p.CallID]
					delete(s.pids, p.CallID)
					reaper := s.reaper
					s.events = append(s.events, toolproc.Event{
						Kind: toolproc.EvKill, CallID: p.CallID, AtMS: nowMS, Reason: f.Reason})
					term := TerminalPreDispatchAbort
					if bound {
						term = TerminalDeliveredButUnknown
					}
					rec := &settlementRecord{
						CallID:                 p.CallID,
						Reason:                 f.Reason,
						Advice:                 f.Advice,
						CancelRequestedAt:      time.UnixMilli(nowMS),
						RequestedAtMS:          nowMS,
						DeadlineMS:             nowMS,
						ReapedAt:               time.UnixMilli(nowMS),
						ReapedAtMS:             nowMS,
						State:                  settlementReaped,
						Cancelled:              cancel != nil,
						ReportedReaped:         true,
						WasBound:               bound,
						TerminalClassification: term,
					}
					if s.settlements == nil {
						s.settlements = make(map[string]*settlementRecord)
					}
					s.settlements[p.CallID] = rec
					s.mu.Unlock()
					Kill(p.CallID, f.Reason)
					if cancel != nil {
						cancel()
						act.Cancelled = true
					}
					// The OS lever runs outside the lock: reapers shell out (taskkill)
					// and must not stall other observers.
					if bound && reaper != nil {
						act.Reaped, act.ReapDetail = reaper(pid)
						s.mu.Lock()
						rec.Reaped = act.Reaped
						rec.ReapDetail = act.ReapDetail
						s.mu.Unlock()
					}
					act.TerminalClassification = term
				} else {
					rec := s.settlements[p.CallID]
					firstRequest := false
					_, bound := s.pids[p.CallID]
					if rec == nil {
						firstRequest = true
						rec = &settlementRecord{
							CallID:            p.CallID,
							Reason:            f.Reason,
							Advice:            f.Advice,
							CancelRequestedAt: time.UnixMilli(nowMS),
							RequestedAtMS:     nowMS,
							DeadlineMS:        nowMS + graceToMS(grace),
							State:             settlementPending,
							WasBound:          bound,
						}
						if s.settlements == nil {
							s.settlements = make(map[string]*settlementRecord)
						}
						s.settlements[p.CallID] = rec
					} else if bound {
						rec.WasBound = true
					}
					cancel := s.cancels[p.CallID]
					delete(s.cancels, p.CallID)
					s.mu.Unlock()

					if firstRequest && cancel != nil {
						cancel()
						act.Cancelled = true
						s.mu.Lock()
						rec.Cancelled = true
						s.mu.Unlock()
					}

					s.mu.Lock()
					if rec.Settled || rec.State == settlementSettled {
						act.Settled = true
						act.Reaped = false
						act.TerminalClassification = rec.TerminalClassification
						rec.ReportedSettled = true
						s.mu.Unlock()
						break
					}

					if nowMS < rec.DeadlineMS {
						act.SettlementPending = true
						act.Settled = false
						act.Reaped = false
						rec.ReportedPending = true
						s.mu.Unlock()
						break
					}

					// Grace has expired without clean exit/settlement!
					// Execute forceful OS tree reap and record revocation.
					rec.State = settlementReaped
					rec.ReportedReaped = true
					rec.ReapedAt = time.UnixMilli(nowMS)
					rec.ReapedAtMS = nowMS
					term := TerminalDeliveredButUnknown
					if !rec.WasBound {
						term = TerminalPreDispatchAbort
					}
					rec.TerminalClassification = term
					act.TerminalClassification = term
					pid, bound := s.pids[p.CallID]
					delete(s.pids, p.CallID)
					reaper := s.reaper
					s.events = append(s.events, toolproc.Event{
						Kind: toolproc.EvKill, CallID: p.CallID, AtMS: nowMS, Reason: f.Reason})
					s.mu.Unlock()

					Kill(p.CallID, f.Reason)
					if bound && reaper != nil {
						act.Reaped, act.ReapDetail = reaper(pid)
						s.mu.Lock()
						rec.Reaped = act.Reaped
						rec.ReapDetail = act.ReapDetail
						s.mu.Unlock()
					}
					act.Settled = false
				}
			case toolproc.AdviceProbe, toolproc.AdviceQuarantineResult, toolproc.AdviceObserve:
				// probe: the embedder's move; quarantine_result: the Gate's;
				// observe: nothing to do. All reported, none destructive here.
			}
			report.Actions = append(report.Actions, act)
		}
	}

	// Post-check for any settlement records not processed above
	// (e.g. clean exit reported via Exit, or cancel requested explicitly via Cancel)
	type pendingReap struct {
		callID string
		rec    *settlementRecord
		pid    int
		bound  bool
	}
	var reapsToExecute []pendingReap

	s.mu.Lock()
	for callID, rec := range s.settlements {
		if killedNow[callID] {
			continue
		}
		if (rec.Settled || rec.State == settlementSettled) && !rec.ReportedSettled {
			rec.ReportedSettled = true
			report.Actions = append(report.Actions, TickAction{
				CallID:                 rec.CallID,
				Reason:                 rec.Reason,
				Advice:                 rec.Advice,
				Cancelled:              false,
				Settled:                true,
				Reaped:                 false,
				TerminalClassification: rec.TerminalClassification,
			})
			killedNow[callID] = true
		} else if rec.State == settlementPending && nowMS >= rec.DeadlineMS {
			rec.State = settlementReaped
			rec.ReportedReaped = true
			rec.ReapedAt = time.UnixMilli(nowMS)
			rec.ReapedAtMS = nowMS
			term := TerminalDeliveredButUnknown
			if !rec.WasBound {
				term = TerminalPreDispatchAbort
			}
			rec.TerminalClassification = term
			pid, bound := s.pids[callID]
			delete(s.pids, callID)
			s.events = append(s.events, toolproc.Event{
				Kind: toolproc.EvKill, CallID: callID, AtMS: nowMS, Reason: rec.Reason})
			reapsToExecute = append(reapsToExecute, pendingReap{
				callID: callID,
				rec:    rec,
				pid:    pid,
				bound:  bound,
			})
			killedNow[callID] = true
		} else if rec.State == settlementPending && nowMS < rec.DeadlineMS && !rec.ReportedPending {
			rec.ReportedPending = true
			report.Actions = append(report.Actions, TickAction{
				CallID:            rec.CallID,
				Reason:            rec.Reason,
				Advice:            rec.Advice,
				Cancelled:         rec.Cancelled,
				SettlementPending: true,
			})
			killedNow[callID] = true
		}
	}
	reaper := s.reaper
	s.mu.Unlock()

	for _, pr := range reapsToExecute {
		Kill(pr.callID, pr.rec.Reason)
		var reaped bool
		var reapDetail string
		if pr.bound && reaper != nil {
			reaped, reapDetail = reaper(pr.pid)
		}
		s.mu.Lock()
		pr.rec.Reaped = reaped
		pr.rec.ReapDetail = reapDetail
		s.mu.Unlock()
		report.Actions = append(report.Actions, TickAction{
			CallID:                 pr.callID,
			Reason:                 pr.rec.Reason,
			Advice:                 pr.rec.Advice,
			Cancelled:              false,
			Reaped:                 reaped,
			ReapDetail:             reapDetail,
			Settled:                false,
			TerminalClassification: pr.rec.TerminalClassification,
		})
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
	for _, p := range tab.Procs {
		if p.State != toolproc.StateDone && p.State != toolproc.StateKilled {
			continue
		}
		s.mu.Lock()
		sink, sent := s.terminalSink, s.terminalSent[p.CallID]
		if sink != nil && !sent {
			s.terminalSent[p.CallID] = true
		}
		s.mu.Unlock()
		if sink != nil && !sent {
			sink(p)
		}
	}
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
		delete(s.settlements, id)
		delete(s.processGrace, id)
	}
	return nil
}
