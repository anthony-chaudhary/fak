package toolproc

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The out-of-tree refusal codes this leaf's verdict tokens cite (above
// abi.ReasonCoreMax = 1023): the closed core vocabulary in internal/abi is
// human-owned and additive-only, so this leaf reserves a block in the registered
// range — the sanctioned RegisterReason extension path, not a core edit.
// egressfloor holds 1024; this leaf takes 1040–1043 and leaves the gap as
// headroom for the egress family. The CONSUMER registers the names (the
// `fak toolproc` shell today, the gateway supervisor when enforcement wires up):
// this package stays a pure, init-free fold so it needs no defconfig entry.
const (
	ReasonToolDeadlineExceeded abi.ReasonCode = 1040
	ReasonToolHeartbeatStalled abi.ReasonCode = 1041
	ReasonToolOrphaned         abi.ReasonCode = 1042
	ReasonToolResultAfterKill  abi.ReasonCode = 1043
)

// The stable names for the codes above — the closed verdict vocabulary a
// finding cites. A finding never carries free text in Reason.
const (
	ReasonToolDeadlineExceededName = "TOOL_DEADLINE_EXCEEDED"
	ReasonToolHeartbeatStalledName = "TOOL_HEARTBEAT_STALLED"
	ReasonToolOrphanedName         = "TOOL_ORPHANED"
	ReasonToolResultAfterKillName  = "TOOL_RESULT_AFTER_KILL"
)

// ReasonPair is one (code, name) row of this leaf's verdict vocabulary.
type ReasonPair struct {
	Code abi.ReasonCode
	Name string
}

// ReasonPairs lists the vocabulary for a consumer to abi.RegisterReason — kept
// as data so the registration call sites cannot drift from the constants.
func ReasonPairs() []ReasonPair {
	return []ReasonPair{
		{ReasonToolDeadlineExceeded, ReasonToolDeadlineExceededName},
		{ReasonToolHeartbeatStalled, ReasonToolHeartbeatStalledName},
		{ReasonToolOrphaned, ReasonToolOrphanedName},
		{ReasonToolResultAfterKill, ReasonToolResultAfterKillName},
	}
}

// EventKind is the CLOSED journal-event vocabulary. Parse fails closed: an
// unrecognized kind refuses the journal at the boundary, never coerces.
type EventKind string

const (
	EvSpawn      EventKind = "spawn"       // a tool call went long: it is now a process, not a request
	EvPulse      EventKind = "pulse"       // any liveness signal: heartbeat, output chunk, progress, poll
	EvExit       EventKind = "exit"        // the tool call completed (ok or error)
	EvKill       EventKind = "kill"        // the supervisor revoked it mid-flight, citing a closed reason
	EvSessionEnd EventKind = "session_end" // the owning session ended (orphan boundary)
)

// Event is one journal row. The journal is append-only JSONL: unknown FIELDS are
// tolerated (additive evolution), unknown enum TOKENS are refused (closed sets).
type Event struct {
	Kind    EventKind `json:"kind"`
	CallID  string    `json:"call_id,omitempty"` // ToolCall.TraceID of the adjudicated launch
	Session string    `json:"session,omitempty"` // owning session/lease id (spawn, session_end)
	Tool    string    `json:"tool,omitempty"`    // spawn only
	AtMS    int64     `json:"at_unix_ms"`

	// spawn only — the declared runtime envelope, granted at admission alongside
	// the capability verdict. Zero means "not declared" (Config defaults apply).
	DeadlineMS       int64 `json:"deadline_ms,omitempty"`
	HeartbeatEveryMS int64 `json:"heartbeat_every_ms,omitempty"`

	// pulse only (optional progress annotation).
	Done  float64 `json:"done,omitempty"`
	Total float64 `json:"total,omitempty"`

	// exit only: "ok" | "error" (closed).
	Status string `json:"status,omitempty"`

	// kill only: the closed reason token the revocation cites (e.g.
	// TOOL_DEADLINE_EXCEEDED, or an operator token). Required, non-empty.
	Reason string `json:"reason,omitempty"`

	// pulse only (optional): the TraceID of the tool call that carried this
	// signal (e.g. a BashOutput poll) — the launch↔poll correlation the
	// uncorrelated-events gap is about.
	Via string `json:"via,omitempty"`
}

// State is the CLOSED per-proc lifecycle state after the fold.
type State string

const (
	StateRunning State = "RUNNING"
	StateDone    State = "DONE"
	StateKilled  State = "KILLED"
)

// Liveness classifies a RUNNING proc's signal recency against its declared
// cadence. A proc that declared no cadence can never be STALLED — it is QUIET
// (unknown-liveness), which is honest, not green.
type Liveness string

const (
	LivenessLive    Liveness = "LIVE"
	LivenessQuiet   Liveness = "QUIET"
	LivenessStalled Liveness = "STALLED"
	LivenessNA      Liveness = "-" // terminal procs
)

// Advice is the CLOSED supervisor-action vocabulary a finding carries.
type Advice string

const (
	AdviceObserve          Advice = "observe"
	AdviceProbe            Advice = "probe"
	AdviceKill             Advice = "kill"
	AdviceReap             Advice = "reap"
	AdviceQuarantineResult Advice = "quarantine_result"
)

// Finding is one lifecycle verdict on one proc: a closed reason token plus the
// closed advice the supervisor should take. Detail is a deterministic human
// line (no timestamps beyond what the fold was given).
type Finding struct {
	Reason string `json:"reason"`
	Code   uint32 `json:"code"`
	Advice Advice `json:"advice"`
	Detail string `json:"detail"`
}

// Proc is one folded tool process — the row `fak toolproc ps` renders.
type Proc struct {
	CallID  string `json:"call_id"`
	Tool    string `json:"tool"`
	Session string `json:"session,omitempty"`
	State   State  `json:"state"`

	StartMS     int64 `json:"start_unix_ms"`
	EndMS       int64 `json:"end_unix_ms,omitempty"`
	LastPulseMS int64 `json:"last_pulse_unix_ms,omitempty"`
	RuntimeMS   int64 `json:"runtime_ms"`

	DeadlineMS       int64 `json:"deadline_ms,omitempty"` // effective (declared or config default)
	HeartbeatEveryMS int64 `json:"heartbeat_every_ms,omitempty"`

	Liveness   Liveness `json:"liveness"`
	OverdueMS  int64    `json:"overdue_ms,omitempty"`
	Orphaned   bool     `json:"orphaned,omitempty"`
	ExitStatus string   `json:"exit_status,omitempty"`
	KillReason string   `json:"kill_reason,omitempty"`
	LatePulses int      `json:"late_pulses,omitempty"` // benign: signals after a terminal state
	Pulses     int      `json:"pulses,omitempty"`

	Findings []Finding `json:"findings,omitempty"`
}

// Counts is the table's attention summary.
type Counts struct {
	Running  int `json:"running"`
	Done     int `json:"done"`
	Killed   int `json:"killed"`
	Overdue  int `json:"overdue"`
	Stalled  int `json:"stalled"`
	Orphaned int `json:"orphaned"`
}

// Config tunes the fold. Zero values are safe: no default deadline (unbounded
// unless the spawn declared one) and the stall multiplier below.
type Config struct {
	// DefaultDeadlineMS applies to a proc whose spawn declared no deadline.
	// 0 = no default (such a proc is unbounded — visible as QUIET, never OVERDUE).
	DefaultDeadlineMS int64 `json:"default_deadline_ms,omitempty"`
	// StallMultiplier: a proc with declared cadence h is STALLED when
	// now-lastSignal > StallMultiplier×h. 0 => DefaultStallMultiplier.
	StallMultiplier float64 `json:"stall_multiplier,omitempty"`
}

// DefaultStallMultiplier gives a heartbeat two grace beats before the third
// missed beat is called a stall.
const DefaultStallMultiplier = 3.0

// Table is the deterministic fold output: the kernel's process table for tool
// calls at one instant.
type Table struct {
	Schema    string `json:"schema"`
	NowUnixMS int64  `json:"now_unix_ms"`
	Config    Config `json:"config"`
	Procs     []Proc `json:"procs"`
	Counts    Counts `json:"counts"`
	// AttentionNeeded is true when any finding advises more than observe —
	// the one-bit gate a supervisor loop or CI can key on.
	AttentionNeeded bool `json:"attention_needed"`
}

// TableSchema stamps the fold output.
const TableSchema = "fak.toolproc-table.v1"

// ParseEvents reads a JSONL journal. Fail-closed: an unknown kind, a missing
// identity field, a bad enum token, or malformed JSON refuses the whole journal
// with the offending line number — fail at the boundary, do not guess.
func ParseEvents(r io.Reader) ([]Event, error) {
	var out []Event
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			return nil, fmt.Errorf("toolproc: line %d: %v", line, err)
		}
		if err := ValidateEvent(ev); err != nil {
			return nil, fmt.Errorf("toolproc: line %d: %v", line, err)
		}
		out = append(out, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("toolproc: %v", err)
	}
	return out, nil
}

// ValidateEvent enforces the closed vocabulary and per-kind required fields.
func ValidateEvent(ev Event) error {
	if ev.AtMS <= 0 {
		return fmt.Errorf("event %q: at_unix_ms must be positive", ev.Kind)
	}
	switch ev.Kind {
	case EvSpawn:
		if ev.CallID == "" {
			return fmt.Errorf("spawn: call_id required")
		}
		if ev.Tool == "" {
			return fmt.Errorf("spawn %s: tool required", ev.CallID)
		}
		if ev.DeadlineMS < 0 || ev.HeartbeatEveryMS < 0 {
			return fmt.Errorf("spawn %s: negative envelope", ev.CallID)
		}
	case EvPulse:
		if ev.CallID == "" {
			return fmt.Errorf("pulse: call_id required")
		}
	case EvExit:
		if ev.CallID == "" {
			return fmt.Errorf("exit: call_id required")
		}
		if ev.Status != "ok" && ev.Status != "error" {
			return fmt.Errorf("exit %s: status must be ok|error, got %q", ev.CallID, ev.Status)
		}
	case EvKill:
		if ev.CallID == "" {
			return fmt.Errorf("kill: call_id required")
		}
		if strings.TrimSpace(ev.Reason) == "" {
			return fmt.Errorf("kill %s: reason token required", ev.CallID)
		}
	case EvSessionEnd:
		if ev.Session == "" {
			return fmt.Errorf("session_end: session required")
		}
	default:
		return fmt.Errorf("unknown event kind %q", ev.Kind)
	}
	return nil
}

// proc is the mutable fold accumulator for one call.
type proc struct {
	Proc
	killedAtMS     int64
	exitAfterKill  bool
	sessionEndedMS int64
}

// Fold replays a validated journal into the process table at nowMS. It is a
// pure function: same events + same now + same config => byte-identical table.
//
// Tolerated benign races (documented, counted, never guessed):
//   - a pulse after a terminal state increments LatePulses;
//   - an exit after a kill keeps state KILLED and flags TOOL_RESULT_AFTER_KILL
//     (the completion's payload must not be admitted as if the call were live);
//   - a kill after an exit is a no-op (the killer lost the race, nothing to do).
//
// Refused impossible transitions (fail closed):
//   - a duplicate spawn for a live call_id (identity collision);
//   - a non-spawn event for a call_id never spawned;
//   - a spawn owned by a session that already ended (the orphan-leak class:
//     children forked from a dead parent).
func Fold(events []Event, nowMS int64, cfg Config) (Table, error) {
	if nowMS <= 0 {
		return Table{}, fmt.Errorf("toolproc: now_unix_ms must be positive")
	}
	if cfg.StallMultiplier == 0 {
		cfg.StallMultiplier = DefaultStallMultiplier
	}
	if cfg.StallMultiplier < 1 {
		return Table{}, fmt.Errorf("toolproc: stall multiplier must be >= 1")
	}
	if cfg.DefaultDeadlineMS < 0 {
		return Table{}, fmt.Errorf("toolproc: negative default deadline")
	}

	procs := map[string]*proc{}
	sessionEnd := map[string]int64{}
	var order []string

	for _, ev := range events {
		if err := ValidateEvent(ev); err != nil {
			return Table{}, fmt.Errorf("toolproc: %v", err)
		}
		switch ev.Kind {
		case EvSpawn:
			if _, dup := procs[ev.CallID]; dup {
				return Table{}, fmt.Errorf("toolproc: duplicate spawn for call %s", ev.CallID)
			}
			if endMS, dead := sessionEnd[ev.Session]; dead && ev.Session != "" {
				return Table{}, fmt.Errorf("toolproc: spawn %s from session %s which ended at %d", ev.CallID, ev.Session, endMS)
			}
			deadline := ev.DeadlineMS
			if deadline == 0 {
				deadline = cfg.DefaultDeadlineMS
			}
			procs[ev.CallID] = &proc{Proc: Proc{
				CallID: ev.CallID, Tool: ev.Tool, Session: ev.Session,
				State: StateRunning, StartMS: ev.AtMS,
				DeadlineMS: deadline, HeartbeatEveryMS: ev.HeartbeatEveryMS,
			}}
			order = append(order, ev.CallID)
		case EvPulse:
			p, ok := procs[ev.CallID]
			if !ok {
				return Table{}, fmt.Errorf("toolproc: pulse for unknown call %s", ev.CallID)
			}
			if p.State != StateRunning {
				p.LatePulses++
				continue
			}
			p.Pulses++
			if ev.AtMS > p.LastPulseMS {
				p.LastPulseMS = ev.AtMS
			}
		case EvExit:
			p, ok := procs[ev.CallID]
			if !ok {
				return Table{}, fmt.Errorf("toolproc: exit for unknown call %s", ev.CallID)
			}
			switch p.State {
			case StateRunning:
				p.State = StateDone
				p.EndMS = ev.AtMS
				p.ExitStatus = ev.Status
			case StateKilled:
				p.exitAfterKill = true
				p.ExitStatus = ev.Status
			case StateDone:
				return Table{}, fmt.Errorf("toolproc: double exit for call %s", ev.CallID)
			}
		case EvKill:
			p, ok := procs[ev.CallID]
			if !ok {
				return Table{}, fmt.Errorf("toolproc: kill for unknown call %s", ev.CallID)
			}
			if p.State != StateRunning {
				continue // lost the race with completion or a prior kill; nothing to revoke
			}
			p.State = StateKilled
			p.EndMS = ev.AtMS
			p.killedAtMS = ev.AtMS
			p.KillReason = ev.Reason
		case EvSessionEnd:
			if _, dup := sessionEnd[ev.Session]; !dup {
				sessionEnd[ev.Session] = ev.AtMS
			}
		}
	}

	t := Table{Schema: TableSchema, NowUnixMS: nowMS, Config: cfg}
	for _, id := range order {
		p := procs[id]
		if endMS, dead := sessionEnd[p.Session]; dead && p.Session != "" {
			p.sessionEndedMS = endMS
		}
		finalizeProc(p, nowMS, cfg)
		t.Procs = append(t.Procs, p.Proc)
		switch p.State {
		case StateRunning:
			t.Counts.Running++
		case StateDone:
			t.Counts.Done++
		case StateKilled:
			t.Counts.Killed++
		}
		if p.OverdueMS > 0 {
			t.Counts.Overdue++
		}
		if p.Liveness == LivenessStalled {
			t.Counts.Stalled++
		}
		if p.Orphaned {
			t.Counts.Orphaned++
		}
		for _, f := range p.Findings {
			if f.Advice != AdviceObserve {
				t.AttentionNeeded = true
			}
		}
	}
	sort.SliceStable(t.Procs, func(i, j int) bool {
		if t.Procs[i].StartMS != t.Procs[j].StartMS {
			return t.Procs[i].StartMS < t.Procs[j].StartMS
		}
		return t.Procs[i].CallID < t.Procs[j].CallID
	})
	return t, nil
}

// finalizeProc computes runtime, liveness, deadline, and orphan classes plus
// the findings, in a fixed order (deadline, stall, orphan, result-after-kill)
// so the output is deterministic.
func finalizeProc(p *proc, nowMS int64, cfg Config) {
	end := p.EndMS
	if p.State == StateRunning {
		end = nowMS
	}
	p.RuntimeMS = end - p.StartMS
	if p.RuntimeMS < 0 {
		p.RuntimeMS = 0
	}

	p.Liveness = LivenessNA
	if p.State == StateRunning {
		p.Liveness = LivenessQuiet
		if p.HeartbeatEveryMS > 0 {
			last := p.StartMS
			if p.LastPulseMS > last {
				last = p.LastPulseMS
			}
			silence := nowMS - last
			if float64(silence) > cfg.StallMultiplier*float64(p.HeartbeatEveryMS) {
				p.Liveness = LivenessStalled
			} else {
				p.Liveness = LivenessLive
			}
		}
	}

	if p.State == StateRunning && p.DeadlineMS > 0 && p.RuntimeMS > p.DeadlineMS {
		p.OverdueMS = p.RuntimeMS - p.DeadlineMS
		p.Findings = append(p.Findings, Finding{
			Reason: ReasonToolDeadlineExceededName,
			Code:   uint32(ReasonToolDeadlineExceeded),
			Advice: AdviceKill,
			Detail: fmt.Sprintf("running %dms past its %dms deadline", p.OverdueMS, p.DeadlineMS),
		})
	}
	if p.Liveness == LivenessStalled {
		p.Findings = append(p.Findings, Finding{
			Reason: ReasonToolHeartbeatStalledName,
			Code:   uint32(ReasonToolHeartbeatStalled),
			Advice: AdviceProbe,
			Detail: fmt.Sprintf("no signal within %.0fx its %dms cadence", cfg.StallMultiplier, p.HeartbeatEveryMS),
		})
	}
	if p.State == StateRunning && p.sessionEndedMS > 0 {
		p.Orphaned = true
		p.Findings = append(p.Findings, Finding{
			Reason: ReasonToolOrphanedName,
			Code:   uint32(ReasonToolOrphaned),
			Advice: AdviceReap,
			Detail: fmt.Sprintf("owning session %s ended at %d; call still running", p.Session, p.sessionEndedMS),
		})
	}
	if p.exitAfterKill {
		p.Findings = append(p.Findings, Finding{
			Reason: ReasonToolResultAfterKillName,
			Code:   uint32(ReasonToolResultAfterKill),
			Advice: AdviceQuarantineResult,
			Detail: "completion arrived after the kill; its payload must not be admitted as live",
		})
	}
}

// Sample returns a deterministic built-in journal + fold instant that exercises
// every verdict class — the no-key, no-model, no-GPU proof `fak toolproc sample`
// renders. Fixed epoch so the output is byte-stable.
func Sample() ([]Event, int64, Config) {
	const base int64 = 1_700_000_000_000
	now := base + 95_000
	cfg := Config{DefaultDeadlineMS: 0, StallMultiplier: DefaultStallMultiplier}
	events := []Event{
		// healthy, short, done — the contrast row.
		{Kind: EvSpawn, CallID: "t-done", Session: "s1", Tool: "search_repo", AtMS: base},
		{Kind: EvExit, CallID: "t-done", AtMS: base + 2_000, Status: "ok"},
		// healthy long-runner with a live heartbeat.
		{Kind: EvSpawn, CallID: "t-live", Session: "s1", Tool: "train_probe", AtMS: base + 1_000, HeartbeatEveryMS: 10_000},
		{Kind: EvPulse, CallID: "t-live", AtMS: base + 9_000},
		{Kind: EvPulse, CallID: "t-live", AtMS: base + 90_000, Via: "poll-1"},
		// overdue: declared a 30s deadline, still running at +95s.
		{Kind: EvSpawn, CallID: "t-overdue", Session: "s1", Tool: "slow_fetch", AtMS: base, DeadlineMS: 30_000},
		// stalled: 5s cadence, one pulse, silent since +4s.
		{Kind: EvSpawn, CallID: "t-stalled", Session: "s1", Tool: "bg_tail", AtMS: base + 2_000, HeartbeatEveryMS: 5_000},
		{Kind: EvPulse, CallID: "t-stalled", AtMS: base + 4_000},
		// orphaned: its session ended under it.
		{Kind: EvSpawn, CallID: "t-orphan", Session: "s2", Tool: "watch_dir", AtMS: base + 3_000},
		{Kind: EvSessionEnd, Session: "s2", AtMS: base + 20_000},
		// killed at deadline, completion landed late => quarantine the payload.
		{Kind: EvSpawn, CallID: "t-late", Session: "s1", Tool: "big_dump", AtMS: base, DeadlineMS: 10_000},
		{Kind: EvKill, CallID: "t-late", AtMS: base + 12_000, Reason: ReasonToolDeadlineExceededName},
		{Kind: EvExit, CallID: "t-late", AtMS: base + 13_000, Status: "ok"},
	}
	return events, now, cfg
}
