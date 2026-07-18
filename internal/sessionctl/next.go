package sessionctl

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// MoveKind is the closed vocabulary emitted by an autonomous decision point.
type MoveKind string

const (
	MoveContinue MoveKind = "continue"
	MoveRedirect MoveKind = "redirect"
	MoveAnnotate MoveKind = "annotate"
	MoveReanchor MoveKind = "re-anchor"
	MoveHalt     MoveKind = "halt"
)

// RenderKind is the closed boundary action used to apply a move.
type RenderKind string

const (
	RenderUserSplice      RenderKind = "user-splice"
	RenderSystemDirective RenderKind = "system-directive"
	RenderReopen          RenderKind = "reopen"
	RenderStop            RenderKind = "stop"
)

// SessionClass identifies the one rendering authority a trace belongs to.
type SessionClass string

const (
	SessionInteractive SessionClass = "interactive"
	SessionAutonomous  SessionClass = "autonomous"
)

// Move is a decision already made by a gate. Enqueue validates and records it;
// it never reruns Gate.
type Move struct {
	Kind    MoveKind     `json:"kind"`
	Render  RenderKind   `json:"render"`
	Session SessionClass `json:"session_class"`
	Gate    string       `json:"gate"`
	Source  string       `json:"source,omitempty"`
	Payload string       `json:"payload,omitempty"`
	Reason  string       `json:"reason,omitempty"`
	Shadow  bool         `json:"shadow,omitempty"`
}

// NextRecord is the independently readable witness of one attempted move.
type NextRecord struct {
	Sequence uint64    `json:"sequence"`
	At       time.Time `json:"at"`
	Move     Move      `json:"move"`
	Applied  bool      `json:"applied"`
	Refusal  string    `json:"refusal,omitempty"`
}

// ApplyResult distinguishes an applied move from a witnessed no-op/refusal.
type ApplyResult struct {
	Applied bool
	Refusal string
}

// MoveApplier is the sole actuation boundary used by Drain.
type MoveApplier func(Move) (ApplyResult, error)

type queuedMove struct {
	sequence uint64
	move     Move
}

// NextQueue preserves source-site enqueue order across render classes.
type NextQueue struct {
	mu      sync.Mutex
	next    uint64
	class   SessionClass
	moves   []queuedMove
	drained bool
}

// NewNextQueue binds a trace to exactly one session/rendering class.
func NewNextQueue(class SessionClass) (*NextQueue, error) {
	if !class.Valid() {
		return nil, fmt.Errorf("sessionctl next: invalid session class %q", class)
	}
	return &NextQueue{class: class}, nil
}

func (k MoveKind) Valid() bool {
	switch k {
	case MoveContinue, MoveRedirect, MoveAnnotate, MoveReanchor, MoveHalt:
		return true
	default:
		return false
	}
}

func (r RenderKind) Valid() bool {
	switch r {
	case RenderUserSplice, RenderSystemDirective, RenderReopen, RenderStop:
		return true
	default:
		return false
	}
}

func (c SessionClass) Valid() bool { return c == SessionInteractive || c == SessionAutonomous }

// AllowsRender is the render-class XOR: a trace can carry user-facing or
// autonomous directives, never both. Stop is valid for either class.
func (c SessionClass) AllowsRender(render RenderKind) bool {
	switch c {
	case SessionInteractive:
		return render == RenderUserSplice || render == RenderReopen || render == RenderStop
	case SessionAutonomous:
		return render == RenderSystemDirective || render == RenderStop
	default:
		return false
	}
}

// DefaultRender makes the move-to-render mapping exhaustive and testable.
func DefaultRender(kind MoveKind, class SessionClass) (RenderKind, bool) {
	if !kind.Valid() || !class.Valid() {
		return "", false
	}
	if kind == MoveHalt {
		return RenderStop, true
	}
	if kind == MoveReanchor && class == SessionInteractive {
		return RenderReopen, true
	}
	if class == SessionInteractive {
		return RenderUserSplice, true
	}
	return RenderSystemDirective, true
}

// Enqueue validates shape and vocabulary but deliberately never evaluates the gate.
func (q *NextQueue) Enqueue(move Move) error {
	if q == nil {
		return errors.New("sessionctl next: nil queue")
	}
	if !move.Kind.Valid() {
		return fmt.Errorf("sessionctl next: invalid move kind %q", move.Kind)
	}
	if !move.Render.Valid() {
		return fmt.Errorf("sessionctl next: invalid render %q", move.Render)
	}
	if strings.TrimSpace(move.Gate) == "" {
		return errors.New("sessionctl next: gate is required")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.drained {
		return errors.New("sessionctl next: queue already drained")
	}
	if move.Session == "" {
		move.Session = q.class
	}
	if move.Session != q.class || !q.class.AllowsRender(move.Render) {
		return fmt.Errorf("sessionctl next: render %q does not belong to %q trace", move.Render, q.class)
	}
	q.next++
	q.moves = append(q.moves, queuedMove{sequence: q.next, move: move})
	return nil
}

// Drain is the one actuation boundary. It writes one JSONL witness for every
// attempted move, including no-ops, refusals, and applier errors.
func (q *NextQueue) Drain(w io.Writer, apply MoveApplier) error {
	if q == nil || w == nil || apply == nil {
		return errors.New("sessionctl next: drain requires queue, witness, and applier")
	}
	q.mu.Lock()
	if q.drained {
		q.mu.Unlock()
		return errors.New("sessionctl next: queue already drained")
	}
	q.drained = true
	moves := append([]queuedMove(nil), q.moves...)
	q.mu.Unlock()

	enc := json.NewEncoder(w)
	for _, item := range moves {
		result, applyErr := apply(item.move)
		if applyErr != nil {
			result.Applied = false
			result.Refusal = applyErr.Error()
		}
		record := NextRecord{Sequence: item.sequence, At: time.Now().UTC(), Move: item.move, Applied: result.Applied, Refusal: strings.TrimSpace(result.Refusal)}
		if err := enc.Encode(record); err != nil {
			return fmt.Errorf("sessionctl next: write witness: %w", err)
		}
	}
	return nil
}

// WitnessMove runs one move through the shared enqueue/drain boundary and returns
// the decoded witness row. Callers persist that row in their own durable ledger;
// they do not invent a parallel decision schema.
func WitnessMove(move Move, result ApplyResult) (NextRecord, error) {
	if move.Session == "" {
		return NextRecord{}, errors.New("sessionctl next: witness move requires session class")
	}
	q, err := NewNextQueue(move.Session)
	if err != nil {
		return NextRecord{}, err
	}
	if err := q.Enqueue(move); err != nil {
		return NextRecord{}, err
	}
	var witness strings.Builder
	if err := q.Drain(&witness, func(Move) (ApplyResult, error) { return result, nil }); err != nil {
		return NextRecord{}, err
	}
	records, err := ReadNextRecords(strings.NewReader(witness.String()))
	if err != nil {
		return NextRecord{}, err
	}
	if len(records) != 1 {
		return NextRecord{}, fmt.Errorf("sessionctl next: one-move witness produced %d rows", len(records))
	}
	return records[0], nil
}

// ReadNextRecords re-reads the durable JSONL witness rather than trusting an
// in-memory Drain return.
func ReadNextRecords(r io.Reader) ([]NextRecord, error) {
	if r == nil {
		return nil, errors.New("sessionctl next: nil witness reader")
	}
	var records []NextRecord
	s := bufio.NewScanner(r)
	for s.Scan() {
		var record NextRecord
		if err := json.Unmarshal(s.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("sessionctl next: decode witness: %w", err)
		}
		records = append(records, record)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("sessionctl next: read witness: %w", err)
	}
	return records, nil
}

var (
	steerNextMu sync.Mutex
	steerNext   = map[string][]NextRecord{}
)

// RecordSteerNext lowers one admitted freeform operator steer onto the shared
// interactive Next contract. The caller has already consumed the a2achan message;
// this function records that applied user-splice at the same turn boundary.
func RecordSteerNext(trace, payload string, result ApplyResult) {
	trace = strings.TrimSpace(trace)
	if trace == "" || (payload == "" && result.Applied) {
		return
	}
	move := Move{
		Kind: MoveAnnotate, Render: RenderUserSplice,
		Session: SessionInteractive, Gate: "a2achan-recv",
		Source: "agent-turn-boundary", Payload: payload,
	}
	record, err := WitnessMove(move, result)
	if err != nil {
		return
	}
	steerNextMu.Lock()
	steerNext[trace] = append(steerNext[trace], record)
	steerNextMu.Unlock()
}

// ReadSteerNextRecords returns and clears the independently re-readable Next
// witnesses for steer messages applied to trace. An empty mailbox yields no rows.
func ReadSteerNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	steerNextMu.Lock()
	defer steerNextMu.Unlock()
	records := append([]NextRecord(nil), steerNext[trace]...)
	delete(steerNext, trace)
	return records
}

var (
	advisoryNextMu sync.Mutex
	advisoryNext   = map[string][]NextRecord{}
)

// RecordContextAdvisoryNext lowers one consumed context-spike advisory onto the
// shared interactive Next contract. The advisory is rendered as a user splice;
// it annotates the live objective rather than redirecting it.
func RecordContextAdvisoryNext(trace, payload string) {
	trace = strings.TrimSpace(trace)
	if trace == "" || payload == "" {
		return
	}
	move := Move{
		Kind: MoveAnnotate, Render: RenderUserSplice,
		Session: SessionInteractive, Gate: "context-spike",
		Source: "agent-turn-boundary", Payload: payload,
	}
	record, err := WitnessMove(move, ApplyResult{Applied: true})
	if err != nil {
		return
	}
	advisoryNextMu.Lock()
	advisoryNext[trace] = append(advisoryNext[trace], record)
	advisoryNextMu.Unlock()
}

// ReadContextAdvisoryNextRecords returns and clears independently re-readable
// Next witnesses for context advisories consumed by trace.
var (
	stopWitnessNextMu sync.Mutex
	stopWitnessNext   = map[string][]NextRecord{}
)

// RecordStopWitnessNext lowers one denied final answer onto the shared
// interactive Next contract. The denial is rendered as a user splice that
// continues the same objective until the named witness exists.
func RecordStopWitnessNext(trace, payload string) {
	trace = strings.TrimSpace(trace)
	if trace == "" || payload == "" {
		return
	}
	move := Move{
		Kind: MoveContinue, Render: RenderUserSplice,
		Session: SessionInteractive, Gate: "stop-witness",
		Source: "agent-turn-boundary", Payload: payload,
	}
	record, err := WitnessMove(move, ApplyResult{Applied: true})
	if err != nil {
		return
	}
	stopWitnessNextMu.Lock()
	stopWitnessNext[trace] = append(stopWitnessNext[trace], record)
	stopWitnessNextMu.Unlock()
}

// ReadStopWitnessNextRecords returns and clears independently re-readable Next
// witnesses for final-answer denials consumed by trace.
func ReadStopWitnessNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	stopWitnessNextMu.Lock()
	defer stopWitnessNextMu.Unlock()
	records := append([]NextRecord(nil), stopWitnessNext[trace]...)
	delete(stopWitnessNext, trace)
	return records
}

var (
	budgetResetNextMu sync.Mutex
	budgetResetNext   = map[string][]NextRecord{}
)

// RecordBudgetResetNext lowers one context-budget reset onto the shared
// interactive Next contract. The recap re-anchors a newly reopened child
// session while preserving the exact system-message payload.
func RecordBudgetResetNext(trace, payload string) {
	RecordBudgetResetNextResult(trace, payload, ApplyResult{Applied: true})
}

// RecordBudgetResetNextResult records the physical reset-seed outcome. Gateway
// front doors call it only after they splice the seed and attempt fresh-child
// admission, so transaction creation alone can never masquerade as actuation.
func RecordBudgetResetNextResult(trace, payload string, result ApplyResult) {
	trace = strings.TrimSpace(trace)
	if trace == "" || payload == "" {
		return
	}
	move := Move{
		Kind: MoveReanchor, Render: RenderReopen,
		Session: SessionInteractive, Gate: "served-session-reset",
		Source: "gateway-reset-hook", Payload: payload,
	}
	record, err := WitnessMove(move, result)
	if err != nil {
		return
	}
	budgetResetNextMu.Lock()
	budgetResetNext[trace] = append(budgetResetNext[trace], record)
	budgetResetNextMu.Unlock()
}

// ReadBudgetResetNextRecords returns and clears independently re-readable
// Next witnesses for context-budget reopens keyed by the child trace.
func ReadBudgetResetNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	budgetResetNextMu.Lock()
	defer budgetResetNextMu.Unlock()
	records := append([]NextRecord(nil), budgetResetNext[trace]...)
	delete(budgetResetNext, trace)
	return records
}

var (
	guardRecoveryNextMu sync.Mutex
	guardRecoveryNext   = map[string][]NextRecord{}
)

// RecordGuardRecoveryNext lowers one consumed gateway recovery prompt onto the
// shared interactive Next contract. The prompt redirects the next live user turn
// away from a previously refused action.
func RecordGuardRecoveryNext(trace, payload string) {
	trace = strings.TrimSpace(trace)
	if trace == "" || payload == "" {
		return
	}
	move := Move{
		Kind: MoveRedirect, Render: RenderUserSplice,
		Session: SessionInteractive, Gate: "guard-recovery",
		Source: "gateway-anthropic-boundary", Payload: payload,
	}
	record, err := WitnessMove(move, ApplyResult{Applied: true})
	if err != nil {
		return
	}
	guardRecoveryNextMu.Lock()
	guardRecoveryNext[trace] = append(guardRecoveryNext[trace], record)
	guardRecoveryNextMu.Unlock()
}

// ReadGuardRecoveryNextRecords returns and clears independently re-readable
// recovery-prompt Next witnesses for trace.
func ReadGuardRecoveryNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	guardRecoveryNextMu.Lock()
	defer guardRecoveryNextMu.Unlock()
	records := append([]NextRecord(nil), guardRecoveryNext[trace]...)
	delete(guardRecoveryNext, trace)
	return records
}
func ReadContextAdvisoryNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	advisoryNextMu.Lock()
	defer advisoryNextMu.Unlock()
	records := append([]NextRecord(nil), advisoryNext[trace]...)
	delete(advisoryNext, trace)
	return records
}

var (
	toolTerminalWakeNextMu sync.Mutex
	toolTerminalWakeNext   = map[string][]NextRecord{}
)

// RecordToolTerminalWakeNext lowers one consumed background-tool terminal wake
// onto the shared interactive Next contract at the turn boundary that renders it.
func RecordToolTerminalWakeNext(trace, payload string) {
	trace = strings.TrimSpace(trace)
	if trace == "" || payload == "" {
		return
	}
	move := Move{
		Kind: MoveContinue, Render: RenderUserSplice,
		Session: SessionInteractive, Gate: "tool-terminal-wake",
		Source: "agent-turn-boundary", Payload: payload,
	}
	record, err := WitnessMove(move, ApplyResult{Applied: true})
	if err != nil {
		return
	}
	toolTerminalWakeNextMu.Lock()
	toolTerminalWakeNext[trace] = append(toolTerminalWakeNext[trace], record)
	toolTerminalWakeNextMu.Unlock()
}

// ReadToolTerminalWakeNextRecords returns and clears independently re-readable
// terminal-wake Next witnesses for trace.
func ReadToolTerminalWakeNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	toolTerminalWakeNextMu.Lock()
	defer toolTerminalWakeNextMu.Unlock()
	records := append([]NextRecord(nil), toolTerminalWakeNext[trace]...)
	delete(toolTerminalWakeNext, trace)
	return records
}
