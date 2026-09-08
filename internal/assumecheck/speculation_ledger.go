package assumecheck

// speculation_ledger.go — the speculation ledger (#4106): a speculative tool call
// is a load-bearing assumption recorded in the assumecheck substrate; witness
// accept/reject, price the mispredict, and fail closed on effect leaks.
//
// In agentic execution with speculative dispatch (e.g. streaming FSM, heuristic,
// or lookahead drafting), tool calls are executed ahead of the authoritative
// model turn. Every speculative invocation is an ASSUMPTION: that the speculative
// call matches what the model will authoritatively emit, that the tool was provably
// effect-free (read-only and idempotent), and that no provisional state leaks if
// the branch is squashed.
//
// This ledger integrates speculative tracking into assumecheck:
//   - Each speculation is recorded with provenance witness (ReadOnlyProof) and proposer.
//   - Clean commits are witnessed and track saved duration (latency speedup).
//   - Mispredictions are witnessed and squashed with SPECULATION_MISPREDICT, pricing
//     wasted tokens and wasted duration.
//   - Any effect leak triggers a fail-closed audit refusal with SPECULATIVE_EFFECT_LEAK.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Closed refusal tokens for speculation failures.
const (
	TokenSpeculationMispredict = "SPECULATION_MISPREDICT"
	TokenSpeculativeEffectLeak = "SPECULATIVE_EFFECT_LEAK"
)

// SpeculationOutcome represents the lifecycle disposition of a speculative call.
type SpeculationOutcome string

const (
	// OutcomePending — speculative call recorded and running/awaiting authoritative resolution.
	OutcomePending SpeculationOutcome = "PENDING"
	// OutcomeCommitted — authoritative call matched prediction; provisional effects promoted.
	OutcomeCommitted SpeculationOutcome = "COMMITTED"
	// OutcomeSquashed — authoritative call differed; speculative branch discarded.
	OutcomeSquashed SpeculationOutcome = "SQUASHED"
	// OutcomeLeaked — speculative call attempted or caused an unauthorized side-effect leak.
	OutcomeLeaked SpeculationOutcome = "LEAKED"
)

// CostAccounting tracks the computational and latency economics of speculative execution.
type CostAccounting struct {
	WastedTokens    int64         `json:"wasted_tokens"`
	WastedDuration  time.Duration `json:"wasted_duration"`
	SavedDuration   time.Duration `json:"saved_duration"`
	NetSpeedupRatio float64       `json:"net_speedup_ratio"`
}

// SpeculationEntry records a single speculative tool invocation as an assumption.
type SpeculationEntry struct {
	ID                  string             `json:"id"`
	Timestamp           time.Time          `json:"timestamp"`
	RunID               string             `json:"run_id,omitempty"`
	TraceID             string             `json:"trace_id,omitempty"`
	ToolName            string             `json:"tool_name"`
	CallArguments       string             `json:"call_arguments"`
	ActualCallDigest    string             `json:"actual_call_digest,omitempty"`
	SpeculativeProposer string             `json:"speculative_proposer"`
	ReadOnlyProof       string             `json:"read_only_proof"`
	Outcome             SpeculationOutcome `json:"outcome"`
	Reason              string             `json:"reason,omitempty"`
	FailureToken        string             `json:"failure_token,omitempty"`
	CostAccounting      CostAccounting     `json:"cost_accounting"`
}

// AsAssumption projects the speculative entry into an assumecheck.Assumption.
// A speculative call is an assumption: "speculative invocation of tool X matches
// authoritative call without effect leak".
func (e SpeculationEntry) AsAssumption() Assumption {
	owner := e.SpeculativeProposer
	if owner == "" {
		owner = "speculator"
	}
	refusal := ""
	switch e.Outcome {
	case OutcomeSquashed:
		refusal = TokenSpeculationMispredict
	case OutcomeLeaked:
		refusal = TokenSpeculativeEffectLeak
	}
	return Assumption{
		ID:              "speculation:" + e.ID,
		Owner:           owner,
		Statement:       fmt.Sprintf("speculative invocation of tool %q matches authoritative call without effect leak", e.ToolName),
		Level:           LevelSubsystem,
		WitnessKind:     WitnessLedgerRead,
		RefusalReason:   refusal,
		ConfidenceClass: "speculative",
		WitnessStatus:   WitnessWired,
	}
}

// AsVerdict projects the speculative entry's current outcome into an assumecheck.Verdict.
func (e SpeculationEntry) AsVerdict() Verdict {
	a := e.AsAssumption()
	var out Outcome
	reason := e.Reason
	switch e.Outcome {
	case OutcomeCommitted:
		out = OutcomeHolds
		if reason == "" {
			reason = "speculation matched authoritative call and committed"
		}
	case OutcomeSquashed:
		out = OutcomeViolated
		if reason == "" {
			reason = "speculation mispredicted; branch squashed"
		}
	case OutcomeLeaked:
		out = OutcomeViolated
		if reason == "" {
			reason = "speculative effect leak detected; fail-closed audit refusal"
		}
	default:
		out = OutcomeUnverifiable
		if reason == "" {
			reason = "speculation pending authoritative witness"
		}
	}
	return Verdict{
		AssumptionID: a.ID,
		Level:        a.Level,
		Witness:      a.WitnessKind,
		Outcome:      out,
		Reason:       reason,
	}
}

// SpeculationSummary provides aggregated performance and accounting metrics.
type SpeculationSummary struct {
	TotalSpeculations int     `json:"total_speculations"`
	CommittedCount    int     `json:"committed_count"`
	SquashedCount     int     `json:"squashed_count"`
	LeakedCount       int     `json:"leaked_count"`
	PendingCount      int     `json:"pending_count,omitempty"`
	TotalSavedMs      float64 `json:"total_saved_ms"`
	TotalWastedMs     float64 `json:"total_wasted_ms"`
	TotalWastedTokens int64   `json:"total_wasted_tokens"`
	AcceptanceRate    float64 `json:"acceptance_rate"`
	NetSpeedup        float64 `json:"net_speedup"`
}

var (
	// ErrSpeculativeEffectLeak is the sentinel base error for speculative effect leaks.
	ErrSpeculativeEffectLeak = errors.New("speculative effect leak detected: fail-closed audit refusal")
	// ErrSpeculationNotFound indicates the requested speculation ID does not exist.
	ErrSpeculationNotFound = errors.New("speculation entry not found")
	// ErrSpeculationDuplicate indicates an entry with the given ID already exists.
	ErrSpeculationDuplicate = errors.New("speculation entry already exists")
)

// AuditRefusalError represents a fail-closed audit refusal triggered by a speculative anomaly.
type AuditRefusalError struct {
	Token    string `json:"token"`
	Reason   string `json:"reason"`
	EntryID  string `json:"entry_id"`
	Evidence string `json:"evidence"`
}

func (e *AuditRefusalError) Error() string {
	return fmt.Sprintf("fail-closed audit refusal [%s]: entry %s: %s (evidence: %s)",
		e.Token, e.EntryID, e.Reason, e.Evidence)
}

func (e *AuditRefusalError) Unwrap() error {
	return ErrSpeculativeEffectLeak
}

// SpeculationLedger is the thread-safe ledger of speculative tool call assumptions.
type SpeculationLedger struct {
	mu      sync.RWMutex
	entries map[string]*SpeculationEntry
	order   []string
}

// NewSpeculationLedger instantiates an empty speculation ledger.
func NewSpeculationLedger() *SpeculationLedger {
	return &SpeculationLedger{
		entries: make(map[string]*SpeculationEntry),
		order:   make([]string, 0),
	}
}

// RecordSpeculation records a new speculative tool call assumption.
func (l *SpeculationLedger) RecordSpeculation(entry SpeculationEntry) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if entry.ID == "" {
		return errors.New("speculation entry id is required")
	}
	if _, exists := l.entries[entry.ID]; exists {
		return fmt.Errorf("%w: %s", ErrSpeculationDuplicate, entry.ID)
	}
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	if entry.Outcome == "" {
		entry.Outcome = OutcomePending
	}

	copied := entry
	l.entries[entry.ID] = &copied
	l.order = append(l.order, entry.ID)
	return nil
}

// WitnessCommit records a successful speculation match where provisional effects
// are promoted and committed cleanly, tracking the saved latency duration.
func (l *SpeculationLedger) WitnessCommit(id, actualCallDigest string, savedDuration time.Duration) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.entries[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrSpeculationNotFound, id)
	}

	entry.Outcome = OutcomeCommitted
	entry.ActualCallDigest = actualCallDigest
	entry.Reason = ""
	entry.FailureToken = ""
	entry.CostAccounting.SavedDuration = savedDuration
	entry.CostAccounting.WastedDuration = 0
	entry.CostAccounting.NetSpeedupRatio = calcNetSpeedup(savedDuration, 0)
	return nil
}

// WitnessSquash records an incorrect speculation where the speculative branch is squashed,
// pricing the misprediction with wasted duration and wasted token accounting.
func (l *SpeculationLedger) WitnessSquash(id, mispredictReason string, wastedDuration time.Duration, wastedTokens int64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.entries[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrSpeculationNotFound, id)
	}

	entry.Outcome = OutcomeSquashed
	entry.Reason = mispredictReason
	entry.FailureToken = TokenSpeculationMispredict
	if entry.FailureToken == "" {
		entry.FailureToken = abi.ReasonName(abi.ReasonSpeculationMispredict)
	}
	entry.CostAccounting.WastedDuration = wastedDuration
	entry.CostAccounting.WastedTokens = wastedTokens
	entry.CostAccounting.SavedDuration = 0
	entry.CostAccounting.NetSpeedupRatio = calcNetSpeedup(0, wastedDuration)
	return nil
}

// WitnessLeak witnesses an unauthorized side-effect leak from a speculative execution.
// It marks the entry as LEAKED and immediately triggers a fail-closed audit refusal.
func (l *SpeculationLedger) WitnessLeak(id, leakEvidence string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.entries[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrSpeculationNotFound, id)
	}

	entry.Outcome = OutcomeLeaked
	entry.Reason = fmt.Sprintf("speculative effect leak: %s", leakEvidence)
	entry.FailureToken = TokenSpeculativeEffectLeak
	if entry.FailureToken == "" {
		entry.FailureToken = abi.ReasonName(abi.ReasonSpeculativeEffectLeak)
	}
	entry.CostAccounting.NetSpeedupRatio = 0.0

	return &AuditRefusalError{
		Token:    TokenSpeculativeEffectLeak,
		Reason:   entry.Reason,
		EntryID:  id,
		Evidence: leakEvidence,
	}
}

// summaryLocked calculates aggregate metrics assuming l.mu is already held.
func (l *SpeculationLedger) summaryLocked() SpeculationSummary {
	s := SpeculationSummary{
		TotalSpeculations: len(l.order),
	}

	var totalSaved time.Duration
	var totalWasted time.Duration

	for _, id := range l.order {
		e := l.entries[id]
		switch e.Outcome {
		case OutcomeCommitted:
			s.CommittedCount++
		case OutcomeSquashed:
			s.SquashedCount++
		case OutcomeLeaked:
			s.LeakedCount++
		case OutcomePending:
			s.PendingCount++
		}
		totalSaved += e.CostAccounting.SavedDuration
		totalWasted += e.CostAccounting.WastedDuration
		s.TotalWastedTokens += e.CostAccounting.WastedTokens
	}

	s.TotalSavedMs = float64(totalSaved.Microseconds()) / 1000.0
	s.TotalWastedMs = float64(totalWasted.Microseconds()) / 1000.0

	if s.TotalSpeculations > 0 {
		s.AcceptanceRate = math.Round((float64(s.CommittedCount)/float64(s.TotalSpeculations))*1000) / 1000
	}
	s.NetSpeedup = calcNetSpeedup(totalSaved, totalWasted)
	return s
}

// Summary returns aggregate metrics across all recorded speculations in the ledger.
func (l *SpeculationLedger) Summary() SpeculationSummary {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.summaryLocked()
}

// SpeculationExport is the serializable structure for exported ledger states.
type SpeculationExport struct {
	Entries []SpeculationEntry `json:"entries"`
	Summary SpeculationSummary `json:"summary"`
}

// ExportJSON exports the full ledger entries and summary metrics as formatted JSON.
func (l *SpeculationLedger) ExportJSON() ([]byte, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	entries := make([]SpeculationEntry, len(l.order))
	for i, id := range l.order {
		entries[i] = *l.entries[id]
	}

	export := SpeculationExport{
		Entries: entries,
		Summary: l.summaryLocked(),
	}
	return json.MarshalIndent(export, "", "  ")
}

// MarshalJSON marshals the ledger into JSON via ExportJSON.
func (l *SpeculationLedger) MarshalJSON() ([]byte, error) {
	return l.ExportJSON()
}

// Get returns a copy of the speculation entry for the given ID.
func (l *SpeculationLedger) Get(id string) (SpeculationEntry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	entry, ok := l.entries[id]
	if !ok {
		return SpeculationEntry{}, false
	}
	return *entry, true
}

// Entries returns an ordered snapshot of all recorded speculation entries.
func (l *SpeculationLedger) Entries() []SpeculationEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	res := make([]SpeculationEntry, len(l.order))
	for i, id := range l.order {
		res[i] = *l.entries[id]
	}
	return res
}

// Len returns the number of recorded speculations.
func (l *SpeculationLedger) Len() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.order)
}

// calcNetSpeedup computes the normalized speedup ratio from saved vs wasted durations.
// Break-even is 1.0; 100% saved yields 2.0; 100% wasted yields 0.0.
func calcNetSpeedup(saved, wasted time.Duration) float64 {
	s := float64(saved)
	w := float64(wasted)
	if s+w == 0 {
		return 1.0
	}
	ratio := 1.0 + (s-w)/(s+w)
	return math.Round(ratio*1000) / 1000
}
