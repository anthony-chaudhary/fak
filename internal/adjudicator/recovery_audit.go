package adjudicator

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// Standard outcome tokens for refusal recovery.
const (
	OutcomeRecovered = "recovered"
	OutcomeRetried   = "retried"
	OutcomeAbandoned = "abandoned"
)

// RecoveryAuditEntry records a single refusal event and its subsequent recovery outcome.
type RecoveryAuditEntry struct {
	SessionID           string    `json:"session_id"`
	Turn                int       `json:"turn"`
	RefusalReason       string    `json:"refusal_reason"`
	SuggestedNextAction string    `json:"suggested_next_action"`
	SubsequentAction    string    `json:"subsequent_action"`
	Outcome             string    `json:"outcome"`
	Recovered           bool      `json:"recovered"`
	Timestamp           time.Time `json:"timestamp"`
}

// RecoveryAuditLedger records refusal events and evaluates whether agents successfully recovered.
type RecoveryAuditLedger struct {
	mu      sync.RWMutex
	entries []RecoveryAuditEntry
}

// NewRecoveryAuditLedger allocates an initialized, empty RecoveryAuditLedger.
func NewRecoveryAuditLedger() *RecoveryAuditLedger {
	return &RecoveryAuditLedger{
		entries: make([]RecoveryAuditEntry, 0),
	}
}

// RecordRefusal logs a structured refusal event for a session at a specific turn.
func (l *RecoveryAuditLedger) RecordRefusal(sessionID string, turn int, reason string, nextAction string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry := RecoveryAuditEntry{
		SessionID:           sessionID,
		Turn:                turn,
		RefusalReason:       reason,
		SuggestedNextAction: nextAction,
		Recovered:           false,
		Timestamp:           time.Now().UTC(),
	}
	l.entries = append(l.entries, entry)
}

// RecordOutcome updates the outcome of a prior refusal for a session at a turn.
// If an uncompleted refusal exists for (sessionID, turn), it is updated.
// If no uncompleted refusal exists, it updates the latest matching entry or appends a new entry.
func (l *RecoveryAuditLedger) RecordOutcome(sessionID string, turn int, subsequentAction string, outcome string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	norm := strings.ToLower(strings.TrimSpace(outcome))
	isRecovered := norm == OutcomeRecovered

	for i := len(l.entries) - 1; i >= 0; i-- {
		if l.entries[i].SessionID == sessionID && l.entries[i].Turn == turn && l.entries[i].Outcome == "" {
			l.entries[i].SubsequentAction = subsequentAction
			l.entries[i].Outcome = norm
			l.entries[i].Recovered = isRecovered
			return
		}
	}

	for i := len(l.entries) - 1; i >= 0; i-- {
		if l.entries[i].SessionID == sessionID && l.entries[i].Turn == turn {
			l.entries[i].SubsequentAction = subsequentAction
			l.entries[i].Outcome = norm
			l.entries[i].Recovered = isRecovered
			return
		}
	}

	l.entries = append(l.entries, RecoveryAuditEntry{
		SessionID:        sessionID,
		Turn:             turn,
		SubsequentAction: subsequentAction,
		Outcome:          norm,
		Recovered:        isRecovered,
		Timestamp:        time.Now().UTC(),
	})
}

func (l *RecoveryAuditLedger) countsLocked() (recovered, unrecovered, totalResolved int) {
	for _, e := range l.entries {
		norm := strings.ToLower(strings.TrimSpace(e.Outcome))
		if norm == "" && !e.Recovered {
			continue
		}
		totalResolved++
		if e.Recovered || norm == OutcomeRecovered {
			recovered++
		} else {
			unrecovered++
		}
	}
	return recovered, unrecovered, totalResolved
}

// RecoveryRate computes the proportion of resolved refusals that resulted in successful recovery.
// Returns 0.0 when no resolved outcomes have been recorded.
func (l *RecoveryAuditLedger) RecoveryRate() float64 {
	l.mu.RLock()
	defer l.mu.RUnlock()

	recovered, _, totalResolved := l.countsLocked()
	if totalResolved == 0 {
		return 0.0
	}
	return float64(recovered) / float64(totalResolved)
}

// AttritionRate computes the proportion of resolved refusals that resulted in abandonment or failure to recover.
// Returns 0.0 when no resolved outcomes have been recorded.
func (l *RecoveryAuditLedger) AttritionRate() float64 {
	l.mu.RLock()
	defer l.mu.RUnlock()

	_, unrecovered, totalResolved := l.countsLocked()
	if totalResolved == 0 {
		return 0.0
	}
	return float64(unrecovered) / float64(totalResolved)
}

// TotalResolved returns the number of entries that have resolved outcomes.
func (l *RecoveryAuditLedger) TotalResolved() int {
	l.mu.RLock()
	defer l.mu.RUnlock()

	_, _, totalResolved := l.countsLocked()
	return totalResolved
}

// Len returns the total number of entries in the ledger.
func (l *RecoveryAuditLedger) Len() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.entries)
}

// Entries returns an isolated shallow copy of the ledger entries.
func (l *RecoveryAuditLedger) Entries() []RecoveryAuditEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	out := make([]RecoveryAuditEntry, len(l.entries))
	copy(out, l.entries)
	return out
}

// WriteJSONL serializes all ledger entries as newline-delimited JSON objects to w.
func (l *RecoveryAuditLedger) WriteJSONL(w io.Writer) error {
	l.mu.RLock()
	defer l.mu.RUnlock()

	enc := json.NewEncoder(w)
	for _, entry := range l.entries {
		if err := enc.Encode(entry); err != nil {
			return fmt.Errorf("recovery_audit: encode entry: %w", err)
		}
	}
	return nil
}

// ReadJSONL parses newline-delimited JSON objects from r and appends them to the ledger.
func (l *RecoveryAuditLedger) ReadJSONL(r io.Reader) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	reader := bufio.NewReader(r)
	var loaded []RecoveryAuditEntry
	lineNum := 0
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) == 0 && err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("recovery_audit: read jsonl: %w", err)
		}
		lineNum++
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) > 0 {
			var entry RecoveryAuditEntry
			if decErr := json.Unmarshal(trimmed, &entry); decErr != nil {
				return fmt.Errorf("recovery_audit: decode line %d: %w", lineNum, decErr)
			}
			loaded = append(loaded, entry)
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("recovery_audit: read jsonl: %w", err)
		}
	}
	l.entries = append(l.entries, loaded...)
	return nil
}
