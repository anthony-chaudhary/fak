package sessionledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	ToolCallPreparedKind     = "tool-call-prepared"
	ToolDispatchStartedKind  = "tool-dispatch-started"
	ToolCallTerminalKind     = "tool-call-terminal"
	defaultToolCallArguments = "{}"
)

// ToolCall is the durable identity needed to retry a call that provably never
// crossed the dispatch boundary.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ToolTerminalStatus string

const (
	ToolResult  ToolTerminalStatus = "result"
	ToolRefusal ToolTerminalStatus = "refusal"
	ToolSkipped ToolTerminalStatus = "skipped"
)

// ToolTerminal records every state that ends a call without further dispatch.
type ToolTerminal struct {
	CallID  string             `json:"call_id"`
	Status  ToolTerminalStatus `json:"status"`
	Content json.RawMessage    `json:"content,omitempty"`
}

type ToolRepairState string

const (
	ToolCompleted             ToolRepairState = "completed"
	ToolNeverStarted          ToolRepairState = "never-started"
	ToolStartedOutcomeUnknown ToolRepairState = "started/outcome-unknown"
)

type ToolRepairReason string

const ToolOutcomeUnknown ToolRepairReason = "TOOL_OUTCOME_UNKNOWN"

// ToolRepairReceipt is the restart decision for one prepared call. AutoRetry is
// true only when the durable chain proves dispatch never started.
type ToolRepairReceipt struct {
	Call      ToolCall         `json:"call"`
	State     ToolRepairState  `json:"state"`
	Reason    ToolRepairReason `json:"reason,omitempty"`
	AutoRetry bool             `json:"auto_retry"`
	Terminal  *ToolTerminal    `json:"terminal,omitempty"`
}

type toolCallMarker struct {
	CallID string `json:"call_id"`
}

// PrepareToolCall durably records identity and arguments before a caller may
// dispatch. An oversized argument set is refused because eliding it would make a
// claimed safe retry impossible to reproduce.
func (l *Ledger) PrepareToolCall(trace string, call ToolCall) (Entry, error) {
	if err := validateToolCall(call); err != nil {
		return Entry{}, err
	}
	call.ID = strings.TrimSpace(call.ID)
	call.Name = strings.TrimSpace(call.Name)
	call.Arguments = bytes.Clone(call.Arguments)
	if len(call.Arguments) == 0 {
		call.Arguments = json.RawMessage(defaultToolCallArguments)
	}
	content, err := json.Marshal(call)
	if err != nil {
		return Entry{}, fmt.Errorf("sessionledger: marshal prepared tool call: %w", err)
	}
	if len(content) > MaxContentBytes {
		return Entry{}, fmt.Errorf("sessionledger: prepared tool call %q exceeds the %d-byte durable record bound", call.ID, MaxContentBytes)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	receipts, err := l.toolRepairReceiptsLocked(trace, false)
	if err != nil {
		return Entry{}, err
	}
	if _, ok := toolReceiptByID(receipts, call.ID); ok {
		return Entry{}, fmt.Errorf("sessionledger: tool call %q is already prepared on trace %q", call.ID, trace)
	}
	return l.appendDurableToolEntryLocked(trace, ToolCallPreparedKind, content)
}

// MarkToolDispatchStarted is the no-return boundary. Callers must persist this
// marker immediately before invoking the tool; after it exists, a missing
// terminal record is outcome-unknown and cannot be retried automatically.
func (l *Ledger) MarkToolDispatchStarted(trace, callID string) (Entry, error) {
	callID = strings.TrimSpace(callID)
	if callID == "" {
		return Entry{}, errors.New("sessionledger: tool dispatch start needs a call id")
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	receipts, err := l.toolRepairReceiptsLocked(trace, true)
	if err != nil {
		return Entry{}, err
	}
	receipt, ok := toolReceiptByID(receipts, callID)
	if !ok {
		return Entry{}, fmt.Errorf("sessionledger: tool call %q was not prepared on trace %q", callID, trace)
	}
	if receipt.State != ToolNeverStarted {
		return Entry{}, fmt.Errorf("sessionledger: tool call %q cannot start from state %q", callID, receipt.State)
	}
	content, err := json.Marshal(toolCallMarker{CallID: callID})
	if err != nil {
		return Entry{}, err
	}
	return l.appendDurableToolEntryLocked(trace, ToolDispatchStartedKind, content)
}

// RecordToolTerminal persists a result, refusal, or skip. A terminal record wins
// over the start marker during repair and is returned for completed reuse.
func (l *Ledger) RecordToolTerminal(trace string, terminal ToolTerminal) (Entry, error) {
	if err := validateToolTerminal(terminal); err != nil {
		return Entry{}, err
	}
	terminal.CallID = strings.TrimSpace(terminal.CallID)
	terminal.Content = bytes.Clone(terminal.Content)
	content, err := marshalBoundedTerminal(terminal)
	if err != nil {
		return Entry{}, err
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	receipts, err := l.toolRepairReceiptsLocked(trace, true)
	if err != nil {
		return Entry{}, err
	}
	receipt, ok := toolReceiptByID(receipts, terminal.CallID)
	if !ok {
		return Entry{}, fmt.Errorf("sessionledger: tool call %q was not prepared on trace %q", terminal.CallID, trace)
	}
	if receipt.State == ToolCompleted {
		return Entry{}, fmt.Errorf("sessionledger: tool call %q already has a terminal record", terminal.CallID)
	}
	return l.appendDurableToolEntryLocked(trace, ToolCallTerminalKind, content)
}

// RepairToolCalls reconstructs all prepared calls in ledger order. It never
// infers that a missing result means a call did not run: a start without a
// terminal record is fenced with TOOL_OUTCOME_UNKNOWN.
func (l *Ledger) RepairToolCalls(trace string) ([]ToolRepairReceipt, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.toolRepairReceiptsLocked(trace, true)
}

func (l *Ledger) toolRepairReceiptsLocked(trace string, requireTrace bool) ([]ToolRepairReceipt, error) {
	if strings.TrimSpace(trace) == "" {
		return nil, errors.New("sessionledger: tool repair needs a trace")
	}
	head, ok := l.heads[trace]
	if !ok {
		if requireTrace {
			return nil, fmt.Errorf("sessionledger: trace %q not found", trace)
		}
		return nil, nil
	}
	entries, err := l.chainFromLocked(head)
	if err != nil {
		return nil, err
	}

	receipts := make([]ToolRepairReceipt, 0)
	index := make(map[string]int)
	for _, entry := range entries {
		switch entry.Kind {
		case ToolCallPreparedKind:
			var call ToolCall
			if err := json.Unmarshal(entry.Content, &call); err != nil {
				return nil, fmt.Errorf("sessionledger: decode prepared tool call: %w", err)
			}
			if err := validateToolCall(call); err != nil {
				return nil, err
			}
			call.ID = strings.TrimSpace(call.ID)
			call.Name = strings.TrimSpace(call.Name)
			if _, duplicate := index[call.ID]; duplicate {
				return nil, fmt.Errorf("sessionledger: duplicate prepared tool call %q", call.ID)
			}
			call.Arguments = bytes.Clone(call.Arguments)
			index[call.ID] = len(receipts)
			receipts = append(receipts, ToolRepairReceipt{Call: call, State: ToolNeverStarted, AutoRetry: true})

		case ToolDispatchStartedKind:
			var marker toolCallMarker
			if err := json.Unmarshal(entry.Content, &marker); err != nil {
				return nil, fmt.Errorf("sessionledger: decode tool dispatch start: %w", err)
			}
			i, exists := index[marker.CallID]
			if !exists {
				return nil, fmt.Errorf("sessionledger: tool dispatch start %q has no prepared call", marker.CallID)
			}
			if receipts[i].State != ToolNeverStarted {
				return nil, fmt.Errorf("sessionledger: duplicate or late tool dispatch start %q", marker.CallID)
			}
			receipts[i].State = ToolStartedOutcomeUnknown
			receipts[i].Reason = ToolOutcomeUnknown
			receipts[i].AutoRetry = false

		case ToolCallTerminalKind:
			var terminal ToolTerminal
			if err := json.Unmarshal(entry.Content, &terminal); err != nil {
				return nil, fmt.Errorf("sessionledger: decode tool terminal: %w", err)
			}
			if err := validateToolTerminal(terminal); err != nil {
				return nil, err
			}
			i, exists := index[terminal.CallID]
			if !exists {
				return nil, fmt.Errorf("sessionledger: tool terminal %q has no prepared call", terminal.CallID)
			}
			if receipts[i].State == ToolCompleted {
				return nil, fmt.Errorf("sessionledger: duplicate tool terminal %q", terminal.CallID)
			}
			terminal.Content = bytes.Clone(terminal.Content)
			receipts[i].State = ToolCompleted
			receipts[i].Reason = ""
			receipts[i].AutoRetry = false
			receipts[i].Terminal = &terminal
		}
	}
	return receipts, nil
}

func validateToolCall(call ToolCall) error {
	if strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Name) == "" {
		return errors.New("sessionledger: tool call id and name are required")
	}
	if len(call.Arguments) > 0 && !json.Valid(call.Arguments) {
		return fmt.Errorf("sessionledger: tool call %q arguments must be JSON", call.ID)
	}
	return nil
}

func validateToolTerminal(terminal ToolTerminal) error {
	if strings.TrimSpace(terminal.CallID) == "" {
		return errors.New("sessionledger: tool terminal needs a call id")
	}
	switch terminal.Status {
	case ToolResult, ToolRefusal, ToolSkipped:
	default:
		return fmt.Errorf("sessionledger: tool terminal %q has invalid status %q", terminal.CallID, terminal.Status)
	}
	if len(terminal.Content) > 0 && !json.Valid(terminal.Content) {
		return fmt.Errorf("sessionledger: tool terminal %q content must be JSON", terminal.CallID)
	}
	return nil
}

func marshalBoundedTerminal(terminal ToolTerminal) ([]byte, error) {
	content, err := json.Marshal(terminal)
	if err != nil {
		return nil, fmt.Errorf("sessionledger: marshal tool terminal: %w", err)
	}
	if len(content) <= MaxContentBytes {
		return content, nil
	}
	terminal.Content = Elide(terminal.Content)
	content, err = json.Marshal(terminal)
	if err != nil {
		return nil, fmt.Errorf("sessionledger: marshal elided tool terminal: %w", err)
	}
	return content, nil
}

func toolReceiptByID(receipts []ToolRepairReceipt, callID string) (ToolRepairReceipt, bool) {
	for _, receipt := range receipts {
		if receipt.Call.ID == callID {
			return receipt, true
		}
	}
	return ToolRepairReceipt{}, false
}

// appendDurableToolEntryLocked writes and syncs the boundary before exposing it
// in memory. A caller may dispatch only after this function returns success.
func (l *Ledger) appendDurableToolEntryLocked(trace, kind string, content []byte) (Entry, error) {
	parent := l.heads[trace]
	e := Entry{Parent: parent, Kind: kind, Content: bytes.Clone(content)}
	e.Hash = digest(parent, kind, e.Content)
	r := record{Trace: trace, Hash: e.Hash, Parent: e.Parent, Kind: e.Kind, Content: e.Content}
	if err := l.appendDurableRecord(r); err != nil {
		return Entry{}, err
	}
	l.putNode(e)
	l.heads[trace] = e.Hash
	return e, nil
}

func (l *Ledger) appendDurableRecord(r record) error {
	if l.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0700); err != nil {
		return err
	}
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	l.rotateIfLarge(len(line))

	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err = f.Write(line); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}
