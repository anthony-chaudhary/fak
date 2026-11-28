package agent

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// SessionEventKind is the closed OpenHands-derived event-stream discriminator.
type SessionEventKind string

const (
	EventKindAction      SessionEventKind = "action"
	EventKindObservation SessionEventKind = "observation"
)

// SessionEvent is one append-only JSONL record in a session's Action/Observation log.
type SessionEvent struct {
	ID         int64            `json:"id"`
	Timestamp  time.Time        `json:"timestamp"`
	Kind       SessionEventKind `json:"kind"`
	Turn       int              `json:"turn"`
	Source     string           `json:"source"`
	Model      string           `json:"model,omitempty"`
	Messages   []Message        `json:"messages,omitempty"`
	ToolName   string           `json:"tool_name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Content    string           `json:"content,omitempty"`
	IsError    bool             `json:"is_error,omitempty"`
}

// SessionEventLog is a thread-safe append-only JSONL event stream for one session.
type SessionEventLog struct {
	mu     sync.Mutex
	path   string
	nextID int64
}

// NewSessionEventLog opens or creates an append-only JSONL event log at path,
// resuming the monotonic ID counter from any existing events on disk.
func NewSessionEventLog(path string) (*SessionEventLog, error) {
	if path == "" {
		return nil, errors.New("agent: empty session event log path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("agent: mkdir session event log: %w", err)
	}
	var maxID int64
	if _, err := os.Stat(path); err == nil {
		state, err := ReplaySessionEvents(path, 0)
		if err != nil {
			return nil, err
		}
		maxID = state.LastID
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("agent: stat session event log: %w", err)
	}
	return &SessionEventLog{path: path, nextID: maxID + 1}, nil
}

// Append assigns a monotonic ID and timestamp (if zero) and appends ev as one JSON line.
func (l *SessionEventLog) Append(ev SessionEvent) (SessionEvent, error) {
	if ev.Kind != EventKindAction && ev.Kind != EventKindObservation {
		return SessionEvent{}, fmt.Errorf("agent: unknown session event kind %q", ev.Kind)
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	ev.ID = l.nextID
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now().UTC()
	}
	line, err := json.Marshal(ev)
	if err != nil {
		return SessionEvent{}, fmt.Errorf("agent: marshal session event: %w", err)
	}
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return SessionEvent{}, fmt.Errorf("agent: open session event log: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(append(line, '\n')); err != nil {
		return SessionEvent{}, fmt.Errorf("agent: write session event: %w", err)
	}
	l.nextID++
	return ev, nil
}

// ModelRequestObserver returns a loop_session ModelRequestObserver that records each
// ModelRequestBoundary as an Action event before the Planner call proceeds.
func (l *SessionEventLog) ModelRequestObserver() ModelRequestObserver {
	return func(b ModelRequestBoundary) error {
		msgs := make([]Message, len(b.Messages))
		copy(msgs, b.Messages)
		_, err := l.Append(SessionEvent{
			Kind:     EventKindAction,
			Turn:     b.Turn,
			Source:   "model_request",
			Model:    b.Model,
			Messages: msgs,
		})
		return err
	}
}

// RecordObservation appends a tool or environment observation event to the stream.
func (l *SessionEventLog) RecordObservation(turn int, toolCallID, toolName, content string, isError bool) (SessionEvent, error) {
	return l.Append(SessionEvent{
		Kind:       EventKindObservation,
		Turn:       turn,
		Source:     "tool_result",
		ToolCallID: toolCallID,
		ToolName:   toolName,
		Content:    content,
		IsError:    isError,
	})
}

// SessionReplayState holds the state reconstructed by replaying a session event log
// from a cursor ID.
type SessionReplayState struct {
	LastID       int64
	LastTurn     int
	Model        string
	Messages     []Message
	Observations []SessionEvent
	EventCount   int
}

// ReplaySessionEvents reads the JSONL event stream at path, validates monotonic IDs
// and closed event kinds, and reconstructs state for all events with ID > fromID.
func ReplaySessionEvents(path string, fromID int64) (*SessionReplayState, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("agent: open session event log for replay: %w", err)
	}
	defer f.Close()

	state := &SessionReplayState{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var prevID int64
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var ev SessionEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			return nil, fmt.Errorf("agent: replay line %d: %w", lineNum, err)
		}
		if ev.ID <= prevID {
			return nil, fmt.Errorf("agent: replay line %d: non-monotonic id %d <= %d", lineNum, ev.ID, prevID)
		}
		if ev.Kind != EventKindAction && ev.Kind != EventKindObservation {
			return nil, fmt.Errorf("agent: replay line %d: unknown event kind %q", lineNum, ev.Kind)
		}
		prevID = ev.ID
		state.LastID = ev.ID
		if ev.ID <= fromID {
			continue
		}
		state.EventCount++
		if ev.Turn > state.LastTurn {
			state.LastTurn = ev.Turn
		}
		switch ev.Kind {
		case EventKindAction:
			if ev.Model != "" {
				state.Model = ev.Model
			}
			if len(ev.Messages) > 0 {
				state.Messages = append([]Message(nil), ev.Messages...)
			}
		case EventKindObservation:
			state.Observations = append(state.Observations, ev)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("agent: scan session event log: %w", err)
	}
	return state, nil
}
