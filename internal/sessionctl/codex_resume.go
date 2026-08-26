package sessionctl

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// CodexStartMode makes upstream lifecycle intent explicit. A missing thread is
// never silently replaced by a new one.
type CodexStartMode string

const (
	CodexNew    CodexStartMode = "new"
	CodexResume CodexStartMode = "resume"
	CodexFork   CodexStartMode = "fork"
)

type CodexRecoveryReason string

const (
	CodexThreadMissing      CodexRecoveryReason = "thread_missing"
	CodexThreadIncompatible CodexRecoveryReason = "thread_incompatible"
	CodexInputLeaseHeld     CodexRecoveryReason = "input_lease_held"
	CodexStaleEpoch         CodexRecoveryReason = "stale_epoch"
)

// CodexRecoveryError is recoverable by an explicit New, Resume, or Fork choice.
type CodexRecoveryError struct {
	Reason  CodexRecoveryReason
	Choices []CodexStartMode
	Detail  string
}

func (e *CodexRecoveryError) Error() string {
	return fmt.Sprintf("codex session: %s: %s", e.Reason, e.Detail)
}

type CodexCoordinates struct {
	ThreadID       string `json:"thread_id,omitempty"`
	AdapterVersion string `json:"adapter_version"`
}

type CodexAddress struct {
	SessionID string `json:"session_id"`
	Cursor    uint64 `json:"cursor"`
	Epoch     uint64 `json:"epoch"`
}

type CodexSemanticEvent struct {
	Address    CodexAddress        `json:"address"`
	DeliveryID string              `json:"delivery_id"`
	Partial    bool                `json:"partial,omitempty"`
	Event      harnesskit.Envelope `json:"event"`
}

type CodexSessionState struct {
	SessionID   string               `json:"session_id"`
	Coordinates CodexCoordinates     `json:"coordinates"`
	Epoch       uint64               `json:"epoch"`
	Cursor      uint64               `json:"cursor"`
	Events      []CodexSemanticEvent `json:"events,omitempty"`
}

type CodexStateStore interface {
	Load(sessionID string) (CodexSessionState, error)
	Save(CodexSessionState) error
}

var ErrCodexStateNotFound = errors.New("codex session state not found")

type MemoryCodexStateStore struct {
	mu     sync.Mutex
	states map[string]CodexSessionState
}

func NewMemoryCodexStateStore() *MemoryCodexStateStore {
	return &MemoryCodexStateStore{states: make(map[string]CodexSessionState)}
}
func (s *MemoryCodexStateStore) Load(id string) (CodexSessionState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.states[id]
	if !ok {
		return CodexSessionState{}, ErrCodexStateNotFound
	}
	return cloneCodexState(v), nil
}
func (s *MemoryCodexStateStore) Save(v CodexSessionState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.states[v.SessionID] = cloneCodexState(v)
	return nil
}

// FileCodexStateStore persists provider coordinates and semantic history with
// write-rename durability. The file name is deliberately derived from a safe
// logical-session identifier rather than an upstream thread identifier.
type FileCodexStateStore struct {
	Dir string
	mu  sync.Mutex
}

func (s *FileCodexStateStore) path(id string) (string, error) {
	if id == "" || filepath.Base(id) != id || id == "." || id == ".." {
		return "", errors.New("codex session: unsafe logical session id")
	}
	return filepath.Join(s.Dir, id+".json"), nil
}
func (s *FileCodexStateStore) Load(id string) (CodexSessionState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, err := s.path(id)
	if err != nil {
		return CodexSessionState{}, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return CodexSessionState{}, ErrCodexStateNotFound
	}
	if err != nil {
		return CodexSessionState{}, err
	}
	var v CodexSessionState
	if err := json.Unmarshal(b, &v); err != nil {
		return v, err
	}
	return v, nil
}
func (s *FileCodexStateStore) Save(v CodexSessionState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, err := s.path(v.SessionID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.Dir, 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.Dir, ".codex-session-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err = tmp.Chmod(0o600); err == nil {
		_, err = tmp.Write(b)
	}
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, p)
}

type CodexExecution struct {
	SessionID, ThreadID, Lease string
	Epoch                      uint64
	Mode                       CodexStartMode
}

type codexSubscriber struct {
	after uint64
	ch    chan CodexSemanticEvent
}
type CodexSession struct {
	mu             sync.Mutex
	store          CodexStateStore
	state          CodexSessionState
	adapterVersion string
	lease          string
	deliveries     map[string]CodexAddress
	subscribers    map[uint64]codexSubscriber
	nextSubscriber uint64
}

func OpenCodexSession(store CodexStateStore, sessionID, adapterVersion string) (*CodexSession, error) {
	if store == nil || sessionID == "" || adapterVersion == "" {
		return nil, errors.New("codex session: store, logical session id, and adapter version are required")
	}
	st, err := store.Load(sessionID)
	if errors.Is(err, ErrCodexStateNotFound) {
		st = CodexSessionState{SessionID: sessionID}
	} else if err != nil {
		return nil, err
	}
	c := &CodexSession{store: store, state: st, adapterVersion: adapterVersion, deliveries: make(map[string]CodexAddress), subscribers: make(map[uint64]codexSubscriber)}
	for _, event := range st.Events {
		if event.DeliveryID != "" {
			c.deliveries[event.DeliveryID] = event.Address
		}
	}
	return c, nil
}

func (c *CodexSession) Begin(mode CodexStartMode, lease string) (CodexExecution, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if lease == "" {
		return CodexExecution{}, errors.New("codex session: input lease token is required")
	}
	if c.lease != "" {
		return CodexExecution{}, &CodexRecoveryError{Reason: CodexInputLeaseHeld, Choices: []CodexStartMode{mode}, Detail: "another writer owns the session"}
	}
	thread := c.state.Coordinates.ThreadID
	switch mode {
	case CodexNew:
		if thread != "" {
			return CodexExecution{}, &CodexRecoveryError{Reason: CodexThreadIncompatible, Choices: []CodexStartMode{CodexResume, CodexFork}, Detail: "New cannot replace persisted upstream state"}
		}
	case CodexResume:
		if thread == "" {
			return CodexExecution{}, &CodexRecoveryError{Reason: CodexThreadMissing, Choices: []CodexStartMode{CodexNew, CodexFork}, Detail: "Resume requires a persisted Codex thread"}
		}
		if c.state.Coordinates.AdapterVersion != c.adapterVersion {
			return CodexExecution{}, &CodexRecoveryError{Reason: CodexThreadIncompatible, Choices: []CodexStartMode{CodexNew, CodexFork}, Detail: fmt.Sprintf("persisted adapter %q, current %q", c.state.Coordinates.AdapterVersion, c.adapterVersion)}
		}
	case CodexFork:
		// The old coordinate is returned so the adapter can ask Codex to fork it.
		if thread == "" {
			return CodexExecution{}, &CodexRecoveryError{Reason: CodexThreadMissing, Choices: []CodexStartMode{CodexNew}, Detail: "Fork requires a persisted Codex thread"}
		}
	default:
		return CodexExecution{}, fmt.Errorf("codex session: unknown start mode %q", mode)
	}
	c.state.Epoch++
	c.lease = lease
	if err := c.store.Save(c.state); err != nil {
		c.state.Epoch--
		c.lease = ""
		return CodexExecution{}, err
	}
	return CodexExecution{SessionID: c.state.SessionID, ThreadID: thread, Epoch: c.state.Epoch, Lease: lease, Mode: mode}, nil
}

func (c *CodexSession) RecordThread(epoch uint64, threadID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireEpoch(epoch); err != nil {
		return err
	}
	if threadID == "" {
		return errors.New("codex session: empty thread id")
	}
	if c.state.Coordinates.ThreadID != "" && c.state.Coordinates.ThreadID != threadID {
		return &CodexRecoveryError{Reason: CodexThreadIncompatible, Choices: []CodexStartMode{CodexFork}, Detail: "upstream changed thread identity"}
	}
	prior := c.state.Coordinates
	c.state.Coordinates = CodexCoordinates{ThreadID: threadID, AdapterVersion: c.adapterVersion}
	if err := c.store.Save(c.state); err != nil {
		c.state.Coordinates = prior
		return err
	}
	return nil
}

// RecordFork replaces the provider coordinate only for an explicitly selected
// Fork operation. The fak logical-session identity and cursor remain unchanged.
func (c *CodexSession) RecordFork(epoch uint64, priorThreadID, forkThreadID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireEpoch(epoch); err != nil {
		return err
	}
	if priorThreadID == "" || forkThreadID == "" || c.state.Coordinates.ThreadID != priorThreadID {
		return &CodexRecoveryError{Reason: CodexThreadIncompatible, Choices: []CodexStartMode{CodexResume, CodexNew}, Detail: "fork source does not match persisted Codex thread"}
	}
	prior := c.state.Coordinates
	c.state.Coordinates = CodexCoordinates{ThreadID: forkThreadID, AdapterVersion: c.adapterVersion}
	if err := c.store.Save(c.state); err != nil {
		c.state.Coordinates = prior
		return err
	}
	return nil
}

func (c *CodexSession) Append(epoch uint64, deliveryID string, partial bool, event harnesskit.Envelope) (CodexAddress, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireEpoch(epoch); err != nil {
		return CodexAddress{}, false, err
	}
	if deliveryID == "" {
		return CodexAddress{}, false, errors.New("codex session: delivery id is required")
	}
	if address, ok := c.deliveries[deliveryID]; ok {
		return address, true, nil
	}
	c.state.Cursor++
	address := CodexAddress{SessionID: c.state.SessionID, Cursor: c.state.Cursor, Epoch: epoch}
	item := CodexSemanticEvent{Address: address, DeliveryID: deliveryID, Partial: partial, Event: event}
	c.state.Events = append(c.state.Events, item)
	if err := c.store.Save(c.state); err != nil {
		c.state.Cursor--
		c.state.Events = c.state.Events[:len(c.state.Events)-1]
		return CodexAddress{}, false, err
	}
	c.deliveries[deliveryID] = address
	for _, sub := range c.subscribers {
		if address.Cursor <= sub.after {
			continue
		}
		select {
		case sub.ch <- item:
		default:
			// A slow browser catches up by addressed replay; it must never
			// stall the single writer while the session lock is held.
		}
	}
	return address, false, nil
}

// Attach registers the live tail while holding the same lock used to capture
// replay. Events therefore cannot fall into a replay/live handoff gap.
func (c *CodexSession) Attach(after uint64, buffer int) ([]CodexSemanticEvent, <-chan CodexSemanticEvent, func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if buffer < 1 {
		buffer = 64
	}
	replay := make([]CodexSemanticEvent, 0)
	for _, e := range c.state.Events {
		if e.Address.Cursor > after {
			replay = append(replay, e)
		}
	}
	c.nextSubscriber++
	id := c.nextSubscriber
	ch := make(chan CodexSemanticEvent, buffer)
	c.subscribers[id] = codexSubscriber{after: c.state.Cursor, ch: ch}
	cancel := func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if s, ok := c.subscribers[id]; ok {
			delete(c.subscribers, id)
			close(s.ch)
		}
	}
	return replay, ch, cancel
}
func (c *CodexSession) Release(lease string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.lease == lease {
		c.lease = ""
	}
}
func (c *CodexSession) State() CodexSessionState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return cloneCodexState(c.state)
}
func (c *CodexSession) requireEpoch(epoch uint64) error {
	if epoch != c.state.Epoch {
		return &CodexRecoveryError{Reason: CodexStaleEpoch, Choices: []CodexStartMode{CodexResume}, Detail: fmt.Sprintf("event epoch %d, current %d", epoch, c.state.Epoch)}
	}
	return nil
}
func cloneCodexState(v CodexSessionState) CodexSessionState {
	v.Events = append([]CodexSemanticEvent(nil), v.Events...)
	return v
}
