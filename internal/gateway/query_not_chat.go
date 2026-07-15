package gateway

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

const queryNotChatEnforceEnv = "FAK_QUERY_NOT_CHAT_ENFORCE"

var ErrQueryNotChatViolation = errors.New("gateway: query-not-chat invariant violated")

// ManagedQuerySession separates the stable originating-task pin from the one
// swappable working-state slot. It intentionally contains no transcript slice.
type ManagedQuerySession struct {
	SessionID             string `json:"session_id"`
	PinnedOriginatingTask string `json:"pinned_originating_task"`
	WorkingState          string `json:"working_state,omitempty"`
	Swaps                 uint64 `json:"swaps"`
}

type QueryNotChatVerdict struct {
	Session  ManagedQuerySession `json:"session"`
	Observed bool                `json:"observed"`
	Allowed  bool                `json:"allowed"`
	Reason   string              `json:"reason,omitempty"`
}

type QueryNotChatRegistry struct {
	mu       sync.Mutex
	sessions map[string]ManagedQuerySession
	observed atomic.Uint64
	rejected atomic.Uint64
}

func NewQueryNotChatRegistry() *QueryNotChatRegistry {
	return &QueryNotChatRegistry{sessions: make(map[string]ManagedQuerySession)}
}

// Open pins exactly one content-addressed originating task for the session.
func (r *QueryNotChatRegistry) Open(sessionID string, originatingTask []byte, workingState string) (ManagedQuerySession, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" || len(originatingTask) == 0 {
		return ManagedQuerySession{}, ErrQueryNotChatViolation
	}
	pin := ctxplan.Digest(originatingTask)
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.sessions[sessionID]; ok {
		if existing.PinnedOriginatingTask != pin {
			return existing, ErrQueryNotChatViolation
		}
		return existing, nil
	}
	s := ManagedQuerySession{SessionID: sessionID, PinnedOriginatingTask: pin, WorkingState: workingState}
	r.sessions[sessionID] = s
	return s, nil
}

// Swap replaces volatile state while proving the stable pin is unchanged.
func (r *QueryNotChatRegistry) Swap(sessionID, expectedPin, workingState string) (ManagedQuerySession, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.sessions[sessionID]
	if !ok || s.PinnedOriginatingTask != expectedPin {
		return s, ErrQueryNotChatViolation
	}
	s.WorkingState = workingState
	s.Swaps++
	r.sessions[sessionID] = s
	return s, nil
}

// ObserveAppend detects chat-style accumulation. Default behavior records and
// allows for soak; FAK_QUERY_NOT_CHAT_ENFORCE=true turns the same seam into a gate.
func (r *QueryNotChatRegistry) ObserveAppend(sessionID, expectedPin, appendedState string) (QueryNotChatVerdict, error) {
	r.mu.Lock()
	s, ok := r.sessions[sessionID]
	r.mu.Unlock()
	violates := !ok || s.PinnedOriginatingTask != expectedPin || strings.TrimSpace(appendedState) != ""
	if !violates {
		return QueryNotChatVerdict{Session: s, Allowed: true}, nil
	}
	r.observed.Add(1)
	verdict := QueryNotChatVerdict{Session: s, Observed: true, Allowed: !queryNotChatEnforced(), Reason: "append would accumulate swappable transcript state"}
	if verdict.Allowed {
		return verdict, nil
	}
	r.rejected.Add(1)
	return verdict, ErrQueryNotChatViolation
}

func queryNotChatEnforced() bool {
	on, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(queryNotChatEnforceEnv)))
	return err == nil && on
}

func (r *QueryNotChatRegistry) Metrics() (observed, rejected uint64) {
	return r.observed.Load(), r.rejected.Load()
}
