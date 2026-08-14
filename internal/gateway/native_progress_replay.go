package gateway

import (
	"errors"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const nativeProgressReplayLimit = 256

var errNativeProgressCursorTooOld = errors.New("native progress cursor is older than the retained replay window")

type nativeProgressReplay struct {
	mu       sync.Mutex
	sessions map[string][]agent.ProgressEvent
}

func (r *nativeProgressReplay) append(session string, ev agent.ProgressEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sessions == nil {
		r.sessions = map[string][]agent.ProgressEvent{}
	}
	events := append(r.sessions[session], ev)
	if len(events) > nativeProgressReplayLimit {
		events = append([]agent.ProgressEvent(nil), events[len(events)-nativeProgressReplayLimit:]...)
	}
	r.sessions[session] = events
}
func (r *nativeProgressReplay) after(session string, cursor uint64) ([]agent.ProgressEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	events := r.sessions[session]
	if len(events) == 0 {
		return nil, nil
	}
	if cursor+1 < events[0].Seq {
		return nil, errNativeProgressCursorTooOld
	}
	out := make([]agent.ProgressEvent, 0, len(events))
	for _, ev := range events {
		if ev.Seq > cursor {
			out = append(out, ev)
		}
	}
	return out, nil
}
func nativeProgressCursor(raw string) (uint64, error) {
	if raw == "" {
		return 0, nil
	}
	return strconv.ParseUint(raw, 10, 64)
}

// Native progress delivery is at-least-once. Clients deduplicate on (session, seq).
// A sequence jump is an observable drop; an expired bounded cursor is refused rather
// than silently returning a partial tail.
