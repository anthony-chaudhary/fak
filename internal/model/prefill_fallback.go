package model

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"sync"
	"sync/atomic"
)

// ErrPrefillReplayFallback is the sentinel under a device-only refusal to serve a
// prompt through the scalar token-replay route after native whole-sequence
// prefill declined. Ordinary serving may take that route; a Halo appliance
// profile may not, because the decline is exactly the 17-22 vs 421 tok/s prefill
// gap it exists to prevent.
var ErrPrefillReplayFallback = errors.New("model: device-only execution refuses scalar token-replay prefill")

// PrefillFallbackOtherReason absorbs reasons past the label cap or outside the
// label grammar, so a caller can never mint unbounded series.
const PrefillFallbackOtherReason = "other"

const maxPrefillFallbackReasons = 32

// prefillFallbackReasonName admits the hyphenated decline-reason vocabulary
// verbatim, so a counter label joins to the receipt's decline_reason string.
var prefillFallbackReasonName = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}$`)

var prefillFallbacks struct {
	mu       sync.RWMutex
	byReason map[string]*atomic.Uint64
}

// PrefillFallbackCount is one decline reason's cumulative count.
type PrefillFallbackCount struct {
	Reason string
	Count  uint64
}

// DeclarePrefillFallbackReason exposes a wired reason at 0 before it first fires,
// so "wired and never declined" reads differently from "not instrumented".
func DeclarePrefillFallbackReason(reason string) { prefillFallbackCounter(reason) }

// CountPrefillFallback records one whole-sequence prefill that declined to the
// scalar token-replay route, labelled by its stable decline reason.
func CountPrefillFallback(reason string) { prefillFallbackCounter(reason).Add(1) }

func prefillFallbackCounter(reason string) *atomic.Uint64 {
	if !prefillFallbackReasonName.MatchString(reason) {
		reason = PrefillFallbackOtherReason
	}
	prefillFallbacks.mu.RLock()
	c := prefillFallbacks.byReason[reason]
	prefillFallbacks.mu.RUnlock()
	if c != nil {
		return c
	}
	prefillFallbacks.mu.Lock()
	defer prefillFallbacks.mu.Unlock()
	if prefillFallbacks.byReason == nil {
		prefillFallbacks.byReason = make(map[string]*atomic.Uint64)
	}
	if c = prefillFallbacks.byReason[reason]; c != nil {
		return c
	}
	if len(prefillFallbacks.byReason) >= maxPrefillFallbackReasons {
		reason = PrefillFallbackOtherReason
		if c = prefillFallbacks.byReason[reason]; c != nil {
			return c
		}
	}
	c = new(atomic.Uint64)
	prefillFallbacks.byReason[reason] = c
	return c
}

// observePrefillFallback is the counting seam for one declined whole-sequence
// prefill. It is a package-level var so a test can observe the reason without a
// Session-scoped field; the default records the process-wide labelled counter.
var observePrefillFallback = func(reason string) { CountPrefillFallback(reason) }

// notePrefillFallback records one whole-sequence prefill that declined to the
// scalar token-replay route and refuses it on a device-only session that has no
// admitted bypass. Ordinary serving keeps the fallback; a Halo appliance profile
// must not, because the decline is the prefill-throughput gap it exists to catch.
func (s *Session) notePrefillFallback(status Qwen35SequencePrefillRouteStatus) {
	observePrefillFallback(status.DeclineReason)
	if s == nil || s.executionPolicy != ExecutionPolicyDeviceOnly || s.hostPathBypass != "" {
		return
	}
	s.refuseHostFallbackCause(
		"prefill "+Qwen35SequencePrefillFallbackPath,
		fmt.Errorf("%w: %s", ErrPrefillReplayFallback, status.DeclineReason),
	)
}

// PrefillFallbackCounts returns every declared or fired reason, sorted by reason.
func PrefillFallbackCounts() []PrefillFallbackCount {
	prefillFallbacks.mu.RLock()
	out := make([]PrefillFallbackCount, 0, len(prefillFallbacks.byReason))
	for reason, c := range prefillFallbacks.byReason {
		out = append(out, PrefillFallbackCount{Reason: reason, Count: c.Load()})
	}
	prefillFallbacks.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Reason < out[j].Reason })
	return out
}
