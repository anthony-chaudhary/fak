package servechain

import (
	"sync"
	"time"

	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

// rateWindow is the chain's fixed one-minute rate window over the manifest's
// neutral Limits. Requests and committed tokens are counted per wallclock
// UTC minute; a saturated window refuses new requests until the minute rolls.
//
// Commit discipline: the request count folds in only after a request has
// passed every admission stage (so chain-refused requests never consume
// rate), and tokens fold in at usage capture. A request that crosses the
// token cap still runs — its tokens are unknown at admission time — and the
// next request in that minute is refused.
type rateWindow struct {
	mu     sync.Mutex
	rpm    int   // requests per minute; 0 = uncapped
	tpm    int   // tokens per minute; 0 = uncapped
	minute int64 // unix minute index of the current bucket
	reqs   int
	tokens int64
}

func newRateWindow(l billingmanifest.Limits) *rateWindow {
	return &rateWindow{rpm: l.RequestsPerMinute, tpm: l.TokensPerMinute}
}

// saturated reports whether one more request may enter. It rolls the bucket
// if the minute has moved on.
func (w *rateWindow) saturated(now time.Time) bool {
	if w == nil || (w.rpm <= 0 && w.tpm <= 0) {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.rollLocked(now)
	if w.rpm > 0 && w.reqs >= w.rpm {
		return true
	}
	if w.tpm > 0 && w.tokens >= int64(w.tpm) {
		return true
	}
	return false
}

// commitRequest folds one fully admitted request into the window.
func (w *rateWindow) commitRequest(now time.Time) {
	if w == nil || w.rpm <= 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.rollLocked(now)
	w.reqs++
}

// commitTokens folds captured usage into the window.
func (w *rateWindow) commitTokens(tokens int64, now time.Time) {
	if w == nil || w.tpm <= 0 || tokens <= 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.rollLocked(now)
	w.tokens += tokens
}

// rollLocked resets the counters when the wallclock minute has advanced.
// Callers hold mu.
func (w *rateWindow) rollLocked(now time.Time) {
	minute := now.Unix() / 60
	if minute != w.minute {
		w.minute = minute
		w.reqs = 0
		w.tokens = 0
	}
}
