package ctxmmu

import (
	"errors"
	"fmt"
	"sync"
)

// TokenCounts keeps economically distinct token classes separate throughout a request.
type TokenCounts struct {
	UncachedInput int `json:"uncached_input"`
	CachedInput   int `json:"cached_input"`
	Output        int `json:"output"`
}

func (c TokenCounts) valid() bool { return c.UncachedInput >= 0 && c.CachedInput >= 0 && c.Output >= 0 }
func (c TokenCounts) add(o TokenCounts) TokenCounts {
	return TokenCounts{c.UncachedInput + o.UncachedInput, c.CachedInput + o.CachedInput, c.Output + o.Output}
}
func (c TokenCounts) subFloor(o TokenCounts) TokenCounts {
	return TokenCounts{max(0, c.UncachedInput-o.UncachedInput), max(0, c.CachedInput-o.CachedInput), max(0, c.Output-o.Output)}
}

// TokenUsage uses pointers because absent provider detail is unknown, not zero.
type TokenUsage struct {
	UncachedInput *int `json:"uncached_input,omitempty"`
	CachedInput   *int `json:"cached_input,omitempty"`
	Output        *int `json:"output,omitempty"`
}

// TokenErrors is the signed observed-minus-forecast error for known classes.
type TokenErrors struct {
	UncachedInput *int `json:"uncached_input,omitempty"`
	CachedInput   *int `json:"cached_input,omitempty"`
	Output        *int `json:"output,omitempty"`
}

// ReconciledTokens is the durable accounting result keyed by the admission request.
type ReconciledTokens struct {
	RequestID                string      `json:"request_id"`
	ProfileID                string      `json:"profile_id"`
	Status                   string      `json:"status"`
	Attempts                 int         `json:"attempts"`
	Forecast                 TokenCounts `json:"forecast"`
	Reserved                 TokenCounts `json:"reserved"`
	Observed                 TokenUsage  `json:"observed"`
	Released                 TokenCounts `json:"released"`
	ForecastError            TokenErrors `json:"forecast_error"`
	ReuseExpectationAccuracy *float64    `json:"reuse_expectation_accuracy,omitempty"`
}

type tokenEntry struct {
	record ReconciledTokens
	known  [3]bool
	obs    TokenCounts
	done   bool
}

// TokenReconciler carries a request/profile identity from admission through execution
// and provider completion while preserving unknown usage fields.
type TokenReconciler struct {
	mu      sync.Mutex
	entries map[string]*tokenEntry
}

func NewTokenReconciler() *TokenReconciler {
	return &TokenReconciler{entries: make(map[string]*tokenEntry)}
}

// Admit records the guard forecast and reserves that capacity before execution.
func (r *TokenReconciler) Admit(requestID, profileID string, forecast TokenCounts) error {
	if requestID == "" || profileID == "" {
		return errors.New("request and profile IDs are required")
	}
	if !forecast.valid() {
		return errors.New("token counts cannot be negative")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.entries[requestID]; exists {
		return fmt.Errorf("request %q already admitted", requestID)
	}
	r.entries[requestID] = &tokenEntry{record: ReconciledTokens{RequestID: requestID, ProfileID: profileID, Status: "admitted", Forecast: forecast, Reserved: forecast}}
	return nil
}

// BeginAttempt marks server execution. Retries retain the original request/profile ID.
func (r *TokenReconciler) BeginAttempt(requestID, profileID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, err := r.entry(requestID, profileID)
	if err != nil {
		return err
	}
	if e.done {
		return errors.New("request already completed")
	}
	e.record.Attempts++
	e.record.Status = "executing"
	return nil
}

// Observe adds one provider usage update. It supports final and streaming updates;
// nil classes remain unknown rather than becoming zero.
func (r *TokenReconciler) Observe(requestID, profileID string, usage TokenUsage) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, err := r.entry(requestID, profileID)
	if err != nil {
		return err
	}
	if e.done {
		return errors.New("request already completed")
	}
	vals := []*int{usage.UncachedInput, usage.CachedInput, usage.Output}
	for _, v := range vals {
		if v != nil && *v < 0 {
			return errors.New("observed token counts cannot be negative")
		}
	}
	if usage.UncachedInput != nil {
		e.obs.UncachedInput += *usage.UncachedInput
		e.known[0] = true
	}
	if usage.CachedInput != nil {
		e.obs.CachedInput += *usage.CachedInput
		e.known[1] = true
	}
	if usage.Output != nil {
		e.obs.Output += *usage.Output
		e.known[2] = true
	}
	return nil
}

// Complete releases unused reservations for success, cancellation, retry exhaustion,
// or provider error. Known observed usage is conserved; unknown detail is never imputed.
func (r *TokenReconciler) Complete(requestID, profileID, status string) (ReconciledTokens, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, err := r.entry(requestID, profileID)
	if err != nil {
		return ReconciledTokens{}, err
	}
	if e.done {
		return ReconciledTokens{}, errors.New("request already completed")
	}
	switch status {
	case "success", "cancelled", "provider_error", "retry_exhausted":
	default:
		return ReconciledTokens{}, fmt.Errorf("invalid completion status %q", status)
	}
	e.done = true
	e.record.Status = status
	e.record.Released = e.record.Reserved.subFloor(e.obs)
	setUsageAndErrors(e)
	return e.record, nil
}

func (r *TokenReconciler) Read(requestID string) (ReconciledTokens, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.entries[requestID]
	if !ok || !e.done {
		return ReconciledTokens{}, false
	}
	return e.record, true
}

func (r *TokenReconciler) entry(requestID, profileID string) (*tokenEntry, error) {
	e, ok := r.entries[requestID]
	if !ok {
		return nil, fmt.Errorf("request %q was not admitted", requestID)
	}
	if e.record.ProfileID != profileID {
		return nil, fmt.Errorf("profile ID changed for request %q", requestID)
	}
	return e, nil
}

func setUsageAndErrors(e *tokenEntry) {
	obs := [3]int{e.obs.UncachedInput, e.obs.CachedInput, e.obs.Output}
	forecast := [3]int{e.record.Forecast.UncachedInput, e.record.Forecast.CachedInput, e.record.Forecast.Output}
	usage := []*(*int){&e.record.Observed.UncachedInput, &e.record.Observed.CachedInput, &e.record.Observed.Output}
	errs := []*(*int){&e.record.ForecastError.UncachedInput, &e.record.ForecastError.CachedInput, &e.record.ForecastError.Output}
	for i := range obs {
		if e.known[i] {
			v, d := obs[i], obs[i]-forecast[i]
			*usage[i] = &v
			*errs[i] = &d
		}
	}
	if e.known[0] && e.known[1] {
		forecastInput := forecast[0] + forecast[1]
		observedInput := obs[0] + obs[1]
		if forecastInput > 0 && observedInput > 0 {
			want, got := float64(forecast[1])/float64(forecastInput), float64(obs[1])/float64(observedInput)
			accuracy := 1 - abs(want-got)
			e.record.ReuseExpectationAccuracy = &accuracy
		}
	}
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
