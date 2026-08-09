package microagent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// APIProviderShape declares only controls and evidence exposed by an API-only provider.
type APIProviderShape struct {
	Name                 string
	ReuseControl         string
	ReuseEvidence        string
	RequestsPerMinute    int
	TokensPerMinute      int
	Concurrency          int
	MaxSpendMicros       int64
	PromptMicrosPerToken int64
	OutputMicrosPerToken int64
}

// APIAdmission is a process-local API-only admission controller. It bounds concurrency,
// sliding-window RPM/TPM, and estimated spend without inferring provider-side KV state.
type APIAdmission struct {
	shape APIProviderShape
	now   func() time.Time

	mu          sync.Mutex
	requests    []apiUsage
	reserved    int
	spendMicros int64
	inFlight    int
	slots       chan struct{}
}

type apiUsage struct {
	at     time.Time
	tokens int
}

// APILease accounts for one admitted provider request.
type APILease struct {
	owner      *APIAdmission
	reserved   int
	costMicros int64
	once       sync.Once
}

func NewAPIAdmission(shape APIProviderShape) (*APIAdmission, error) {
	if strings.TrimSpace(shape.Name) == "" || shape.Concurrency <= 0 || shape.RequestsPerMinute <= 0 || shape.TokensPerMinute <= 0 {
		return nil, errors.New("microagent: API provider name, concurrency, RPM, and TPM must be positive")
	}
	if shape.ReuseControl == "" {
		shape.ReuseControl = "byte-identical-prefix"
	}
	if shape.ReuseEvidence == "" {
		shape.ReuseEvidence = "opaque"
	}
	return &APIAdmission{shape: shape, now: time.Now, slots: make(chan struct{}, shape.Concurrency)}, nil
}

// Acquire reserves one request and its estimated token/spend envelope. It fails closed
// locally instead of knowingly sending an over-limit request to the provider.
func (a *APIAdmission) Acquire(ctx context.Context, estimatedPromptTokens, maxOutputTokens int) (*APILease, error) {
	if estimatedPromptTokens < 0 || maxOutputTokens < 0 {
		return nil, errors.New("microagent: estimated token counts must be nonnegative")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case a.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	admitted := false
	defer func() {
		if !admitted {
			<-a.slots
		}
	}()
	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.now()
	cutoff := now.Add(-time.Minute)
	kept := a.requests[:0]
	windowTokens := 0
	for _, u := range a.requests {
		if u.at.After(cutoff) {
			kept = append(kept, u)
			windowTokens += u.tokens
		}
	}
	a.requests = kept
	if len(a.requests) >= a.shape.RequestsPerMinute {
		return nil, fmt.Errorf("microagent: provider RPM envelope exhausted (%d)", a.shape.RequestsPerMinute)
	}
	tokens := estimatedPromptTokens + maxOutputTokens
	if windowTokens+tokens > a.shape.TokensPerMinute {
		return nil, fmt.Errorf("microagent: provider TPM envelope would be exceeded: used=%d requested=%d limit=%d", windowTokens, tokens, a.shape.TokensPerMinute)
	}
	cost := int64(estimatedPromptTokens)*a.shape.PromptMicrosPerToken + int64(maxOutputTokens)*a.shape.OutputMicrosPerToken
	if a.shape.MaxSpendMicros > 0 && a.spendMicros+cost > a.shape.MaxSpendMicros {
		return nil, fmt.Errorf("microagent: provider spend envelope would be exceeded: used=%d requested=%d limit=%d micro-units", a.spendMicros, cost, a.shape.MaxSpendMicros)
	}
	a.requests = append(a.requests, apiUsage{at: now, tokens: tokens})
	a.reserved += tokens
	a.spendMicros += cost
	a.inFlight++
	admitted = true
	return &APILease{owner: a, reserved: tokens, costMicros: cost}, nil
}

// Reconcile replaces the conservative reservation with provider-observed usage.
// Failed requests stay fully charged when this method is not called.
func (l *APILease) Reconcile(promptTokens, completionTokens int) {
	if l == nil || l.owner == nil || promptTokens < 0 || completionTokens < 0 {
		return
	}
	actualTokens := promptTokens + completionTokens
	actualCost := int64(promptTokens)*l.owner.shape.PromptMicrosPerToken + int64(completionTokens)*l.owner.shape.OutputMicrosPerToken
	l.owner.mu.Lock()
	if delta := actualTokens - l.reserved; delta != 0 && len(l.owner.requests) > 0 {
		l.owner.requests[len(l.owner.requests)-1].tokens += delta
		l.owner.reserved += delta
		l.reserved = actualTokens
	}
	l.owner.spendMicros += actualCost - l.costMicros
	l.costMicros = actualCost
	l.owner.mu.Unlock()
}

// Release returns only the concurrency slot. Rate and spend reservations remain charged,
// because an attempted provider request consumes those envelopes even when it fails.
func (l *APILease) Release() {
	if l == nil || l.owner == nil {
		return
	}
	l.once.Do(func() {
		l.owner.mu.Lock()
		l.owner.inFlight--
		l.owner.mu.Unlock()
		<-l.owner.slots
	})
}

// RetryAfter parses either delta-seconds or an HTTP-date. Invalid/missing values return zero.
func RetryAfter(h http.Header, now time.Time) time.Duration {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(raw); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	when, err := http.ParseTime(raw)
	if err != nil || !when.After(now) {
		return 0
	}
	return when.Sub(now)
}

// ReuseClaim reports the strongest claim permitted by the provider shape and observed bill.
func (a *APIAdmission) ReuseClaim(cachedPromptTokens int64) string {
	if cachedPromptTokens > 0 && a.shape.ReuseEvidence != "opaque" {
		return "provider-billed-cache-hit"
	}
	if a.shape.ReuseEvidence == "opaque" {
		return "not-observable"
	}
	return "provider-reported-zero-cache"
}

func (a *APIAdmission) Shape() APIProviderShape { return a.shape }
