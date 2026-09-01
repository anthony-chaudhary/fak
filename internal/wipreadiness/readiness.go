// Package wipreadiness models a reusable observation receipt for fresh-work admission.
package wipreadiness

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"time"
)

const Schema = "fak-wip-readiness/1"

type Verdict string

const (
	VerdictCurrent Verdict = "current"
	VerdictBlocked Verdict = "blocked"
)

type ReasonCode string

const (
	ReasonMissing                  ReasonCode = "missing"
	ReasonStale                    ReasonCode = "stale"
	ReasonPartialHost              ReasonCode = "partial-host"
	ReasonUnknownSchema            ReasonCode = "unknown-schema"
	ReasonUnavailable              ReasonCode = "unavailable"
	ReasonDiagnosticallyIncomplete ReasonCode = "diagnostically-incomplete"
)

type Diagnostic struct {
	Source  string `json:"source,omitempty"`
	Code    string `json:"code"`
	Message string `json:"message,omitempty"`
}

type Summary struct {
	Total       int `json:"total"`
	Dirty       int `json:"dirty,omitempty"`
	RemoteOwned int `json:"remote_owned,omitempty"`
}

type Source struct {
	Name           string       `json:"name"`
	Schema         string       `json:"schema,omitempty"`
	ExpectedSchema string       `json:"expected_schema,omitempty"`
	ObservedAt     time.Time    `json:"observed_at,omitempty"`
	Available      bool         `json:"available"`
	Complete       bool         `json:"complete"`
	Summary        Summary      `json:"summary"`
	Diagnostics    []Diagnostic `json:"diagnostics,omitempty"`
}

type HostCoverage struct {
	Expected []string `json:"expected,omitempty"`
	Observed []string `json:"observed,omitempty"`
}

type Ownership string

const (
	OwnershipLocal  Ownership = "local"
	OwnershipRemote Ownership = "remote"
)

// Work is evidence only. It deliberately carries no cleanup or reclamation action.
type Work struct {
	ID        string    `json:"id"`
	Dirty     bool      `json:"dirty,omitempty"`
	Ownership Ownership `json:"ownership"`
	Host      string    `json:"host,omitempty"`
}

type Observation struct {
	ObservedAt  time.Time    `json:"observed_at"`
	Queue       Source       `json:"queue"`
	Inventory   Source       `json:"inventory"`
	Lifecycle   Source       `json:"lifecycle"`
	Capacity    Source       `json:"capacity"`
	Hosts       HostCoverage `json:"hosts"`
	Diagnostics []Diagnostic `json:"diagnostics,omitempty"`
	Work        []Work       `json:"work,omitempty"`
}

type Scanner interface {
	Scan(context.Context) (Observation, error)
}

type FreshnessPolicy struct {
	MaxAge time.Duration `json:"max_age"`
}

type Receipt struct {
	Schema       string          `json:"schema"`
	ObservedAt   time.Time       `json:"observed_at"`
	ExpiresAt    time.Time       `json:"expires_at"`
	Freshness    FreshnessPolicy `json:"freshness"`
	Verdict      Verdict         `json:"verdict"`
	Reasons      []ReasonCode    `json:"reasons,omitempty"`
	Queue        Source          `json:"queue"`
	Inventory    Source          `json:"inventory"`
	Lifecycle    Source          `json:"lifecycle"`
	Capacity     Source          `json:"capacity"`
	Hosts        HostCoverage    `json:"hosts"`
	Diagnostics  []Diagnostic    `json:"diagnostics,omitempty"`
	Work         []Work          `json:"work,omitempty"`
	EvidenceOnly bool            `json:"evidence_only"`
}

func Build(observation Observation, observedAt time.Time, maxAge time.Duration) Receipt {
	if observedAt.IsZero() {
		observedAt = observation.ObservedAt
	}
	r := Receipt{
		Schema: Schema, ObservedAt: observation.ObservedAt, ExpiresAt: observation.ObservedAt.Add(maxAge),
		Freshness: FreshnessPolicy{MaxAge: maxAge}, Queue: observation.Queue, Inventory: observation.Inventory,
		Lifecycle: observation.Lifecycle, Capacity: observation.Capacity, Hosts: observation.Hosts,
		Diagnostics: append([]Diagnostic(nil), observation.Diagnostics...), Work: append([]Work(nil), observation.Work...),
		EvidenceOnly: true,
	}
	var reasons []ReasonCode
	for _, source := range []*Source{&r.Queue, &r.Inventory, &r.Lifecycle, &r.Capacity} {
		if !source.Available {
			reasons = appendReason(reasons, ReasonUnavailable)
		}
		if source.ExpectedSchema != "" && source.Schema != source.ExpectedSchema {
			reasons = appendReason(reasons, ReasonUnknownSchema)
		}
		if !source.Complete || len(source.Diagnostics) != 0 {
			reasons = appendReason(reasons, ReasonDiagnosticallyIncomplete)
		}
		if !source.ObservedAt.IsZero() && maxAge >= 0 && observedAt.Sub(source.ObservedAt) > maxAge {
			reasons = appendReason(reasons, ReasonStale)
		}
		r.Diagnostics = append(r.Diagnostics, source.Diagnostics...)
	}
	if observation.ObservedAt.IsZero() || maxAge < 0 || observedAt.Sub(observation.ObservedAt) > maxAge {
		reasons = appendReason(reasons, ReasonStale)
	}
	if missingHosts(r.Hosts) {
		reasons = appendReason(reasons, ReasonPartialHost)
	}
	if len(r.Diagnostics) != 0 {
		reasons = appendReason(reasons, ReasonDiagnosticallyIncomplete)
	}
	r.Reasons = reasons
	if len(reasons) == 0 {
		r.Verdict = VerdictCurrent
	} else {
		r.Verdict = VerdictBlocked
	}
	return r
}

func appendReason(reasons []ReasonCode, reason ReasonCode) []ReasonCode {
	for _, existing := range reasons {
		if existing == reason {
			return reasons
		}
	}
	return append(reasons, reason)
}

func missingHosts(coverage HostCoverage) bool {
	observed := make(map[string]struct{}, len(coverage.Observed))
	for _, host := range coverage.Observed {
		observed[host] = struct{}{}
	}
	for _, host := range coverage.Expected {
		if _, ok := observed[host]; !ok {
			return true
		}
	}
	return false
}

// Cache reuses one scan until its receipt expires.
type Cache struct {
	mu      sync.Mutex
	scanner Scanner
	maxAge  time.Duration
	now     func() time.Time
	receipt *Receipt
}

func NewCache(scanner Scanner, maxAge time.Duration) *Cache {
	return NewCacheWithClock(scanner, maxAge, time.Now)
}

func NewCacheWithClock(scanner Scanner, maxAge time.Duration, now func() time.Time) *Cache {
	return &Cache{scanner: scanner, maxAge: maxAge, now: now}
}

func (c *Cache) Receipt(ctx context.Context) Receipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	if c.receipt != nil && now.Before(c.receipt.ExpiresAt) {
		return *c.receipt
	}
	if c.scanner == nil {
		r := unavailableReceipt(now, c.maxAge, errors.New("scanner is nil"))
		c.receipt = &r
		return r
	}
	observation, err := c.scanner.Scan(ctx)
	if err != nil {
		r := unavailableReceipt(now, c.maxAge, err)
		c.receipt = &r
		return r
	}
	r := Build(observation, now, c.maxAge)
	c.receipt = &r
	return r
}

func unavailableReceipt(now time.Time, maxAge time.Duration, err error) Receipt {
	diagnostic := Diagnostic{Code: string(ReasonUnavailable), Message: err.Error()}
	return Receipt{Schema: Schema, ObservedAt: now, ExpiresAt: now.Add(maxAge), Freshness: FreshnessPolicy{MaxAge: maxAge}, Verdict: VerdictBlocked, Reasons: []ReasonCode{ReasonUnavailable}, Diagnostics: []Diagnostic{diagnostic}, EvidenceOnly: true}
}

type Intent string

const (
	IntentFreshStart               Intent = "fresh-start"
	IntentRecovery                 Intent = "recovery"
	IntentLanding                  Intent = "landing"
	IntentSafety                   Intent = "safety"
	IntentParking                  Intent = "parking"
	IntentAlreadyOwnedContinuation Intent = "already-owned-continuation"
)

type AdmissionRequest struct {
	Intent         Intent `json:"intent"`
	OverrideReason string `json:"override_reason,omitempty"`
}

type Admission struct {
	Admitted       bool         `json:"admitted"`
	Exempt         bool         `json:"exempt,omitempty"`
	Overridden     bool         `json:"overridden,omitempty"`
	OverrideReason string       `json:"override_reason,omitempty"`
	Reasons        []ReasonCode `json:"reasons,omitempty"`
}

func Admit(receipt *Receipt, request AdmissionRequest) Admission {
	if exempt(request.Intent) {
		return Admission{Admitted: true, Exempt: true}
	}
	if receipt != nil && receipt.Schema == Schema && receipt.Verdict == VerdictCurrent && len(receipt.Reasons) == 0 {
		return Admission{Admitted: true}
	}
	reasons := []ReasonCode{ReasonMissing}
	if receipt != nil {
		reasons = append([]ReasonCode(nil), receipt.Reasons...)
		if receipt.Schema != Schema {
			reasons = appendReason(reasons, ReasonUnknownSchema)
		}
		if len(reasons) == 0 {
			reasons = append(reasons, ReasonUnavailable)
		}
	}
	if reason := strings.TrimSpace(request.OverrideReason); reason != "" {
		return Admission{Admitted: true, Overridden: true, OverrideReason: reason, Reasons: reasons}
	}
	return Admission{Reasons: reasons}
}

func exempt(intent Intent) bool {
	switch intent {
	case IntentRecovery, IntentLanding, IntentSafety, IntentParking, IntentAlreadyOwnedContinuation:
		return true
	default:
		return false
	}
}

// PreservedWork returns dirty or remotely owned rows as evidence. The returned
// values confer no cleanup authority.
func (r Receipt) PreservedWork() []Work {
	var preserved []Work
	for _, work := range r.Work {
		if work.Dirty || work.Ownership == OwnershipRemote {
			preserved = append(preserved, work)
		}
	}
	sort.SliceStable(preserved, func(i, j int) bool { return preserved[i].ID < preserved[j].ID })
	return preserved
}
