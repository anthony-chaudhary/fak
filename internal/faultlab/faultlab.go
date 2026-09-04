// Package faultlab provides a fault injection laboratory for agentic serving,
// network stream disruptions, JSON corruption, mid-turn truncation, and simulated kernel faults.
//
// The package is pure Go and stdlib-only, designed to test resilience against real-world
// failures observed in LLM agent pipelines: partial network reads, ill-formed tool-call JSON,
// simulated OOM / memory pressure, catastrophic kernel resets, and latency tail spikes.
package faultlab

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// FaultType defines the classification of simulated failures.
// Invariant: A FaultType must be one of the registered closed-vocabulary constants.
type FaultType string

const (
	// Truncation truncates payload data mid-turn or cuts streams short before EOF.
	Truncation FaultType = "truncation"

	// CorruptedJSON injects structural syntax errors into JSON payloads.
	CorruptedJSON FaultType = "corrupted_json"

	// LatencySpike introduces an artificial delay or timeout spike.
	LatencySpike FaultType = "latency_spike"

	// MemoryPressure simulates out-of-memory or high resource exhaustion.
	MemoryPressure FaultType = "memory_pressure"

	// NetworkDrop simulates an abrupt disconnection or dropped socket.
	NetworkDrop FaultType = "network_drop"

	// HostReset simulates an unrecoverable host panic, crash, or reset.
	HostReset FaultType = "host_reset"
)

// IsValid reports whether the FaultType is a known member of the closed vocabulary.
// Invariant: Returns true strictly for registered fault types; unrecognized types return false.
func (t FaultType) IsValid() bool {
	switch t {
	case Truncation, CorruptedJSON, LatencySpike, MemoryPressure, NetworkDrop, HostReset:
		return true
	default:
		return false
	}
}

// String returns the string representation of the fault type.
func (t FaultType) String() string {
	return string(t)
}

// Sentinel errors returned or recorded by faultlab.
// Invariant: Sentinel errors are immutable and comparable using errors.Is.
var (
	// ErrFaultInjected indicates an unclassified or default simulated fault was injected.
	ErrFaultInjected = errors.New("faultlab: fault injected")
	// ErrTruncated indicates payload data was cut short before normal completion.
	ErrTruncated = errors.New("faultlab: payload truncated mid-stream")
	// ErrCorruptedJSON indicates malformed bytes were injected into a JSON payload.
	ErrCorruptedJSON = errors.New("faultlab: corrupted JSON payload")
	// ErrLatencySpike indicates artificial latency exceeded the operational deadline.
	ErrLatencySpike = errors.New("faultlab: latency spike exceeded deadline")
	// ErrMemoryPressure indicates simulated out-of-memory or resource exhaustion.
	ErrMemoryPressure = errors.New("faultlab: simulated memory pressure / OOM")
	// ErrNetworkDrop indicates an abrupt connection drop or socket disconnect.
	ErrNetworkDrop = errors.New("faultlab: simulated network drop / disconnect")
	// ErrHostReset indicates an unrecoverable host panic, crash, or reset.
	ErrHostReset = errors.New("faultlab: simulated host panic or reset")
	// ErrInvalidRule indicates a fault rule configuration was malformed.
	ErrInvalidRule = errors.New("faultlab: invalid fault rule")
	// ErrRuleNotFound indicates the requested rule ID was not found in the registry.
	ErrRuleNotFound = errors.New("faultlab: rule not found")
	// ErrScenarioNotFound indicates the requested scenario ID was not found in the registry.
	ErrScenarioNotFound = errors.New("faultlab: scenario not found")
)

// FaultRule configures the conditions, target, and characteristics of a fault to inject.
// Invariant: Target pattern matching defaults to wildcard "*" if empty.
// Guard: Probability must reside in the closed interval [0.0, 1.0].
type FaultRule struct {
	ID            string            `json:"id"`
	Target        string            `json:"target"`                   // Exact name, wildcard "*", or glob pattern (e.g. "tool:*", "stream/*")
	Type          FaultType         `json:"type"`                     // FaultType to trigger
	Probability   float64           `json:"probability"`              // 0.0 to 1.0 (default: 1.0)
	Delay         time.Duration     `json:"delay"`                    // Latency duration or injection wait
	TruncateRatio float64           `json:"truncate_ratio,omitempty"` // Fractional payload to preserve (0.0 < r < 1.0)
	TruncateBytes int               `json:"truncate_bytes,omitempty"` // Explicit byte cutoff if > 0
	MaxHits       int               `json:"max_hits,omitempty"`       // Max triggers before deactivating (0 = unlimited)
	Active        bool              `json:"active"`                   // Whether the rule is enabled
	CustomPayload []byte            `json:"custom_payload,omitempty"` // Optional custom override payload
	Metadata      map[string]string `json:"metadata,omitempty"`       // Extensible metadata
}

// NewFaultRule returns an enabled FaultRule with default 100% probability.
// Invariant: The constructed rule has Probability initialized to 1.0 and Active set to true.
func NewFaultRule(id string, fType FaultType, target string) FaultRule {
	return FaultRule{
		ID:          id,
		Type:        fType,
		Target:      target,
		Probability: 1.0,
		Active:      true,
	}
}

// FaultScenario represents a named group of rules simulating a coordinated failure environment.
// Invariant: When enabled, all constituent rules are registered and activated.
type FaultScenario struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Rules       []FaultRule       `json:"rules"`
	Enabled     bool              `json:"enabled"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// FaultHit records one incident of a fault being injected.
type FaultHit struct {
	RuleID    string        `json:"rule_id"`
	Type      FaultType     `json:"type"`
	Target    string        `json:"target"`
	Timestamp time.Time     `json:"timestamp"`
	Latency   time.Duration `json:"latency,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// FaultReport contains aggregated statistics and recent history of injected faults.
type FaultReport struct {
	TotalInjections int64               `json:"total_injections"`
	HitsByType      map[FaultType]int64 `json:"hits_by_type"`
	HitsByTarget    map[string]int64    `json:"hits_by_target"`
	HitsByRule      map[string]int64    `json:"hits_by_rule"`
	RecentHits      []FaultHit          `json:"recent_hits"`
	ActiveRules     int                 `json:"active_rules"`
	ActiveScenarios int                 `json:"active_scenarios"`
}

// Option configures a FaultInjector instance.
type Option func(*FaultInjector)

// WithSeed sets a deterministic random seed for probabilistic fault evaluation.
// Invariant: Injector RNG produces reproducible results for identical seeds.
func WithSeed(seed int64) Option {
	return func(fi *FaultInjector) {
		fi.rng = rand.New(rand.NewSource(seed))
	}
}

// WithMaxRecentHits sets the maximum number of recent hits kept in memory.
// Guard: Values <= 0 are ignored to maintain a positive hit history capacity.
func WithMaxRecentHits(max int) Option {
	return func(fi *FaultInjector) {
		if max > 0 {
			fi.maxRecentHits = max
		}
	}
}

// WithTimeNow overrides the clock provider for deterministic timestamping.
func WithTimeNow(fn func() time.Time) Option {
	return func(fi *FaultInjector) {
		if fn != nil {
			fi.timeNow = fn
		}
	}
}

// WithSleep overrides the sleep provider for testing delays without wall-clock waits.
func WithSleep(fn func(context.Context, time.Duration) error) Option {
	return func(fi *FaultInjector) {
		if fn != nil {
			fi.sleep = fn
		}
	}
}

// FaultInjector manages fault rules, evaluates targets, disrupts streams, and collects metrics.
// Invariant: All exported methods are safe for concurrent access across goroutines.
type FaultInjector struct {
	mu            sync.RWMutex
	rules         map[string]*FaultRule
	ruleOrder     []string
	scenarios     map[string]*FaultScenario
	ruleHits      map[string]int64
	hitsByType    map[FaultType]int64
	hitsByTarget  map[string]int64
	totalHits     int64
	recentHits    []FaultHit
	maxRecentHits int

	rngMu sync.Mutex
	rng   *rand.Rand

	timeNow func() time.Time
	sleep   func(context.Context, time.Duration) error
}

// NewFaultInjector instantiates a thread-safe fault injector.
// Invariant: Initializes internal rule and metrics maps, defaulting to a 100-hit recent history cap.
func NewFaultInjector(opts ...Option) *FaultInjector {
	fi := &FaultInjector{
		rules:         make(map[string]*FaultRule),
		scenarios:     make(map[string]*FaultScenario),
		ruleHits:      make(map[string]int64),
		hitsByType:    make(map[FaultType]int64),
		hitsByTarget:  make(map[string]int64),
		maxRecentHits: 100,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
		timeNow:       time.Now,
		sleep:         defaultSleep,
	}

	for _, opt := range opts {
		opt(fi)
	}

	return fi
}

func defaultSleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// RegisterRule registers or updates a fault rule.
// Invariant: Rule registration preserves insertion order for deterministic target matching.
// Guard: Fails with ErrInvalidRule if rule ID is empty, type is invalid, or probability is out of range.
func (fi *FaultInjector) RegisterRule(rule FaultRule) error {
	if rule.ID == "" {
		return fmt.Errorf("%w: rule ID cannot be empty", ErrInvalidRule)
	}
	if !rule.Type.IsValid() {
		return fmt.Errorf("%w: unrecognized fault type %q", ErrInvalidRule, rule.Type)
	}
	if rule.Probability < 0.0 || rule.Probability > 1.0 {
		return fmt.Errorf("%w: probability must be between 0.0 and 1.0", ErrInvalidRule)
	}
	if rule.Probability == 0.0 {
		rule.Probability = 1.0
	}

	fi.mu.Lock()
	defer fi.mu.Unlock()

	rCopy := rule
	if _, exists := fi.rules[rule.ID]; !exists {
		fi.ruleOrder = append(fi.ruleOrder, rule.ID)
	}
	fi.rules[rule.ID] = &rCopy
	return nil
}

// RemoveRule deletes a rule by ID.
// Invariant: Removes rule from both the lookup table and the deterministic evaluation order.
// Guard: Returns ErrRuleNotFound if the rule ID is not registered.
func (fi *FaultInjector) RemoveRule(id string) error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	if _, ok := fi.rules[id]; !ok {
		return fmt.Errorf("%w: %s", ErrRuleNotFound, id)
	}
	delete(fi.rules, id)

	newOrder := make([]string, 0, len(fi.ruleOrder)-1)
	for _, rID := range fi.ruleOrder {
		if rID != id {
			newOrder = append(newOrder, rID)
		}
	}
	fi.ruleOrder = newOrder
	return nil
}

// EnableRule enables a registered rule.
// Invariant: Modifies active state in place without altering rule ordering.
// Guard: Returns ErrRuleNotFound if the rule ID is not registered.
func (fi *FaultInjector) EnableRule(id string) error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	r, ok := fi.rules[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrRuleNotFound, id)
	}
	r.Active = true
	return nil
}

// DisableRule disables a registered rule.
// Invariant: Inactive rules are bypassed during target matching.
// Guard: Returns ErrRuleNotFound if the rule ID is not registered.
func (fi *FaultInjector) DisableRule(id string) error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	r, ok := fi.rules[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrRuleNotFound, id)
	}
	r.Active = false
	return nil
}

// GetRule returns a copy of the specified rule.
// Invariant: Returns a decoupled copy to prevent caller mutation of internal state.
// Guard: Returns ErrRuleNotFound if the rule ID is not registered.
func (fi *FaultInjector) GetRule(id string) (*FaultRule, error) {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	r, ok := fi.rules[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrRuleNotFound, id)
	}
	rCopy := *r
	return &rCopy, nil
}

// GetRules returns all registered rules in registration order.
// Invariant: Returns deep copies of rules ordered by registration sequence.
func (fi *FaultInjector) GetRules() []FaultRule {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	out := make([]FaultRule, 0, len(fi.ruleOrder))
	for _, id := range fi.ruleOrder {
		if r, ok := fi.rules[id]; ok {
			out = append(out, *r)
		}
	}
	return out
}

// RegisterScenario adds a named scenario and activates its rules if scenario.Enabled is true.
// Invariant: Scenario registration stores scenario metadata and activates constituent rules when enabled.
// Guard: Fails with ErrInvalidRule if scenario ID is empty.
func (fi *FaultInjector) RegisterScenario(scenario FaultScenario) error {
	if scenario.ID == "" {
		return fmt.Errorf("%w: scenario ID cannot be empty", ErrInvalidRule)
	}

	fi.mu.Lock()
	defer fi.mu.Unlock()

	scCopy := scenario
	fi.scenarios[scenario.ID] = &scCopy

	if scenario.Enabled {
		for _, r := range scenario.Rules {
			rCopy := r
			rCopy.Active = true
			if _, exists := fi.rules[r.ID]; !exists {
				fi.ruleOrder = append(fi.ruleOrder, r.ID)
			}
			fi.rules[r.ID] = &rCopy
		}
	}
	return nil
}

// EnableScenario activates a scenario and all associated rules.
// Invariant: Scenario marked enabled and all constituent rules set to active in registry.
// Guard: Returns ErrScenarioNotFound if the scenario ID is not registered.
func (fi *FaultInjector) EnableScenario(id string) error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	sc, ok := fi.scenarios[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrScenarioNotFound, id)
	}
	sc.Enabled = true
	for _, r := range sc.Rules {
		rCopy := r
		rCopy.Active = true
		if _, exists := fi.rules[r.ID]; !exists {
			fi.ruleOrder = append(fi.ruleOrder, r.ID)
		}
		fi.rules[r.ID] = &rCopy
	}
	return nil
}

// DisableScenario deactivates a scenario and its associated rules.
// Invariant: Scenario marked disabled and all constituent rules set to inactive in registry.
// Guard: Returns ErrScenarioNotFound if the scenario ID is not registered.
func (fi *FaultInjector) DisableScenario(id string) error {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	sc, ok := fi.scenarios[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrScenarioNotFound, id)
	}
	sc.Enabled = false
	for _, r := range sc.Rules {
		if existing, ok := fi.rules[r.ID]; ok {
			existing.Active = false
		}
	}
	return nil
}

// GetScenario returns a scenario by ID.
// Invariant: Returns a decoupled copy of the stored scenario.
// Guard: Returns ErrScenarioNotFound if the scenario ID is not registered.
func (fi *FaultInjector) GetScenario(id string) (*FaultScenario, error) {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	sc, ok := fi.scenarios[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrScenarioNotFound, id)
	}
	scCopy := *sc
	return &scCopy, nil
}

// matchRule finds the first matching, active, within-quota rule for target.
func (fi *FaultInjector) matchRule(target string) *FaultRule {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	for _, id := range fi.ruleOrder {
		rule, ok := fi.rules[id]
		if !ok || !rule.Active {
			continue
		}
		if rule.MaxHits > 0 {
			if hits := fi.ruleHits[rule.ID]; hits >= int64(rule.MaxHits) {
				continue
			}
		}
		if !matchTarget(rule.Target, target) {
			continue
		}
		prob := rule.Probability
		if prob <= 0 {
			prob = 1.0
		}
		if prob < 1.0 {
			fi.rngMu.Lock()
			roll := fi.rng.Float64()
			fi.rngMu.Unlock()
			if roll > prob {
				continue
			}
		}
		rCopy := *rule
		return &rCopy
	}
	return nil
}

func matchTarget(pattern, target string) bool {
	if pattern == "" || pattern == "*" {
		return true
	}
	if pattern == target {
		return true
	}
	matched, err := filepath.Match(pattern, target)
	if err == nil && matched {
		return true
	}
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		if strings.HasPrefix(target, prefix) {
			return true
		}
	}
	return false
}

// Inject applies fault rules against an in-memory data payload for a specific target.
// Returns the disrupted (or original) bytes alongside any injected fault error.
// Invariant: Unmatched targets bypass fault injection and return unmodified data with nil error.
// Guard: When rule delay is specified, respect context deadline and fail-closed if canceled.
func (fi *FaultInjector) Inject(ctx context.Context, target string, data []byte) ([]byte, error) {
	rule := fi.matchRule(target)
	if rule == nil {
		return data, nil
	}

	if rule.Delay > 0 {
		if err := fi.sleep(ctx, rule.Delay); err != nil {
			fi.recordHitInternal(rule.ID, rule.Type, target, rule.Delay, err)
			return nil, err
		}
	}

	switch rule.Type {
	case Truncation:
		var payload []byte
		if len(rule.CustomPayload) > 0 {
			payload = rule.CustomPayload
		} else {
			payload = TruncateData(data, rule.TruncateRatio, rule.TruncateBytes)
		}
		fi.recordHitInternal(rule.ID, Truncation, target, rule.Delay, ErrTruncated)
		return payload, ErrTruncated

	case CorruptedJSON:
		var payload []byte
		if len(rule.CustomPayload) > 0 {
			payload = rule.CustomPayload
		} else {
			payload = CorruptJSON(data)
		}
		fi.recordHitInternal(rule.ID, CorruptedJSON, target, rule.Delay, ErrCorruptedJSON)
		return payload, ErrCorruptedJSON

	case LatencySpike:
		fi.recordHitInternal(rule.ID, LatencySpike, target, rule.Delay, nil)
		return data, nil

	case MemoryPressure:
		fi.recordHitInternal(rule.ID, MemoryPressure, target, rule.Delay, ErrMemoryPressure)
		return nil, ErrMemoryPressure

	case NetworkDrop:
		fi.recordHitInternal(rule.ID, NetworkDrop, target, rule.Delay, ErrNetworkDrop)
		return nil, ErrNetworkDrop

	case HostReset:
		fi.recordHitInternal(rule.ID, HostReset, target, rule.Delay, ErrHostReset)
		return nil, ErrHostReset

	default:
		fi.recordHitInternal(rule.ID, rule.Type, target, rule.Delay, ErrFaultInjected)
		return nil, ErrFaultInjected
	}
}

// InterceptReader wraps an io.Reader to inject streaming faults (truncation, delays, drops, corruptions).
// Invariant: Stream evaluation is deferred until the initial Read call on the returned reader.
func (fi *FaultInjector) InterceptReader(ctx context.Context, target string, r io.Reader) io.Reader {
	return &faultReader{
		ctx:        ctx,
		target:     target,
		underlying: r,
		injector:   fi,
	}
}

type faultReader struct {
	ctx        context.Context
	target     string
	underlying io.Reader
	injector   *FaultInjector

	mu        sync.Mutex
	evaluated bool
	rule      *FaultRule
	bytesRead int64
	delayed   bool
	hitRecord bool
}

// Read reads data from the underlying stream while injecting active fault conditions.
// Invariant: Halts reading immediately with simulated errors (e.g. ErrNetworkDrop) when matched.
func (fr *faultReader) Read(p []byte) (int, error) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.evaluated {
		fr.rule = fr.injector.matchRule(fr.target)
		fr.evaluated = true
	}

	if fr.rule == nil {
		return fr.underlying.Read(p)
	}

	if fr.rule.Delay > 0 && !fr.delayed {
		fr.delayed = true
		if err := fr.injector.sleep(fr.ctx, fr.rule.Delay); err != nil {
			if !fr.hitRecord {
				fr.hitRecord = true
				fr.injector.recordHitInternal(fr.rule.ID, fr.rule.Type, fr.target, fr.rule.Delay, err)
			}
			return 0, err
		}
	}

	switch fr.rule.Type {
	case NetworkDrop:
		if !fr.hitRecord {
			fr.hitRecord = true
			fr.injector.recordHitInternal(fr.rule.ID, NetworkDrop, fr.target, fr.rule.Delay, ErrNetworkDrop)
		}
		return 0, ErrNetworkDrop

	case HostReset:
		if !fr.hitRecord {
			fr.hitRecord = true
			fr.injector.recordHitInternal(fr.rule.ID, HostReset, fr.target, fr.rule.Delay, ErrHostReset)
		}
		return 0, ErrHostReset

	case MemoryPressure:
		if !fr.hitRecord {
			fr.hitRecord = true
			fr.injector.recordHitInternal(fr.rule.ID, MemoryPressure, fr.target, fr.rule.Delay, ErrMemoryPressure)
		}
		return 0, ErrMemoryPressure

	case Truncation:
		limit := int64(fr.rule.TruncateBytes)
		if limit <= 0 {
			if fr.rule.TruncateRatio > 0.0 && fr.rule.TruncateRatio < 1.0 {
				limit = int64(float64(len(p)) * fr.rule.TruncateRatio)
			}
			if limit <= 0 {
				limit = 16
			}
		}

		if fr.bytesRead >= limit {
			if !fr.hitRecord {
				fr.hitRecord = true
				fr.injector.recordHitInternal(fr.rule.ID, Truncation, fr.target, fr.rule.Delay, ErrTruncated)
			}
			return 0, io.EOF
		}

		remain := limit - fr.bytesRead
		buf := p
		if int64(len(buf)) > remain {
			buf = buf[:remain]
		}

		n, err := fr.underlying.Read(buf)
		fr.bytesRead += int64(n)

		if fr.bytesRead >= limit && !fr.hitRecord {
			fr.hitRecord = true
			fr.injector.recordHitInternal(fr.rule.ID, Truncation, fr.target, fr.rule.Delay, ErrTruncated)
		}
		return n, err

	case CorruptedJSON:
		n, err := fr.underlying.Read(p)
		if n > 0 {
			corrupted := CorruptJSON(p[:n])
			copy(p[:n], corrupted)
			if !fr.hitRecord {
				fr.hitRecord = true
				fr.injector.recordHitInternal(fr.rule.ID, CorruptedJSON, fr.target, fr.rule.Delay, ErrCorruptedJSON)
			}
		}
		return n, err

	case LatencySpike:
		n, err := fr.underlying.Read(p)
		if !fr.hitRecord {
			fr.hitRecord = true
			fr.injector.recordHitInternal(fr.rule.ID, LatencySpike, fr.target, fr.rule.Delay, nil)
		}
		return n, err

	default:
		return fr.underlying.Read(p)
	}
}

// Close closes the underlying reader if it implements io.Closer.
// Invariant: Calling Close on a reader whose underlying source does not implement io.Closer is a safe no-op.
func (fr *faultReader) Close() error {
	if closer, ok := fr.underlying.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// RecordHit manually records an observed fault hit into the injector's ledger.
// Invariant: Hit metrics and recent hit ring buffers are updated under mutual exclusion.
func (fi *FaultInjector) RecordHit(ruleID string, faultType FaultType, target string, err error) {
	fi.recordHitInternal(ruleID, faultType, target, 0, err)
}

func (fi *FaultInjector) recordHitInternal(ruleID string, faultType FaultType, target string, latency time.Duration, err error) {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	fi.totalHits++
	fi.ruleHits[ruleID]++
	fi.hitsByType[faultType]++
	fi.hitsByTarget[target]++

	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	hit := FaultHit{
		RuleID:    ruleID,
		Type:      faultType,
		Target:    target,
		Timestamp: fi.timeNow(),
		Latency:   latency,
		Error:     errStr,
	}

	fi.recentHits = append(fi.recentHits, hit)
	if len(fi.recentHits) > fi.maxRecentHits {
		fi.recentHits = fi.recentHits[len(fi.recentHits)-fi.maxRecentHits:]
	}
}

// Report returns an aggregated snapshot of all recorded fault metrics.
// Invariant: Returns deep-copied maps and slices to protect internal counters from external mutation.
func (fi *FaultInjector) Report() FaultReport {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	byType := make(map[FaultType]int64, len(fi.hitsByType))
	for k, v := range fi.hitsByType {
		byType[k] = v
	}

	byTarget := make(map[string]int64, len(fi.hitsByTarget))
	for k, v := range fi.hitsByTarget {
		byTarget[k] = v
	}

	byRule := make(map[string]int64, len(fi.ruleHits))
	for k, v := range fi.ruleHits {
		byRule[k] = v
	}

	recent := make([]FaultHit, len(fi.recentHits))
	copy(recent, fi.recentHits)

	activeRules := 0
	for _, r := range fi.rules {
		if r.Active {
			activeRules++
		}
	}

	activeScenarios := 0
	for _, sc := range fi.scenarios {
		if sc.Enabled {
			activeScenarios++
		}
	}

	return FaultReport{
		TotalInjections: fi.totalHits,
		HitsByType:      byType,
		HitsByTarget:    byTarget,
		HitsByRule:      byRule,
		RecentHits:      recent,
		ActiveRules:     activeRules,
		ActiveScenarios: activeScenarios,
	}
}

// Reset clears all registered rules, scenarios, and collected metrics.
// Invariant: Restores injector state to an empty registry and zeroed counters.
func (fi *FaultInjector) Reset() {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	fi.rules = make(map[string]*FaultRule)
	fi.ruleOrder = nil
	fi.scenarios = make(map[string]*FaultScenario)
	fi.ruleHits = make(map[string]int64)
	fi.hitsByType = make(map[FaultType]int64)
	fi.hitsByTarget = make(map[string]int64)
	fi.totalHits = 0
	fi.recentHits = nil
}

// ResetMetrics clears hit counts and history while preserving rules and scenarios.
// Invariant: Rule definitions and scenario configurations remain intact.
func (fi *FaultInjector) ResetMetrics() {
	fi.mu.Lock()
	defer fi.mu.Unlock()

	fi.ruleHits = make(map[string]int64)
	fi.hitsByType = make(map[FaultType]int64)
	fi.hitsByTarget = make(map[string]int64)
	fi.totalHits = 0
	fi.recentHits = nil
}

// CorruptJSON produces malformed bytes guaranteed to fail standard JSON parsing.
// Invariant: The returned payload fails json.Valid verification.
func CorruptJSON(data []byte) []byte {
	if len(data) == 0 {
		return []byte(`{"faultlab_corrupted": `)
	}

	out := make([]byte, len(data))
	copy(out, data)

	// Invalidate closing braces/brackets
	for i := len(out) - 1; i >= 0; i-- {
		if out[i] == '}' || out[i] == ']' {
			out[i] = '?'
			return out
		}
	}

	// Invalidate quotes
	for i := len(out) - 1; i >= 0; i-- {
		if out[i] == '"' {
			out[i] = '\n'
			return out
		}
	}

	// Append trailing malformed tokens
	return append(out, []byte(` , :::: "unclosed`)...)
}

// TruncateData slices data according to the provided ratio or explicit byte limit.
// Invariant: Never exceeds len(data); slices up to maxBytes if > 0, else by ratio in (0.0, 1.0), defaulting to half.
func TruncateData(data []byte, ratio float64, maxBytes int) []byte {
	if len(data) == 0 {
		return data
	}
	if maxBytes > 0 {
		if maxBytes < len(data) {
			return data[:maxBytes]
		}
		return data
	}
	if ratio > 0.0 && ratio < 1.0 {
		cut := int(float64(len(data)) * ratio)
		if cut <= 0 {
			cut = 1
		}
		if cut > len(data) {
			cut = len(data)
		}
		return data[:cut]
	}
	half := len(data) / 2
	if half == 0 && len(data) > 0 {
		half = 1
	}
	return data[:half]
}
