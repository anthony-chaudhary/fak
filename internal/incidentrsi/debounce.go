package incidentrsi

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

const debounceSnapshotSchema = "fak-incident-rsi-debounce/1"

// AdmissionState mirrors the closed state vocabulary of the incident trigger
// receipt without depending on that receipt's concrete Go types.
type AdmissionState string

const (
	AdmissionCollecting         AdmissionState = "COLLECTING"
	AdmissionThresholdReady     AdmissionState = "THRESHOLD_READY"
	AdmissionMaxWaitReady       AdmissionState = "MAX_WAIT_READY"
	AdmissionCooldownSuppressed AdmissionState = "COOLDOWN_SUPPRESSED"
)

// AdmissionReason is the bounded reason vocabulary emitted with every decision.
type AdmissionReason string

const (
	AdmissionBelowThreshold AdmissionReason = "BELOW_THRESHOLD"
	AdmissionByThreshold    AdmissionReason = "THRESHOLD_REACHED"
	AdmissionByMaxWait      AdmissionReason = "MAX_WAIT_REACHED"
	AdmissionDuringCooldown AdmissionReason = "COOLDOWN_ACTIVE"
)

// DebounceConfig controls one persistent debounce/admission domain.
type DebounceConfig struct {
	Threshold        int
	CollectionWindow time.Duration
	MaxWait          time.Duration
	Cooldown         time.Duration
	MaxEntries       int
	MaxReplayIDs     int
	MaxBackwardSkew  time.Duration
	MaxForwardSkew   time.Duration
	LatencyBuckets   []time.Duration
}

// DefaultDebounceConfig bounds both retained state and cross-restart clock skew.
func DefaultDebounceConfig() DebounceConfig {
	return DebounceConfig{
		Threshold:        3,
		CollectionWindow: 30 * time.Second,
		MaxWait:          5 * time.Minute,
		Cooldown:         30 * time.Minute,
		MaxEntries:       1024,
		MaxReplayIDs:     64,
		MaxBackwardSkew:  time.Minute,
		MaxForwardSkew:   10 * time.Minute,
		LatencyBuckets:   []time.Duration{time.Second, 5 * time.Second, 30 * time.Second, time.Minute, 5 * time.Minute},
	}
}

// DebounceObservation contains only privacy-safe identity. Fingerprint must be
// the content-free fingerprint declared by the trigger contract.
type DebounceObservation struct {
	Fingerprint   string
	ContractMajor int
	ObservationID string
	ObservedAt    time.Time
}

// AdmissionDecision is directly mappable to the debounce portion of a trigger
// receipt. Identity fields belong in receipts/ledgers, never metric labels.
type AdmissionDecision struct {
	Fingerprint       string
	ContractMajor     int
	ObservationID     string
	State             AdmissionState
	Reason            AdmissionReason
	FirstSeen         time.Time
	LastSeen          time.Time
	OccurrenceCount   int
	Threshold         int
	WindowEnds        time.Time
	MaxWaitEnds       time.Time
	LastAdmission     time.Time
	NextEligibleAt    time.Time
	AdmissionID       string
	Admitted          bool
	Replay            bool
	ClockSkewAdjusted bool
}

// DebounceOutcome preserves the product fault even when debounce maintenance
// fails. Callers may launch only when Decision.Admitted is true.
type DebounceOutcome struct {
	ProductFault     error
	Decision         AdmissionDecision
	MaintenanceError error
}

// Clock returns times with a monotonic reading when the implementation can.
// RealClock uses time.Now, so elapsed comparisons are monotonic in-process.
type Clock interface {
	Now() time.Time
}

type RealClock struct{}

func (RealClock) Now() time.Time { return time.Now().UTC() }

// DebounceStore persists a complete bounded snapshot atomically from the
// component's perspective.
type DebounceStore interface {
	Load() (DebounceSnapshot, error)
	Save(DebounceSnapshot) error
}

// DebounceSnapshot is exported so alternate ledger stores can persist it.
type DebounceSnapshot struct {
	Schema  string          `json:"schema"`
	Entries []DebounceEntry `json:"entries"`
	Metrics DebounceMetrics `json:"metrics"`
}

// DebounceEntry is the durable state for one fingerprint/contract-major key.
type DebounceEntry struct {
	Fingerprint     string          `json:"fingerprint"`
	ContractMajor   int             `json:"contract_major"`
	FirstSeen       time.Time       `json:"first_seen"`
	LastSeen        time.Time       `json:"last_seen"`
	OccurrenceCount int             `json:"occurrence_count"`
	Threshold       int             `json:"threshold"`
	WindowEnds      time.Time       `json:"window_ends"`
	MaxWaitEnds     time.Time       `json:"max_wait_ends"`
	LastAdmission   time.Time       `json:"last_admission,omitempty"`
	NextEligibleAt  time.Time       `json:"next_eligible_at,omitempty"`
	AdmissionID     string          `json:"admission_id,omitempty"`
	AdmissionReason AdmissionReason `json:"admission_reason,omitempty"`
	Replays         []ReplayRecord  `json:"replays,omitempty"`
}

// ReplayRecord makes retries idempotent across process restart.
type ReplayRecord struct {
	ObservationID string            `json:"observation_id"`
	Decision      AdmissionDecision `json:"decision"`
}

// DebounceMetrics contains bounded aggregate counters and fixed histogram
// buckets. It intentionally has no fingerprint, observation, or admission labels.
type DebounceMetrics struct {
	Observations         uint64   `json:"observations"`
	Collecting           uint64   `json:"collecting"`
	ThresholdAdmissions  uint64   `json:"threshold_admissions"`
	MaxWaitAdmissions    uint64   `json:"max_wait_admissions"`
	CooldownSuppressions uint64   `json:"cooldown_suppressions"`
	Replays              uint64   `json:"replays"`
	Evictions            uint64   `json:"evictions"`
	PersistenceFailures  uint64   `json:"persistence_failures"`
	ClockAdjustments     uint64   `json:"clock_adjustments"`
	LatencyUpperMillis   []int64  `json:"latency_upper_ms"`
	LatencyCounts        []uint64 `json:"latency_counts"`
}

// FileDebounceStore is a deterministic JSON snapshot store.
type FileDebounceStore struct{ Path string }

func (s FileDebounceStore) Load() (DebounceSnapshot, error) {
	data, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return DebounceSnapshot{}, nil
	}
	if err != nil {
		return DebounceSnapshot{}, err
	}
	var snapshot DebounceSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return DebounceSnapshot{}, err
	}
	return snapshot, nil
}

func (s FileDebounceStore) Save(snapshot DebounceSnapshot) error {
	if s.Path == "" {
		return errors.New("incidentrsi: empty debounce store path")
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return err
	}
	data, err := json.Marshal(snapshot)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.Path), ".incidentrsi-debounce-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, s.Path); err == nil {
		return nil
	}
	// Windows cannot replace an existing file with Rename. The store remains
	// fail-closed: remove only the explicitly configured snapshot, then replace.
	if err := os.Remove(s.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(tmpName, s.Path)
}

// Debouncer serializes decisions and persists each mutation before exposing an
// admission. A failed save therefore cannot authorize a duplicate launch.
type Debouncer struct {
	mu      sync.Mutex
	config  DebounceConfig
	store   DebounceStore
	clock   Clock
	entries map[string]DebounceEntry
	metrics DebounceMetrics
}

// NewDebouncer restores persisted state. Unknown snapshot schemas fail closed.
func NewDebouncer(config DebounceConfig, store DebounceStore, clock Clock) (*Debouncer, error) {
	config = normalizeDebounceConfig(config)
	if store == nil {
		return nil, errors.New("incidentrsi: debounce store is required")
	}
	if clock == nil {
		clock = RealClock{}
	}
	d := &Debouncer{config: config, store: store, clock: clock, entries: make(map[string]DebounceEntry)}
	d.metrics.LatencyUpperMillis = durationMillis(config.LatencyBuckets)
	d.metrics.LatencyCounts = make([]uint64, len(config.LatencyBuckets)+1)
	snapshot, err := store.Load()
	if err != nil {
		return nil, fmt.Errorf("incidentrsi: load debounce state: %w", err)
	}
	if snapshot.Schema != "" && snapshot.Schema != debounceSnapshotSchema {
		return nil, fmt.Errorf("incidentrsi: unsupported debounce schema %q", snapshot.Schema)
	}
	for _, entry := range snapshot.Entries {
		if err := validateEntry(entry); err != nil {
			return nil, err
		}
		d.entries[entryKey(entry.Fingerprint, entry.ContractMajor)] = entry
	}
	if snapshot.Schema != "" {
		d.metrics = snapshot.Metrics
		d.metrics.LatencyUpperMillis = durationMillis(config.LatencyBuckets)
		if len(d.metrics.LatencyCounts) != len(config.LatencyBuckets)+1 {
			d.metrics.LatencyCounts = make([]uint64, len(config.LatencyBuckets)+1)
		}
	}
	d.trimToLimit()
	return d, nil
}

// Handle preserves originalFault independently from maintenance status.
func (d *Debouncer) Handle(originalFault error, observation DebounceObservation) DebounceOutcome {
	decision, err := d.Observe(observation)
	return DebounceOutcome{ProductFault: originalFault, Decision: decision, MaintenanceError: err}
}

// Observe records one occurrence. Exact threshold beats max-wait; an active
// cooldown suppresses either ready state. Exact cooldown expiry is eligible.
func (d *Debouncer) Observe(observation DebounceObservation) (AdmissionDecision, error) {
	if observation.Fingerprint == "" || observation.ContractMajor <= 0 || observation.ObservationID == "" {
		return AdmissionDecision{}, errors.New("incidentrsi: fingerprint, positive contract major, and observation ID are required")
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	key := entryKey(observation.Fingerprint, observation.ContractMajor)
	entry, exists := d.entries[key]
	if exists {
		if replay, ok := findReplay(entry.Replays, observation.ObservationID); ok {
			replay.Replay = true
			d.metrics.Replays++
			return replay, nil
		}
	}

	now := observation.ObservedAt
	fromClock := now.IsZero()
	if fromClock {
		now = d.clock.Now()
	}
	now = now.UTC()
	adjusted := false
	if exists && fromClock {
		now, adjusted = d.boundRestartSkew(now, entry.LastSeen)
	}
	if adjusted {
		d.metrics.ClockAdjustments++
	}

	if !exists {
		d.evictFor(key)
		entry = d.newEntry(observation, now)
	} else {
		burstComplete := entry.AdmissionID != "" || entry.OccurrenceCount >= entry.Threshold
		maxWaitDue := !now.Before(entry.MaxWaitEnds)
		quietWindowExpired := !now.Before(entry.WindowEnds)
		if burstComplete || (quietWindowExpired && !maxWaitDue) {
			lastAdmission, nextEligible := entry.LastAdmission, entry.NextEligibleAt
			entry = d.newEntry(observation, now)
			entry.LastAdmission, entry.NextEligibleAt = lastAdmission, nextEligible
		} else {
			entry.OccurrenceCount++
			entry.LastSeen = now
			entry.WindowEnds = now.Add(d.config.CollectionWindow)
		}
	}

	decision := d.decide(entry, observation.ObservationID, now)
	decision.ClockSkewAdjusted = adjusted
	if decision.Admitted {
		entry.LastAdmission = now
		entry.NextEligibleAt = now.Add(d.config.Cooldown)
		entry.AdmissionReason = decision.Reason
		entry.AdmissionID = admissionIdentity(key, entry.FirstSeen, decision.Reason)
		decision.LastAdmission = entry.LastAdmission
		decision.NextEligibleAt = entry.NextEligibleAt
		decision.AdmissionID = entry.AdmissionID
	}
	entry.Replays = append(entry.Replays, ReplayRecord{ObservationID: observation.ObservationID, Decision: decision})
	entry.Replays = boundedReplays(entry.Replays, d.config.MaxReplayIDs)
	d.entries[key] = entry
	d.recordDecision(decision)

	if err := d.store.Save(d.snapshot()); err != nil {
		d.metrics.PersistenceFailures++
		// Admission is not authorized until the identity and cooldown are durable.
		decision.Admitted = false
		return decision, fmt.Errorf("incidentrsi: persist debounce decision: %w", err)
	}
	return decision, nil
}

// Metrics returns a copy of bounded aggregate telemetry.
func (d *Debouncer) Metrics() DebounceMetrics {
	d.mu.Lock()
	defer d.mu.Unlock()
	m := d.metrics
	m.LatencyUpperMillis = append([]int64(nil), m.LatencyUpperMillis...)
	m.LatencyCounts = append([]uint64(nil), m.LatencyCounts...)
	return m
}

func (d *Debouncer) newEntry(observation DebounceObservation, now time.Time) DebounceEntry {
	return DebounceEntry{
		Fingerprint: observation.Fingerprint, ContractMajor: observation.ContractMajor,
		FirstSeen: now, LastSeen: now, OccurrenceCount: 1, Threshold: d.config.Threshold,
		WindowEnds: now.Add(d.config.CollectionWindow), MaxWaitEnds: now.Add(d.config.MaxWait),
	}
}

func (d *Debouncer) decide(entry DebounceEntry, observationID string, now time.Time) AdmissionDecision {
	decision := AdmissionDecision{
		Fingerprint: entry.Fingerprint, ContractMajor: entry.ContractMajor, ObservationID: observationID,
		FirstSeen: entry.FirstSeen, LastSeen: entry.LastSeen, OccurrenceCount: entry.OccurrenceCount,
		Threshold: entry.Threshold, WindowEnds: entry.WindowEnds, MaxWaitEnds: entry.MaxWaitEnds,
		LastAdmission: entry.LastAdmission, NextEligibleAt: entry.NextEligibleAt,
	}
	readyState, readyReason := AdmissionCollecting, AdmissionBelowThreshold
	if entry.OccurrenceCount >= entry.Threshold {
		readyState, readyReason = AdmissionThresholdReady, AdmissionByThreshold
	} else if !now.Before(entry.MaxWaitEnds) {
		readyState, readyReason = AdmissionMaxWaitReady, AdmissionByMaxWait
	}
	if readyState == AdmissionCollecting {
		decision.State, decision.Reason = readyState, readyReason
		return decision
	}
	if !entry.NextEligibleAt.IsZero() && now.Before(entry.NextEligibleAt) {
		decision.State, decision.Reason = AdmissionCooldownSuppressed, AdmissionDuringCooldown
		return decision
	}
	decision.State, decision.Reason, decision.Admitted = readyState, readyReason, true
	return decision
}

func (d *Debouncer) boundRestartSkew(now, last time.Time) (time.Time, bool) {
	if now.Before(last) {
		// Backward time never decreases persisted state. The configured bound is
		// retained as policy documentation; all backward movement clamps to last.
		return last, true
	}
	if d.config.MaxForwardSkew > 0 && now.Sub(last) > d.config.MaxForwardSkew {
		return last.Add(d.config.MaxForwardSkew), true
	}
	return now, false
}

func (d *Debouncer) evictFor(newKey string) {
	if _, exists := d.entries[newKey]; exists || len(d.entries) < d.config.MaxEntries {
		return
	}
	keys := make([]string, 0, len(d.entries))
	for key := range d.entries {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := d.entries[keys[i]], d.entries[keys[j]]
		if a.LastSeen.Equal(b.LastSeen) {
			return keys[i] < keys[j]
		}
		return a.LastSeen.Before(b.LastSeen)
	})
	delete(d.entries, keys[0])
	d.metrics.Evictions++
}

func (d *Debouncer) trimToLimit() {
	for len(d.entries) > d.config.MaxEntries {
		d.evictFor("__restore__")
	}
}

func (d *Debouncer) snapshot() DebounceSnapshot {
	entries := make([]DebounceEntry, 0, len(d.entries))
	for _, entry := range d.entries {
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool {
		return entryKey(entries[i].Fingerprint, entries[i].ContractMajor) < entryKey(entries[j].Fingerprint, entries[j].ContractMajor)
	})
	return DebounceSnapshot{Schema: debounceSnapshotSchema, Entries: entries, Metrics: d.metrics}
}

func (d *Debouncer) recordDecision(decision AdmissionDecision) {
	d.metrics.Observations++
	switch {
	case decision.Admitted && decision.Reason == AdmissionByThreshold:
		d.metrics.ThresholdAdmissions++
	case decision.Admitted && decision.Reason == AdmissionByMaxWait:
		d.metrics.MaxWaitAdmissions++
	case decision.State == AdmissionCooldownSuppressed:
		d.metrics.CooldownSuppressions++
	default:
		d.metrics.Collecting++
	}
	latency := decision.LastSeen.Sub(decision.FirstSeen)
	bucket := len(d.config.LatencyBuckets)
	for i, upper := range d.config.LatencyBuckets {
		if latency <= upper {
			bucket = i
			break
		}
	}
	d.metrics.LatencyCounts[bucket]++
}

func normalizeDebounceConfig(config DebounceConfig) DebounceConfig {
	defaults := DefaultDebounceConfig()
	if config.Threshold <= 0 {
		config.Threshold = defaults.Threshold
	}
	if config.CollectionWindow <= 0 {
		config.CollectionWindow = defaults.CollectionWindow
	}
	if config.MaxWait < config.CollectionWindow {
		config.MaxWait = defaults.MaxWait
	}
	if config.MaxWait < config.CollectionWindow {
		config.MaxWait = config.CollectionWindow
	}
	if config.Cooldown <= 0 {
		config.Cooldown = defaults.Cooldown
	}
	if config.MaxEntries <= 0 {
		config.MaxEntries = defaults.MaxEntries
	}
	if config.MaxReplayIDs <= 0 {
		config.MaxReplayIDs = defaults.MaxReplayIDs
	}
	if config.MaxBackwardSkew <= 0 {
		config.MaxBackwardSkew = defaults.MaxBackwardSkew
	}
	if config.MaxForwardSkew <= 0 {
		config.MaxForwardSkew = defaults.MaxForwardSkew
	}
	if len(config.LatencyBuckets) == 0 {
		config.LatencyBuckets = defaults.LatencyBuckets
	}
	config.LatencyBuckets = append([]time.Duration(nil), config.LatencyBuckets...)
	sort.Slice(config.LatencyBuckets, func(i, j int) bool { return config.LatencyBuckets[i] < config.LatencyBuckets[j] })
	return config
}

func validateEntry(entry DebounceEntry) error {
	if entry.Fingerprint == "" || entry.ContractMajor <= 0 || entry.Threshold <= 0 || entry.FirstSeen.IsZero() || entry.LastSeen.Before(entry.FirstSeen) {
		return errors.New("incidentrsi: invalid persisted debounce entry")
	}
	return nil
}

func entryKey(fingerprint string, major int) string {
	return fmt.Sprintf("%d\x00%s", major, fingerprint)
}

func admissionIdentity(key string, firstSeen time.Time, reason AdmissionReason) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("incidentrsi-admission-v1\x00%s\x00%s\x00%s", key, firstSeen.UTC().Format(time.RFC3339Nano), reason)))
	return "irsi-admit-v1-" + hex.EncodeToString(h[:])
}

func findReplay(records []ReplayRecord, id string) (AdmissionDecision, bool) {
	for _, record := range records {
		if record.ObservationID == id {
			return record.Decision, true
		}
	}
	return AdmissionDecision{}, false
}

func boundedReplays(records []ReplayRecord, limit int) []ReplayRecord {
	if len(records) <= limit {
		return records
	}
	sort.Slice(records, func(i, j int) bool {
		a, b := records[i].Decision.LastSeen, records[j].Decision.LastSeen
		if a.Equal(b) {
			return records[i].ObservationID < records[j].ObservationID
		}
		return a.Before(b)
	})
	return append([]ReplayRecord(nil), records[len(records)-limit:]...)
}

func durationMillis(values []time.Duration) []int64 {
	result := make([]int64, len(values))
	for i, value := range values {
		result[i] = value.Milliseconds()
	}
	return result
}
