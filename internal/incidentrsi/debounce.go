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

// BurstStatus is the externally visible state of one incident burst.
type BurstStatus string

const (
	BurstCollecting         BurstStatus = "collecting"
	BurstThresholdReady     BurstStatus = "threshold_ready"
	BurstMaxWaitReady       BurstStatus = "max_wait_ready"
	BurstCooldownSuppressed BurstStatus = "cooldown_suppressed"
)

// DebounceConfig keeps burst collection, liveness, cooldown, and retention
// independent. MaxWait is measured from FirstSeen and may extend past the
// collection window; observations after WindowEnd do not join that burst.
type DebounceConfig struct {
	Threshold        int
	CollectionWindow time.Duration
	MaxWait          time.Duration
	Cooldown         time.Duration
	Retention        time.Duration
	MaxEntries       int
	MaxObservations  int
	MaxClockSkew     time.Duration
}

func DefaultDebounceConfig() DebounceConfig {
	return DebounceConfig{
		Threshold: 3, CollectionWindow: 5 * time.Minute, MaxWait: 15 * time.Minute,
		Cooldown: 30 * time.Minute, Retention: 24 * time.Hour, MaxEntries: 1024,
		MaxObservations: 4096, MaxClockSkew: 5 * time.Minute,
	}
}

// Observation identifies a retry and its producer compatibility boundary.
// Fingerprint must already be privacy-safe; ProducerMajor prevents incompatible
// producer contracts from sharing state.
type Observation struct {
	Fingerprint   string
	ProducerMajor int
	ObservationID string
}

// BurstRecord is the complete persisted state for one keyed burst.
type BurstRecord struct {
	Fingerprint      string      `json:"fingerprint"`
	ProducerMajor    int         `json:"producer_major"`
	BurstID          string      `json:"burst_id"`
	FirstSeen        time.Time   `json:"first_seen"`
	LastSeen         time.Time   `json:"last_seen"`
	OccurrenceCount  int         `json:"occurrence_count"`
	Threshold        int         `json:"threshold"`
	WindowEnd        time.Time   `json:"window_end"`
	MaxWaitDeadline  time.Time   `json:"max_wait_deadline"`
	Status           BurstStatus `json:"status"`
	AdmissionID      string      `json:"admission_id,omitempty"`
	LastAdmission    time.Time   `json:"last_admission,omitempty"`
	NextEligibleTime time.Time   `json:"next_eligible_time,omitempty"`
	ObservationIDs   []string    `json:"observation_ids,omitempty"`
}

// IncidentTrigger is the bounded decision projection consumed by the sibling
// fak-incident-rsi-trigger/1 contract. It intentionally contains no raw error.
type IncidentTrigger struct {
	Schema           string      `json:"schema"`
	Fingerprint      string      `json:"fingerprint"`
	ProducerMajor    int         `json:"producer_major"`
	BurstID          string      `json:"burst_id"`
	ObservationID    string      `json:"observation_id"`
	State            BurstStatus `json:"state"`
	OccurrenceCount  int         `json:"occurrence_count"`
	Threshold        int         `json:"threshold"`
	FirstSeen        time.Time   `json:"first_seen"`
	LastSeen         time.Time   `json:"last_seen"`
	WindowEnd        time.Time   `json:"window_end"`
	MaxWaitDeadline  time.Time   `json:"max_wait_deadline"`
	AdmissionID      string      `json:"admission_id,omitempty"`
	LastAdmission    time.Time   `json:"last_admission,omitempty"`
	NextEligibleTime time.Time   `json:"next_eligible_time,omitempty"`
}

// DebounceDecision returns the original product failure unchanged. Sidecar
// persistence/maintenance failures are bounded separately so they cannot hide it.
type DebounceDecision struct {
	Trigger          IncidentTrigger
	Admitted         bool
	ProductFailure   error
	MaintenanceError error
}

// BurstStore persists a complete bounded snapshot atomically.
type BurstStore interface {
	Load() ([]BurstRecord, error)
	Save([]BurstRecord) error
}

// MemoryBurstStore is a restart-capable store for embedders and tests.
type MemoryBurstStore struct {
	mu                   sync.Mutex
	records              []BurstRecord
	LoadError, SaveError error
}

func (s *MemoryBurstStore) Load() ([]BurstRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.LoadError != nil {
		return nil, s.LoadError
	}
	return cloneRecords(s.records), nil
}
func (s *MemoryBurstStore) Save(records []BurstRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.SaveError != nil {
		return s.SaveError
	}
	s.records = cloneRecords(records)
	return nil
}

// FileBurstStore stores JSON using write-and-rename in the destination directory.
type FileBurstStore struct{ Path string }

func (s FileBurstStore) Load() ([]BurstRecord, error) {
	data, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var records []BurstRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, err
	}
	return records, nil
}
func (s FileBurstStore) Save(records []BurstRecord) error {
	data, err := json.Marshal(records)
	if err != nil {
		return err
	}
	dir := filepath.Dir(s.Path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".incident-rsi-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	ok := false
	defer func() {
		tmp.Close()
		if !ok {
			_ = os.Remove(name)
		}
	}()
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err = os.Rename(name, s.Path); err != nil {
		return err
	}
	ok = true
	return nil
}

type Clock interface{ Now() time.Time }
type wallClock struct{}

func (wallClock) Now() time.Time { return time.Now() }

// DebounceMetrics contains only bounded aggregate dimensions.
type DebounceMetrics struct {
	Decisions           map[BurstStatus]uint64
	Admissions          uint64
	PersistenceFailures uint64
	ClockSkewClamps     uint64
	LatencyBuckets      [5]uint64
}

// Debouncer owns durable burst admission. Its mutex makes one process linearizable;
// saving the admitted identity before returning makes retries exact-once across restart.
type Debouncer struct {
	mu      sync.Mutex
	cfg     DebounceConfig
	store   BurstStore
	clock   Clock
	records map[string]BurstRecord
	metrics DebounceMetrics
	loadErr error
}

func NewDebouncer(cfg DebounceConfig, store BurstStore, clock Clock) *Debouncer {
	d := DefaultDebounceConfig()
	if cfg.Threshold <= 0 {
		cfg.Threshold = d.Threshold
	}
	if cfg.CollectionWindow <= 0 {
		cfg.CollectionWindow = d.CollectionWindow
	}
	if cfg.MaxWait <= 0 {
		cfg.MaxWait = d.MaxWait
	}
	if cfg.Cooldown <= 0 {
		cfg.Cooldown = d.Cooldown
	}
	if cfg.Retention <= 0 {
		cfg.Retention = d.Retention
	}
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = d.MaxEntries
	}
	if cfg.MaxObservations <= 0 {
		cfg.MaxObservations = d.MaxObservations
	}
	if cfg.MaxClockSkew <= 0 {
		cfg.MaxClockSkew = d.MaxClockSkew
	}
	if store == nil {
		store = &MemoryBurstStore{}
	}
	if clock == nil {
		clock = wallClock{}
	}
	b := &Debouncer{cfg: cfg, store: store, clock: clock, records: map[string]BurstRecord{}, metrics: DebounceMetrics{Decisions: map[BurstStatus]uint64{}}}
	records, err := store.Load()
	b.loadErr = err
	if err == nil {
		for _, r := range records {
			if validRecord(r) {
				b.records[keyFor(r.Fingerprint, r.ProducerMajor)] = r
			}
		}
	}
	return b
}

// Observe records an occurrence and atomically admits a ready burst. Boundary
// precedence is: existing admission retry, cooldown suppression, threshold at
// or before WindowEnd, maximum wait at or after its deadline, then collecting.
func (d *Debouncer) Observe(obs Observation, productFailure error) DebounceDecision {
	return d.decide(obs, productFailure, true, d.clock.Now())
}

func (d *Debouncer) decide(obs Observation, productFailure error, addOccurrence bool, now time.Time) DebounceDecision {
	d.mu.Lock()
	defer d.mu.Unlock()
	result := DebounceDecision{ProductFailure: productFailure}
	if d.loadErr != nil {
		result.MaintenanceError = fmt.Errorf("load debounce state: %w", d.loadErr)
		d.metrics.PersistenceFailures++
		return result
	}
	if obs.Fingerprint == "" || obs.ProducerMajor <= 0 || obs.ObservationID == "" {
		result.MaintenanceError = errors.New("invalid incident debounce observation")
		return result
	}
	before := cloneRecordMap(d.records)
	key := keyFor(obs.Fingerprint, obs.ProducerMajor)
	r, exists := d.records[key]
	if exists {
		now = d.clampNow(now, r.LastSeen)
	}

	// A retry for an admitted burst returns its stable identity. Once its
	// collection window closes, a new observation starts the next burst while
	// retaining the prior cooldown boundary.
	if exists && r.AdmissionID != "" && !now.After(r.WindowEnd) {
		return d.finish(r, obs.ObservationID, false, result)
	}
	if !exists || (r.AdmissionID != "" && now.After(r.WindowEnd)) {
		if !exists && len(d.records) >= d.cfg.MaxEntries && !d.evictOne(now) {
			result.MaintenanceError = errors.New("incident debounce capacity is fully protected")
			return result
		}
		r = newBurst(obs, now, d.cfg, r)
	}

	if addOccurrence && !contains(r.ObservationIDs, obs.ObservationID) && !now.After(r.WindowEnd) {
		if len(r.ObservationIDs) >= d.cfg.MaxObservations {
			result.MaintenanceError = errors.New("incident burst observation capacity reached")
			return result
		}
		r.ObservationIDs = append(r.ObservationIDs, obs.ObservationID)
		r.OccurrenceCount++
		if now.After(r.LastSeen) {
			r.LastSeen = now
		}
	}

	ready := BurstCollecting
	if r.OccurrenceCount >= r.Threshold && !now.After(r.WindowEnd) {
		ready = BurstThresholdReady
	} else if !now.Before(r.MaxWaitDeadline) {
		ready = BurstMaxWaitReady
	}
	if ready != BurstCollecting && !r.NextEligibleTime.IsZero() && now.Before(r.NextEligibleTime) {
		r.Status = BurstCooldownSuppressed
	} else {
		r.Status = ready
	}

	admitted := false
	if (r.Status == BurstThresholdReady || r.Status == BurstMaxWaitReady) && r.AdmissionID == "" {
		r.AdmissionID = stableID("admission", r.BurstID)
		r.LastAdmission = now
		r.NextEligibleTime = now.Add(d.cfg.Cooldown)
		admitted = true
	}
	d.records[key] = r
	if err := d.store.Save(d.snapshot()); err != nil {
		d.records = before
		result.MaintenanceError = fmt.Errorf("save debounce state: %w", err)
		d.metrics.PersistenceFailures++
		return result
	}
	return d.finish(r, obs.ObservationID, admitted, result)
}

// Tick advances liveness without adding an occurrence.
func (d *Debouncer) Tick(fingerprint string, producerMajor int, productFailure error) DebounceDecision {
	now := d.clock.Now()
	return d.decide(Observation{Fingerprint: fingerprint, ProducerMajor: producerMajor, ObservationID: stableID("tick", fingerprint, fmt.Sprint(producerMajor), now.String())}, productFailure, false, now)
}

func (d *Debouncer) Metrics() DebounceMetrics {
	d.mu.Lock()
	defer d.mu.Unlock()
	m := d.metrics
	m.Decisions = map[BurstStatus]uint64{}
	for k, v := range d.metrics.Decisions {
		m.Decisions[k] = v
	}
	return m
}

func (d *Debouncer) finish(r BurstRecord, observationID string, admitted bool, out DebounceDecision) DebounceDecision {
	out.Admitted = admitted
	out.Trigger = IncidentTrigger{Schema: "fak-incident-rsi-trigger/1", Fingerprint: r.Fingerprint, ProducerMajor: r.ProducerMajor, BurstID: r.BurstID, ObservationID: observationID, State: r.Status, OccurrenceCount: r.OccurrenceCount, Threshold: r.Threshold, FirstSeen: r.FirstSeen, LastSeen: r.LastSeen, WindowEnd: r.WindowEnd, MaxWaitDeadline: r.MaxWaitDeadline, AdmissionID: r.AdmissionID, LastAdmission: r.LastAdmission, NextEligibleTime: r.NextEligibleTime}
	d.metrics.Decisions[r.Status]++
	if admitted {
		d.metrics.Admissions++
	}
	latency := r.LastSeen.Sub(r.FirstSeen)
	i := 4
	for j, b := range []time.Duration{time.Second, time.Minute, 5 * time.Minute, time.Hour} {
		if latency <= b {
			i = j
			break
		}
	}
	d.metrics.LatencyBuckets[i]++
	return out
}

func (d *Debouncer) clampNow(now, last time.Time) time.Time {
	if now.Before(last) {
		d.metrics.ClockSkewClamps++
		return last
	}
	if d.cfg.MaxClockSkew > 0 && now.Sub(last) > d.cfg.MaxClockSkew {
		// Forward time remains authoritative for liveness, but the bounded counter
		// exposes the discontinuity; persisted deadlines prevent duplicate admission.
		d.metrics.ClockSkewClamps++
	}
	return now
}

func (d *Debouncer) evictOne(now time.Time) bool {
	candidates := make([]BurstRecord, 0, len(d.records))
	for _, r := range d.records {
		protectedUntil := r.LastSeen.Add(d.cfg.Retention)
		if r.NextEligibleTime.After(protectedUntil) {
			protectedUntil = r.NextEligibleTime
		}
		if !now.Before(protectedUntil) {
			candidates = append(candidates, r)
		}
	}
	if len(candidates) == 0 {
		return false
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].LastSeen.Equal(candidates[j].LastSeen) {
			return keyFor(candidates[i].Fingerprint, candidates[i].ProducerMajor) < keyFor(candidates[j].Fingerprint, candidates[j].ProducerMajor)
		}
		return candidates[i].LastSeen.Before(candidates[j].LastSeen)
	})
	delete(d.records, keyFor(candidates[0].Fingerprint, candidates[0].ProducerMajor))
	return true
}

func newBurst(obs Observation, now time.Time, cfg DebounceConfig, prior BurstRecord) BurstRecord {
	return BurstRecord{Fingerprint: obs.Fingerprint, ProducerMajor: obs.ProducerMajor, BurstID: stableID("burst", obs.Fingerprint, fmt.Sprint(obs.ProducerMajor), now.UTC().Format(time.RFC3339Nano)), FirstSeen: now, LastSeen: now, Threshold: cfg.Threshold, WindowEnd: now.Add(cfg.CollectionWindow), MaxWaitDeadline: now.Add(cfg.MaxWait), Status: BurstCollecting, LastAdmission: prior.LastAdmission, NextEligibleTime: prior.NextEligibleTime}
}
func keyFor(f string, major int) string { return fmt.Sprintf("%d\x00%s", major, f) }
func stableID(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
func validRecord(r BurstRecord) bool {
	return r.Fingerprint != "" && r.ProducerMajor > 0 && r.BurstID != "" && r.Threshold > 0 && !r.FirstSeen.IsZero() && !r.WindowEnd.Before(r.FirstSeen) && !r.MaxWaitDeadline.Before(r.FirstSeen)
}
func cloneRecords(in []BurstRecord) []BurstRecord {
	out := append([]BurstRecord(nil), in...)
	for i := range out {
		out[i].ObservationIDs = append([]string(nil), out[i].ObservationIDs...)
	}
	return out
}
func (d *Debouncer) snapshot() []BurstRecord {
	out := make([]BurstRecord, 0, len(d.records))
	for _, r := range d.records {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		return keyFor(out[i].Fingerprint, out[i].ProducerMajor) < keyFor(out[j].Fingerprint, out[j].ProducerMajor)
	})
	return cloneRecords(out)
}
func cloneRecordMap(in map[string]BurstRecord) map[string]BurstRecord {
	out := make(map[string]BurstRecord, len(in))
	for key, record := range in {
		record.ObservationIDs = append([]string(nil), record.ObservationIDs...)
		out[key] = record
	}
	return out
}
