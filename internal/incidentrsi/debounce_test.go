package incidentrsi

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

type memoryDebounceStore struct {
	mu       sync.Mutex
	snapshot DebounceSnapshot
	failSave error
}

func (s *memoryDebounceStore) Load() (DebounceSnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneSnapshot(s.snapshot), nil
}

func (s *memoryDebounceStore) Save(snapshot DebounceSnapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failSave != nil {
		return s.failSave
	}
	s.snapshot = cloneSnapshot(snapshot)
	return nil
}

func cloneSnapshot(snapshot DebounceSnapshot) DebounceSnapshot {
	data, _ := json.Marshal(snapshot)
	var clone DebounceSnapshot
	_ = json.Unmarshal(data, &clone)
	return clone
}

type fakeDebounceClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeDebounceClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeDebounceClock) Set(now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = now
}

func testDebounceConfig() DebounceConfig {
	return DebounceConfig{
		Threshold:        3,
		CollectionWindow: 10 * time.Second,
		MaxWait:          30 * time.Second,
		Cooldown:         20 * time.Second,
		MaxEntries:       8,
		MaxReplayIDs:     8,
		MaxBackwardSkew:  2 * time.Second,
		MaxForwardSkew:   5 * time.Second,
		LatencyBuckets:   []time.Duration{time.Second, 10 * time.Second},
	}
}

func observeAt(t *testing.T, d *Debouncer, fingerprint, id string, at time.Time) AdmissionDecision {
	t.Helper()
	decision, err := d.Observe(DebounceObservation{Fingerprint: fingerprint, ContractMajor: 1, ObservationID: id, ObservedAt: at})
	if err != nil {
		t.Fatalf("Observe: %v", err)
	}
	return decision
}

func TestDebounceBoundaryPrecedence(t *testing.T) {
	base := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name string
		run  func(*testing.T, *Debouncer)
	}{
		{
			name: "collecting below threshold",
			run: func(t *testing.T, d *Debouncer) {
				got := observeAt(t, d, "fp", "one", base)
				if got.State != AdmissionCollecting || got.Reason != AdmissionBelowThreshold || got.Admitted {
					t.Fatalf("got %+v", got)
				}
			},
		},
		{
			name: "threshold beats max wait on exact boundary",
			run: func(t *testing.T, d *Debouncer) {
				observeAt(t, d, "fp", "one", base)
				observeAt(t, d, "fp", "two", base.Add(time.Second))
				got := observeAt(t, d, "fp", "three", base.Add(30*time.Second))
				if got.State != AdmissionThresholdReady || got.Reason != AdmissionByThreshold || !got.Admitted {
					t.Fatalf("got %+v", got)
				}
			},
		},
		{
			name: "max wait admits below threshold",
			run: func(t *testing.T, d *Debouncer) {
				observeAt(t, d, "fp", "one", base)
				got := observeAt(t, d, "fp", "two", base.Add(30*time.Second))
				if got.State != AdmissionMaxWaitReady || got.Reason != AdmissionByMaxWait || !got.Admitted {
					t.Fatalf("got %+v", got)
				}
				next := observeAt(t, d, "fp", "three", base.Add(31*time.Second))
				if next.OccurrenceCount != 1 || !next.FirstSeen.Equal(base.Add(31*time.Second)) || next.State != AdmissionCollecting {
					t.Fatalf("next burst got %+v", next)
				}
			},
		},
		{
			name: "quiet window starts a new burst before max wait",
			run: func(t *testing.T, d *Debouncer) {
				observeAt(t, d, "fp", "one", base)
				got := observeAt(t, d, "fp", "two", base.Add(10*time.Second))
				if got.OccurrenceCount != 1 || !got.FirstSeen.Equal(base.Add(10*time.Second)) || got.State != AdmissionCollecting {
					t.Fatalf("got %+v", got)
				}
			},
		},
		{
			name: "cooldown suppresses threshold and exact expiry admits",
			run: func(t *testing.T, d *Debouncer) {
				first := observeAt(t, d, "fp", "a", base)
				observeAt(t, d, "fp", "b", base.Add(time.Second))
				first = observeAt(t, d, "fp", "c", base.Add(2*time.Second))
				if !first.Admitted {
					t.Fatal("first burst not admitted")
				}
				observeAt(t, d, "fp", "d", base.Add(12*time.Second))
				observeAt(t, d, "fp", "e", base.Add(13*time.Second))
				suppressed := observeAt(t, d, "fp", "f", base.Add(14*time.Second))
				if suppressed.State != AdmissionCooldownSuppressed || suppressed.Reason != AdmissionDuringCooldown || suppressed.Admitted {
					t.Fatalf("suppressed got %+v", suppressed)
				}
				observeAt(t, d, "fp", "g", base.Add(22*time.Second))
				observeAt(t, d, "fp", "h", base.Add(22*time.Second))
				admitted := observeAt(t, d, "fp", "i", base.Add(22*time.Second))
				if !admitted.Admitted || admitted.State != AdmissionThresholdReady {
					t.Fatalf("expiry got %+v", admitted)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			d, err := NewDebouncer(testDebounceConfig(), &memoryDebounceStore{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			test.run(t, d)
		})
	}
}

func TestConcurrentAtMostOneAdmissionIdentity(t *testing.T) {
	config := testDebounceConfig()
	config.Threshold = 1
	d, err := NewDebouncer(config, &memoryDebounceStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 8, 31, 13, 0, 0, 0, time.UTC)
	const workers = 64
	results := make(chan AdmissionDecision, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			decision, observeErr := d.Observe(DebounceObservation{
				Fingerprint: "privacy-safe", ContractMajor: 1,
				ObservationID: fmt.Sprintf("obs-%02d", i), ObservedAt: base,
			})
			if observeErr != nil {
				t.Errorf("Observe: %v", observeErr)
				return
			}
			results <- decision
		}(i)
	}
	wg.Wait()
	close(results)
	admissions := 0
	identities := map[string]bool{}
	for result := range results {
		if result.Admitted {
			admissions++
			identities[result.AdmissionID] = true
		}
	}
	if admissions != 1 || len(identities) != 1 {
		t.Fatalf("admissions=%d identities=%v", admissions, identities)
	}
}

func TestReplayAndRestartPreserveReadyAndAdmittedBursts(t *testing.T) {
	store := &memoryDebounceStore{}
	config := testDebounceConfig()
	base := time.Date(2026, 8, 31, 14, 0, 0, 0, time.UTC)
	d1, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	observeAt(t, d1, "fp", "one", base)

	d2, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	admitted := observeAt(t, d2, "fp", "two", base.Add(config.MaxWait))
	if !admitted.Admitted || admitted.AdmissionID == "" {
		t.Fatalf("got %+v", admitted)
	}

	d3, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	replay := observeAt(t, d3, "fp", "two", base.Add(config.MaxWait))
	if !replay.Replay || replay.AdmissionID != admitted.AdmissionID || !replay.Admitted {
		t.Fatalf("admitted=%+v replay=%+v", admitted, replay)
	}
	if got := d3.Metrics(); got.Replays != 1 {
		t.Fatalf("replays=%d", got.Replays)
	}
}

func TestContractMajorPartitionsState(t *testing.T) {
	config := testDebounceConfig()
	config.Threshold = 2
	d, err := NewDebouncer(config, &memoryDebounceStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 8, 31, 15, 0, 0, 0, time.UTC)
	observeAt(t, d, "same-fingerprint", "v1-one", base)
	got, err := d.Observe(DebounceObservation{Fingerprint: "same-fingerprint", ContractMajor: 2, ObservationID: "v2-one", ObservedAt: base})
	if err != nil {
		t.Fatal(err)
	}
	if got.OccurrenceCount != 1 || got.Admitted {
		t.Fatalf("got %+v", got)
	}
}

func TestClockSkewIsBoundedAcrossRestart(t *testing.T) {
	store := &memoryDebounceStore{}
	config := testDebounceConfig()
	base := time.Date(2026, 8, 31, 16, 0, 0, 0, time.UTC)
	clock := &fakeDebounceClock{now: base}
	d1, err := NewDebouncer(config, store, clock)
	if err != nil {
		t.Fatal(err)
	}
	first, err := d1.Observe(DebounceObservation{Fingerprint: "fp", ContractMajor: 1, ObservationID: "one"})
	if err != nil {
		t.Fatal(err)
	}

	clock.Set(base.Add(-time.Hour))
	d2, err := NewDebouncer(config, store, clock)
	if err != nil {
		t.Fatal(err)
	}
	backward, err := d2.Observe(DebounceObservation{Fingerprint: "fp", ContractMajor: 1, ObservationID: "two"})
	if err != nil {
		t.Fatal(err)
	}
	if !backward.ClockSkewAdjusted || backward.LastSeen.Before(first.LastSeen) {
		t.Fatalf("backward got %+v", backward)
	}

	clock.Set(base.Add(24 * time.Hour))
	d3, err := NewDebouncer(config, store, clock)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := d3.Observe(DebounceObservation{Fingerprint: "fp", ContractMajor: 1, ObservationID: "three"})
	if err != nil {
		t.Fatal(err)
	}
	if !forward.ClockSkewAdjusted || forward.LastSeen.Sub(backward.LastSeen) > config.MaxForwardSkew {
		t.Fatalf("forward got %+v", forward)
	}
}

func TestDeterministicEvictionAndBoundedMetrics(t *testing.T) {
	config := testDebounceConfig()
	config.MaxEntries = 2
	store := &memoryDebounceStore{}
	d, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 8, 31, 17, 0, 0, 0, time.UTC)
	observeAt(t, d, "b", "b", base)
	observeAt(t, d, "a", "a", base)
	observeAt(t, d, "c", "c", base.Add(time.Second))
	snapshot, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, entry := range snapshot.Entries {
		got[entry.Fingerprint] = true
	}
	if got["a"] || !got["b"] || !got["c"] || len(got) != 2 {
		t.Fatalf("deterministic eviction got %v", got)
	}
	metrics := d.Metrics()
	if metrics.Evictions != 1 || len(metrics.LatencyCounts) != len(config.LatencyBuckets)+1 {
		t.Fatalf("metrics %+v", metrics)
	}
	encoded, _ := json.Marshal(metrics)
	if string(encoded) == "" || containsAny(string(encoded), "fingerprint", "observation_id", "admission_id") {
		t.Fatalf("identity leaked into metrics: %s", encoded)
	}
}

func TestPersistenceFailurePreservesProductFaultAndDoesNotAuthorizeAdmission(t *testing.T) {
	productFault := errors.New("original gateway transport failure")
	store := &memoryDebounceStore{failSave: errors.New("disk unavailable")}
	config := testDebounceConfig()
	config.Threshold = 1
	d, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	outcome := d.Handle(productFault, DebounceObservation{
		Fingerprint: "fp", ContractMajor: 1, ObservationID: "one",
		ObservedAt: time.Date(2026, 8, 31, 18, 0, 0, 0, time.UTC),
	})
	if !errors.Is(outcome.ProductFault, productFault) || outcome.MaintenanceError == nil || outcome.Decision.Admitted {
		t.Fatalf("outcome %+v", outcome)
	}
}

func TestFileDebounceStoreRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "debounce.json")
	store := FileDebounceStore{Path: path}
	config := testDebounceConfig()
	config.Threshold = 1
	base := time.Date(2026, 8, 31, 19, 0, 0, 0, time.UTC)
	d1, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	admitted := observeAt(t, d1, "fp", "one", base)
	d2, err := NewDebouncer(config, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	replay := observeAt(t, d2, "fp", "one", base)
	if !replay.Replay || replay.AdmissionID != admitted.AdmissionID {
		t.Fatalf("admitted=%+v replay=%+v", admitted, replay)
	}
}

func TestAdmissionVocabularyMatchesTriggerContract(t *testing.T) {
	states := []AdmissionState{AdmissionCollecting, AdmissionThresholdReady, AdmissionMaxWaitReady, AdmissionCooldownSuppressed}
	wantStates := []string{"COLLECTING", "THRESHOLD_READY", "MAX_WAIT_READY", "COOLDOWN_SUPPRESSED"}
	for i := range states {
		if string(states[i]) != wantStates[i] {
			t.Fatalf("state %q", states[i])
		}
	}
	reasons := []AdmissionReason{AdmissionBelowThreshold, AdmissionByThreshold, AdmissionByMaxWait, AdmissionDuringCooldown}
	wantReasons := []string{"BELOW_THRESHOLD", "THRESHOLD_REACHED", "MAX_WAIT_REACHED", "COOLDOWN_ACTIVE"}
	for i := range reasons {
		if string(reasons[i]) != wantReasons[i] {
			t.Fatalf("reason %q", reasons[i])
		}
	}
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		for i := 0; i+len(needle) <= len(value); i++ {
			if value[i:i+len(needle)] == needle {
				return true
			}
		}
	}
	return false
}
