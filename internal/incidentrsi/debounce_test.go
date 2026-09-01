package incidentrsi

import (
	"errors"
	"sync"
	"testing"
	"time"
)

type fakeClock struct{ now time.Time }

func (c *fakeClock) Now() time.Time  { return c.now }
func (c *fakeClock) set(t time.Time) { c.now = t }

func debounceConfig() DebounceConfig {
	return DebounceConfig{Threshold: 3, CollectionWindow: 10 * time.Second, MaxWait: 20 * time.Second, Cooldown: time.Minute, Retention: time.Hour, MaxEntries: 8, MaxObservations: 32, MaxClockSkew: time.Minute}
}
func observation(id string, major int) Observation {
	return Observation{Fingerprint: "irsi-v1-safe", ProducerMajor: major, ObservationID: id}
}

func TestDebounceBoundaryPrecedence(t *testing.T) {
	start := time.Unix(1000, 0).UTC()
	tests := []struct {
		name                string
		threshold           int
		window, maxWait, at time.Duration
		observations        int
		want                BurstStatus
		admitted            bool
	}{
		{"collecting", 3, 10 * time.Second, 20 * time.Second, 5 * time.Second, 1, BurstCollecting, false},
		{"threshold exact before window", 2, 10 * time.Second, 20 * time.Second, 9 * time.Second, 2, BurstThresholdReady, true},
		{"threshold wins at coincident boundary", 2, 10 * time.Second, 10 * time.Second, 10 * time.Second, 2, BurstThresholdReady, true},
		{"max wait exact below threshold", 3, 5 * time.Second, 10 * time.Second, 10 * time.Second, 1, BurstMaxWaitReady, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := debounceConfig()
			cfg.Threshold = tt.threshold
			cfg.CollectionWindow = tt.window
			cfg.MaxWait = tt.maxWait
			clock := &fakeClock{now: start}
			d := NewDebouncer(cfg, &MemoryBurstStore{}, clock)
			var got DebounceDecision
			for i := 0; i < tt.observations; i++ {
				if i > 0 && i == tt.observations-1 {
					clock.set(start.Add(tt.at))
				}
				got = d.Observe(observation(string(rune('a'+i)), 1), errors.New("product fault"))
			}
			if tt.observations == 1 && tt.at > 0 {
				clock.set(start.Add(tt.at))
				got = d.Tick("irsi-v1-safe", 1, errors.New("product fault"))
			}
			if got.Trigger.State != tt.want || got.Admitted != tt.admitted {
				t.Fatalf("decision=%+v want state=%s admitted=%v", got, tt.want, tt.admitted)
			}
			if got.ProductFailure == nil || got.ProductFailure.Error() != "product fault" {
				t.Fatalf("product failure replaced: %+v", got)
			}
		})
	}
}

func TestDebounceCooldownAndNextEligibilityBoundary(t *testing.T) {
	start := time.Unix(2000, 0).UTC()
	cfg := debounceConfig()
	cfg.Threshold = 1
	cfg.CollectionWindow = time.Second
	store := &MemoryBurstStore{}
	clock := &fakeClock{now: start}
	d := NewDebouncer(cfg, store, clock)
	first := d.Observe(observation("one", 1), nil)
	clock.set(start.Add(2 * time.Second))
	suppressed := d.Observe(observation("two", 1), nil)
	if suppressed.Trigger.State != BurstCooldownSuppressed || suppressed.Admitted {
		t.Fatalf("suppressed=%+v", suppressed)
	}
	clock.set(first.Trigger.NextEligibleTime)
	ready := d.Tick("irsi-v1-safe", 1, nil)
	if ready.Trigger.State != BurstMaxWaitReady || !ready.Admitted {
		t.Fatalf("eligible boundary=%+v", ready)
	}
}

func TestDebounceConcurrentRetriesReturnOneAdmissionIdentity(t *testing.T) {
	start := time.Unix(3000, 0).UTC()
	cfg := debounceConfig()
	cfg.Threshold = 1
	clock := &fakeClock{now: start}
	d := NewDebouncer(cfg, &MemoryBurstStore{}, clock)
	const n = 64
	results := make(chan DebounceDecision, n)
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() { defer wg.Done(); results <- d.Observe(observation("same-retry", 1), nil) }()
	}
	wg.Wait()
	close(results)
	ids := map[string]bool{}
	admitted := 0
	for result := range results {
		ids[result.Trigger.AdmissionID] = true
		if result.Admitted {
			admitted++
		}
	}
	if admitted != 1 || len(ids) != 1 || ids[""] {
		t.Fatalf("admitted=%d identities=%v", admitted, ids)
	}
}

func TestDebounceRestartPreservesReadyAndAdmittedBursts(t *testing.T) {
	start := time.Unix(4000, 0).UTC()
	cfg := debounceConfig()
	cfg.Threshold = 2
	store := &MemoryBurstStore{}
	clock := &fakeClock{now: start}
	first := NewDebouncer(cfg, store, clock)
	collecting := first.Observe(observation("one", 1), nil)
	if collecting.Trigger.State != BurstCollecting {
		t.Fatalf("collecting=%+v", collecting)
	}
	restarted := NewDebouncer(cfg, store, clock)
	admitted := restarted.Observe(observation("two", 1), nil)
	if !admitted.Admitted || admitted.Trigger.AdmissionID == "" {
		t.Fatalf("admitted=%+v", admitted)
	}
	restartedAgain := NewDebouncer(cfg, store, clock)
	retry := restartedAgain.Observe(observation("two", 1), nil)
	if retry.Admitted || retry.Trigger.AdmissionID != admitted.Trigger.AdmissionID {
		t.Fatalf("restart retry=%+v admitted=%+v", retry, admitted)
	}
}

func TestDebounceProducerMajorSeparatesState(t *testing.T) {
	clock := &fakeClock{now: time.Unix(5000, 0).UTC()}
	cfg := debounceConfig()
	cfg.Threshold = 2
	d := NewDebouncer(cfg, &MemoryBurstStore{}, clock)
	if got := d.Observe(observation("v1", 1), nil); got.Trigger.OccurrenceCount != 1 {
		t.Fatal(got.Trigger.OccurrenceCount)
	}
	if got := d.Observe(observation("v2", 2), nil); got.Trigger.OccurrenceCount != 1 {
		t.Fatal(got.Trigger.OccurrenceCount)
	}
}

func TestDebounceBackwardSkewDoesNotMoveStateBackward(t *testing.T) {
	start := time.Unix(6000, 0).UTC()
	clock := &fakeClock{now: start}
	d := NewDebouncer(debounceConfig(), &MemoryBurstStore{}, clock)
	first := d.Observe(observation("one", 1), nil)
	clock.set(start.Add(-time.Hour))
	second := d.Observe(observation("two", 1), nil)
	if second.Trigger.LastSeen.Before(first.Trigger.LastSeen) || d.Metrics().ClockSkewClamps != 1 {
		t.Fatalf("first=%v second=%v metrics=%+v", first.Trigger.LastSeen, second.Trigger.LastSeen, d.Metrics())
	}
}

func TestDebouncePersistenceFailurePreservesProductFailure(t *testing.T) {
	product := errors.New("gateway failed")
	store := &MemoryBurstStore{SaveError: errors.New("disk failed")}
	clock := &fakeClock{now: time.Unix(7000, 0).UTC()}
	cfg := debounceConfig()
	cfg.Threshold = 1
	got := NewDebouncer(cfg, store, clock).Observe(observation("one", 1), product)
	if !errors.Is(got.ProductFailure, product) || got.MaintenanceError == nil || got.Admitted {
		t.Fatalf("failure result=%+v", got)
	}
}

func TestDebounceProtectedAdmissionIsNotEvicted(t *testing.T) {
	start := time.Unix(8000, 0).UTC()
	cfg := debounceConfig()
	cfg.Threshold = 1
	cfg.MaxEntries = 1
	clock := &fakeClock{now: start}
	d := NewDebouncer(cfg, &MemoryBurstStore{}, clock)
	admitted := d.Observe(observation("one", 1), nil)
	other := observation("other", 1)
	other.Fingerprint = "irsi-v1-other"
	got := d.Observe(other, nil)
	if got.MaintenanceError == nil {
		t.Fatalf("protected state was evicted: %+v", got)
	}
	retry := d.Observe(observation("one", 1), nil)
	if retry.Trigger.AdmissionID != admitted.Trigger.AdmissionID || retry.Admitted {
		t.Fatalf("retry=%+v admitted=%+v", retry, admitted)
	}
}

func TestDebounceTriggerAndMetricsAreBounded(t *testing.T) {
	clock := &fakeClock{now: time.Unix(9000, 0).UTC()}
	cfg := debounceConfig()
	cfg.Threshold = 1
	got := NewDebouncer(cfg, &MemoryBurstStore{}, clock).Observe(observation("one", 1), nil)
	if got.Trigger.Schema != "fak-incident-rsi-trigger/1" || got.Trigger.Fingerprint != "irsi-v1-safe" || got.Trigger.BurstID == "" {
		t.Fatalf("trigger=%+v", got.Trigger)
	}
	m := NewDebouncer(cfg, &MemoryBurstStore{}, clock).Metrics()
	if len(m.LatencyBuckets) != 5 {
		t.Fatalf("metrics=%+v", m)
	}
}

func TestDebounceRestartReadyBurstKeepsAdmissionIdentity(t *testing.T) {
	start := time.Unix(10000, 0).UTC()
	cfg := debounceConfig()
	cfg.Threshold = 99
	store := &MemoryBurstStore{}
	clock := &fakeClock{now: start}
	first := NewDebouncer(cfg, store, clock)
	first.Observe(observation("one", 1), nil)
	clock.set(start.Add(cfg.MaxWait))
	admitted := first.Tick("irsi-v1-safe", 1, nil)
	if !admitted.Admitted || admitted.Trigger.State != BurstMaxWaitReady {
		t.Fatalf("admitted=%+v", admitted)
	}
	restarted := NewDebouncer(cfg, store, clock)
	replayed := restarted.Observe(observation("one", 1), nil)
	if replayed.Admitted || replayed.Trigger.State != BurstMaxWaitReady || replayed.Trigger.AdmissionID != admitted.Trigger.AdmissionID {
		t.Fatalf("replayed=%+v admitted=%+v", replayed, admitted)
	}
}
