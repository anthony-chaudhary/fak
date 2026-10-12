package computetrace

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestTryEnableWithSampleBudgetPreservesRecorderOwnership(t *testing.T) {
	t.Run("installed recorder", func(t *testing.T) {
		for _, limit := range []int{0, 8} {
			owner, stopOwner := Enable(limit, "owner", "owner-request")
			other, stopOther := TryEnableWithSampleBudget(8, 64, "other", "other-request")
			if other != nil || stopOther != nil {
				if stopOther != nil {
					stopOther()
				}
				stopOwner()
				t.Fatalf("replaced installed owner with event limit %d", limit)
			}
			Record(Event{Operation: "ownership-witness"})
			stopOwner()
			if limit > 0 {
				a := owner.Artifact()
				if len(a.Events) != 1 || a.Events[0].RunID != "owner" || a.Events[0].RequestID != "owner-request" {
					t.Fatalf("owner lost its event or identity: %+v", a)
				}
			}
		}
	})

	t.Run("concurrent claims", func(t *testing.T) {
		type result struct {
			recorder *Recorder
			stop     func()
		}
		start := make(chan struct{})
		results := make(chan result, 2)
		for i := 0; i < 2; i++ {
			go func() {
				<-start
				r, stop := TryEnableWithSampleBudget(8, 64, "race", "race-request")
				results <- result{r, stop}
			}()
		}
		close(start)
		first, second := <-results, <-results
		winners := 0
		for _, got := range []result{first, second} {
			if got.stop != nil {
				defer got.stop()
			}
			if (got.recorder == nil) != (got.stop == nil) {
				t.Fatal("recorder and disable ownership disagree")
			}
			if got.recorder != nil {
				winners++
			}
		}
		if winners != 1 {
			t.Fatalf("concurrent winners = %d, want exactly one", winners)
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestTryEnableWithOptionsPreservesOwnershipAndFiltering(t *testing.T) {
	opts := RecorderOptions{MaxSamples: 8, Operations: []string{"v41.activation"}}
	for _, limit := range []int{0, 8} {
		owner, stopOwner := Enable(limit, "owner", "owner-request")
		other, stopOther := TryEnableWithOptions(1, opts, "other", "other-request")
		if other != nil || stopOther != nil {
			if stopOther != nil {
				stopOther()
			}
			stopOwner()
			t.Fatalf("options claim replaced owner with limit %d", limit)
		}
		Record(Event{Operation: "owner-event"})
		stopOwner()
		if limit > 0 {
			a := owner.Artifact()
			if len(a.Events) != 1 || a.Events[0].Operation != "owner-event" || a.Events[0].RunID != "owner" {
				t.Fatalf("owner or its unfiltered options changed: %+v", a)
			}
		}
	}

	active.RLock()
	previousObserver := active.observer
	active.RUnlock()
	defer SetObserver(previousObserver)
	observed := 0
	SetObserver(func(Event) { observed++ })
	rec, disable := TryEnableWithOptions(1, opts, "filtered", "request")
	if rec == nil || disable == nil {
		t.Fatal("options recorder could not acquire free slot")
	}
	defer disable()
	// The recorder owns its allowlist; mutating caller input cannot change it.
	opts.Operations[0] = "matmul"
	for i := 0; i < 20; i++ {
		Record(Event{Operation: "matmul", SampleValues: make([]float32, 8)})
	}
	Record(Event{Operation: "v41.activation", SampleValues: make([]float32, 8)})
	a := rec.Artifact()
	if len(a.Events) != 1 || a.Events[0].Operation != "v41.activation" || a.RetainedSampleValues != 8 || a.Dropped != 0 || a.DroppedSampleValues != 0 || observed != 21 {
		t.Fatalf("filter/observer/budget contract: artifact=%+v observed=%d", a, observed)
	}
	// An allowed event still consumes the normal bounded event budget.
	Record(Event{Operation: "v41.activation"})
	if rec.Artifact().Dropped != 1 {
		t.Fatal("filter disabled ordinary overflow accounting")
	}

	replacement, stopReplacement := Enable(1, "replacement", "replacement-request")
	defer stopReplacement()
	disable() // a stale options owner must not disable a later explicit owner
	Record(Event{Operation: "replacement-event"})
	if a := replacement.Artifact(); len(a.Events) != 1 || a.Events[0].RunID != "replacement" {
		t.Fatalf("stale disable removed replacement: %+v", a)
	}
}
