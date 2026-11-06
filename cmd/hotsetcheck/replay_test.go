package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// replay_test.go — witnesses for the hotsetcheck fixture driver: determinism, the skew ordering of
// hit rates, the bypass gate on a jitter workload, and the recorded-trace input path. All fixtures
// are small and in-memory; every assertion is on closed counters, never on output prose.

// TestHotSetCheckDeterministic pins the determinism contract: the same options yield byte-identical
// receipts, so a reported hit rate is reproducible from the flags alone.
//
// fak-test:runtime fast est=300ms lane=default
func TestHotSetCheckDeterministic(t *testing.T) {
	opts := Options{Tokens: 40, ZipfExponent: 1.2, HotSetSize: 96, Seed: 7}
	a, err := runFixture(opts)
	if err != nil {
		t.Fatalf("runFixture: %v", err)
	}
	b, err := runFixture(opts)
	if err != nil {
		t.Fatalf("runFixture (repeat): %v", err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatalf("receipts diverged across identical runs:\n%s\nvs\n%s", ja, jb)
	}
}

// TestHotSetCheckSkewRaisesHitRate witnesses the decisive quantity's ordering: at the same budget,
// a skewed (Zipf) trace must hit strictly more often than a uniform one, on BOTH arms — the
// working set concentrates where the ring can hold it.
//
// fak-test:runtime fast est=500ms lane=default
func TestHotSetCheckSkewRaisesHitRate(t *testing.T) {
	for _, withLearner := range []bool{false, true} {
		uniformOpts := Options{Tokens: 60, ZipfExponent: 0, Seed: 3}
		uniform, err := runArm(fixtureEvents(t, uniformOpts), uniformOpts, withLearner)
		if err != nil {
			t.Fatalf("uniform arm: %v", err)
		}
		skewedOpts := Options{Tokens: 60, ZipfExponent: 1.5, HotSetSize: 96, Seed: 3}
		skewed, err := runArm(fixtureEvents(t, skewedOpts), skewedOpts, withLearner)
		if err != nil {
			t.Fatalf("skewed arm: %v", err)
		}
		if skewed.hitRate() <= uniform.hitRate() {
			t.Fatalf("withLearner=%v: skewed hit rate %.4f not above uniform %.4f",
				withLearner, skewed.hitRate(), uniform.hitRate())
		}
	}
}

// TestHotSetCheckBypassGateOnJitter witnesses the hysteresis decision end-to-end: a stable two-hot
// expert plus a stream of cold one-offs. The learner-wired arm must BYPASS the one-offs (no hot
// eviction, DecodeBypasses > 0) while the baseline LRU arm evicts its incumbents (HotEvictions > 0).
//
// fak-test:runtime fast est=50ms lane=default
func TestHotSetCheckBypassGateOnJitter(t *testing.T) {
	// Hand-built jitter stream: hot experts 0/1 re-touched between single touches of cold experts.
	var events []model.ExpertCacheTraceEvent
	for round := 0; round < 8; round++ {
		events = append(events, model.ExpertCacheTraceEvent{Layer: 0, Expert: 0})
		events = append(events, model.ExpertCacheTraceEvent{Layer: 0, Expert: 1})
		events = append(events, model.ExpertCacheTraceEvent{Layer: 0, Expert: 10 + round})
	}
	opts := Options{
		Shape:          model.V41ExpertCacheShape{Layers: 1, Experts: 32, TopK: 3},
		ExpertBytes:    100,
		HotBytes:       200, // holds exactly the two hot experts
		TransientBytes: 200,
		LearnFraction:  0.5, // learn on the first half, measure on the second
		SweepEvery:     -1,  // no sweeps: isolate the decode gate
	}
	base, err := runArm(events, opts, false)
	if err != nil {
		t.Fatalf("baseline arm: %v", err)
	}
	learn, err := runArm(events, opts, true)
	if err != nil {
		t.Fatalf("learner arm: %v", err)
	}
	if base.Stats.HotEvictions == 0 {
		t.Fatalf("baseline LRU performed no hot evictions on the jitter stream; the fixture does not exercise the gate")
	}
	if learn.Stats.DecodeBypasses == 0 {
		t.Fatalf("learner arm bypassed nothing; the hysteresis gate did not engage")
	}
	if learn.Stats.HotEvictions != 0 {
		t.Fatalf("learner arm evicted %d hot entries under the band, want 0 (the incumbents stay)", learn.Stats.HotEvictions)
	}
	if learn.hitRate() < base.hitRate() {
		t.Fatalf("learner arm hit rate %.4f below baseline %.4f on the jitter stream", learn.hitRate(), base.hitRate())
	}
}

// TestHotSetCheckTraceFileRoundTrip witnesses the --trace input path: a recorded event file flows
// through the same driver, and a malformed file is refused with a typed error rather than parsed
// into a phantom slot.
//
// fak-test:runtime fast est=20ms lane=default
func TestHotSetCheckTraceFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.json")
	evs := []model.ExpertCacheTraceEvent{{Layer: 0, Expert: 0}, {Layer: 0, Expert: 1}, {Layer: 0, Expert: 0}}
	body, _ := json.Marshal(traceFile{Schema: traceFileSchema, Events: []traceFileEvent{{0, 0}, {0, 1}, {0, 0}}})
	if err := os.WriteFile(good, body, 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadTraceFile(good)
	if err != nil {
		t.Fatalf("loadTraceFile: %v", err)
	}
	if len(loaded) != len(evs) {
		t.Fatalf("loaded %d events, want %d", len(loaded), len(evs))
	}
	for i := range evs {
		if loaded[i].Layer != evs[i].Layer || loaded[i].Expert != evs[i].Expert {
			t.Fatalf("event %d = (%d,%d), want (%d,%d)", i, loaded[i].Layer, loaded[i].Expert, evs[i].Layer, evs[i].Expert)
		}
	}

	badSchema := filepath.Join(dir, "bad-schema.json")
	if err := os.WriteFile(badSchema, []byte(`{"schema":"wrong","events":[{"layer":0,"expert":0}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadTraceFile(badSchema); err == nil {
		t.Fatal("wrong schema accepted; a foreign envelope must be refused")
	}
	negCoord := filepath.Join(dir, "neg.json")
	if err := os.WriteFile(negCoord, []byte(`{"schema":"`+traceFileSchema+`","events":[{"layer":-1,"expert":0}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadTraceFile(negCoord); err == nil {
		t.Fatal("negative coordinates accepted; a malformed touch must not land on a real slot")
	}
}

// fixtureEvents builds the shared synthetic stream for a test, mirroring buildTrace's synthetic arm.
func fixtureEvents(t *testing.T, opts Options) []model.ExpertCacheTraceEvent {
	t.Helper()
	trace, err := model.GenerateV41ExpertCacheTrace(model.ExpertCacheTraceOptions{
		Shape:        model.V41ExpertCacheShapeDefault(),
		Seed:         opts.Seed,
		Tokens:       opts.Tokens,
		HotSetSize:   opts.HotSetSize,
		ZipfExponent: opts.ZipfExponent,
		ExpertBytes:  1 << 20,
	})
	if err != nil {
		t.Fatalf("GenerateV41ExpertCacheTrace: %v", err)
	}
	return trace.Events
}
