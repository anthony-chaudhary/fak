package gateway

import (
	"context"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
)

func TestFeatureTrackerLifecycle(t *testing.T) {
	catalog := FeatureCatalog{Schema: FeatureSchema, Features: []FeatureStatus{
		{Feature: FeatureVDSO, State: FeatureConfiguredActive},
		{Feature: FeatureElideResults, State: FeatureConfiguredStandby},
		{Feature: FeatureNativeModel, State: FeatureDisabled},
		{Feature: FeatureMetal, State: FeatureRefusedUnavailable},
	}}
	tracker := NewFeatureActivationTracker(catalog)
	catalog.Features[0].Feature = FeatureMetal
	ctx := WithFeatureActivationTracker(context.Background(), tracker)
	if FeatureActivationTrackerFromContext(ctx) != tracker || FeatureActivationTrackerFromContext(context.Background()) != nil {
		t.Fatal("request tracker context isolation failed")
	}
	wantEnabled := []ServeFeature{FeatureElideResults, FeatureVDSO}
	if got := tracker.Snapshot(); !reflect.DeepEqual(got.Enabled, wantEnabled) || len(got.Used) != 0 {
		t.Fatalf("configuration is not activation: %+v", got)
	}
	if tracker.RecordActivation(ServeFeature("unknown"), FeatureOutcomeUsed) || tracker.RecordActivation(FeatureVDSO, "planned") {
		t.Fatal("accepted unknown feature or non-used outcome")
	}
	var accepted atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed) {
				accepted.Add(1)
			}
			_ = tracker.Snapshot()
		}()
	}
	wg.Wait()
	if accepted.Load() != 1 {
		t.Fatalf("duplicate activation accepted %d times", accepted.Load())
	}
	// A real observed operation remains visible even if catalog configuration disagrees.
	if !tracker.RecordActivation(FeaturePolicyFloor, FeatureOutcomeUsed) {
		t.Fatal("suppressed observed use absent from configuration")
	}
	snapshot := tracker.Snapshot()
	snapshot.Enabled[0] = FeatureMetal
	snapshot.Used[0] = FeatureMetal
	final, first := tracker.Finalize()
	if !first || !reflect.DeepEqual(final.Enabled, wantEnabled) || !reflect.DeepEqual(final.Used, []ServeFeature{FeaturePolicyFloor, FeatureVDSO}) {
		t.Fatalf("final snapshot aliases caller or lost use: %+v first=%v", final, first)
	}
	if tracker.RecordActivation(FeatureElideResults, FeatureOutcomeUsed) {
		t.Fatal("accepted activation after finalization")
	}
	if _, first := tracker.Finalize(); first {
		t.Fatal("finalization counted twice")
	}
}

func TestFeatureTrackerNilSafety(t *testing.T) {
	var tracker *FeatureActivationTracker
	if tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed) {
		t.Fatal("nil tracker accepted activation")
	}
	if got := tracker.Snapshot(); len(got.Enabled) != 0 || len(got.Used) != 0 {
		t.Fatalf("nil snapshot=%+v", got)
	}
	if _, first := tracker.Finalize(); first {
		t.Fatal("nil tracker finalized a request")
	}
}
