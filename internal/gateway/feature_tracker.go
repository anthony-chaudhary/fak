package gateway

import (
	"context"
	"sort"
	"sync"
)

// FeatureOutcomeUsed records an accepted execution, not a configured or planned one.
const FeatureOutcomeUsed = "used"

// FeatureActivationSnapshot separates ingress configuration from observed execution.
// Its slices are detached, sorted, and contain only closed catalog identifiers.
type FeatureActivationSnapshot struct {
	Enabled []ServeFeature
	Used    []ServeFeature
}

// FeatureActivationTracker holds one request's evidence, including concurrent tool work.
// Finalize must run after that request's workers have joined.
type FeatureActivationTracker struct {
	mu         sync.Mutex
	enabled    []ServeFeature
	used       map[ServeFeature]struct{}
	finalized  bool
	incomplete bool
}

// NewFeatureActivationTracker captures configuration without inferring execution.
func NewFeatureActivationTracker(catalog FeatureCatalog) *FeatureActivationTracker {
	enabled := make(map[ServeFeature]struct{})
	for _, status := range catalog.Features {
		if knownServeFeature(status.Feature) && (status.State == FeatureConfiguredActive || status.State == FeatureConfiguredStandby) {
			enabled[status.Feature] = struct{}{}
		}
	}
	return &FeatureActivationTracker{enabled: sortedActivationFeatures(enabled)}
}

type featureActivationContextKey struct{}

// WithFeatureActivationTracker attaches a request-owned tracker to a derived context.
func WithFeatureActivationTracker(ctx context.Context, tracker *FeatureActivationTracker) context.Context {
	return context.WithValue(ctx, featureActivationContextKey{}, tracker)
}

// FeatureActivationTrackerFromContext returns nil when tracking is absent.
func FeatureActivationTrackerFromContext(ctx context.Context) *FeatureActivationTracker {
	if ctx == nil {
		return nil
	}
	tracker, _ := ctx.Value(featureActivationContextKey{}).(*FeatureActivationTracker)
	return tracker
}

// RecordActivation returns true only for the first accepted use of a known feature.
// Callers must witness the execution before recording it. Configuration mismatches
// do not suppress actual evidence; Used may therefore contain an ID absent from Enabled.
func (t *FeatureActivationTracker) RecordActivation(feature ServeFeature, outcome string) bool {
	if t == nil || outcome != FeatureOutcomeUsed || !knownServeFeature(feature) {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.finalized {
		return false
	}
	if _, exists := t.used[feature]; exists {
		return false
	}
	if t.used == nil {
		t.used = make(map[ServeFeature]struct{})
	}
	t.used[feature] = struct{}{}
	return true
}

// Snapshot returns a detached observation without ending the request.
func (t *FeatureActivationTracker) Snapshot() FeatureActivationSnapshot {
	if t == nil {
		return FeatureActivationSnapshot{Enabled: []ServeFeature{}, Used: []ServeFeature{}}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.snapshotLocked()
}

// Finalize freezes execution evidence and returns true exactly once. The caller
// uses that boolean to fold cumulative request metrics once, including error turns.
func (t *FeatureActivationTracker) Finalize() (FeatureActivationSnapshot, bool) {
	if t == nil {
		return t.Snapshot(), false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	first := !t.finalized
	t.finalized = true
	return t.snapshotLocked(), first
}

func (t *FeatureActivationTracker) snapshotLocked() FeatureActivationSnapshot {
	return FeatureActivationSnapshot{
		Enabled: append([]ServeFeature{}, t.enabled...),
		Used:    sortedActivationFeatures(t.used),
	}
}

func sortedActivationFeatures(features map[ServeFeature]struct{}) []ServeFeature {
	result := make([]ServeFeature, 0, len(features))
	for feature := range features {
		result = append(result, feature)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

// markFeatureActivationIncomplete prevents a final trailer from certifying a
// request whose asynchronous worker has not joined (for example, cancellation).
func markFeatureActivationIncomplete(ctx context.Context) {
	t := FeatureActivationTrackerFromContext(ctx)
	if t == nil {
		return
	}
	t.mu.Lock()
	t.incomplete = true
	t.mu.Unlock()
}

func (t *FeatureActivationTracker) completeEvidence() bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return !t.incomplete
}
