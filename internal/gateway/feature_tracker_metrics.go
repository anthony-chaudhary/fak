package gateway

import (
	"fmt"
	"strings"
	"sync"
)

// featureActivationMetrics counts requests that actually used a mechanism. Its
// cardinality is bounded by the catalog; no outcome or request data becomes a label.
type featureActivationMetrics struct {
	mu   sync.Mutex
	used map[ServeFeature]uint64
}

// ObserveFinal consumes only the first successful Finalize for each request.
// Repeated uses within that request count once, including duplicate snapshot IDs.
func (m *featureActivationMetrics) ObserveFinal(snapshot FeatureActivationSnapshot) {
	if m == nil {
		return
	}
	seen := make(map[ServeFeature]struct{})
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, feature := range snapshot.Used {
		if !knownServeFeature(feature) {
			continue
		}
		if _, duplicate := seen[feature]; duplicate {
			continue
		}
		seen[feature] = struct{}{}
		if m.used == nil {
			m.used = make(map[ServeFeature]uint64)
		}
		m.used[feature]++
	}
}

// Snapshot returns detached cumulative counts, without resetting them.
func (m *featureActivationMetrics) Snapshot() map[ServeFeature]uint64 {
	result := make(map[ServeFeature]uint64)
	if m == nil {
		return result
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for feature, count := range m.used {
		result[feature] = count
	}
	return result
}

// Render appends a stable Prometheus counter family containing only closed IDs.
func (m *featureActivationMetrics) Render(b *strings.Builder) {
	counts := m.Snapshot()
	features := make(map[ServeFeature]struct{}, len(counts))
	for feature := range counts {
		features[feature] = struct{}{}
	}
	b.WriteString("# HELP fak_gateway_feature_used_requests_total Requests that actually used a feature.\n")
	b.WriteString("# TYPE fak_gateway_feature_used_requests_total counter\n")
	for _, feature := range sortedActivationFeatures(features) {
		fmt.Fprintf(b, "fak_gateway_feature_used_requests_total{feature=%q} %d\n", feature, counts[feature])
	}
}
