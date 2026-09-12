package agentbench

import (
	"errors"
	"fmt"
	"sort"
)

type ComparisonInput struct {
	ArmID, ManifestDigest, ModelID, TokenizerID, RendererID, LoadDigest, CachePreconditionDigest string
	QualifiedConcurrency                                                                         int
	Replicates                                                                                   []ComparisonReplicate
}

type ComparisonReplicate struct {
	Pair                     int                 `json:"pair"`
	Seed                     uint64              `json:"seed"`
	Order                    int                 `json:"order"`
	SharedC2                 normalQualification `json:"shared_c2"`
	OwnCapacity              normalQualification `json:"own_capacity"`
	SharedC2TasksAccepted    bool                `json:"shared_c2_tasks_accepted"`
	OwnCapacityTasksAccepted bool                `json:"own_capacity_tasks_accepted"`
	BootstrapMillis          int64               `json:"bootstrap_ms"`
	BootstrapKnown           bool                `json:"bootstrap_known"`
	EvidenceStatus           string              `json:"evidence_status"`
	EvidenceReason           string              `json:"evidence_reason,omitempty"`
	MetricMillis             map[string]int64    `json:"metric_ms"`
}

type ComparisonResult struct {
	Status                  string                  `json:"status"`
	Reason                  string                  `json:"reason"`
	Pairs                   int                     `json:"pairs"`
	ArmOrder                []string                `json:"arm_order,omitempty"`
	Metrics                 map[string]PairedMetric `json:"metrics,omitempty"`
	MetricCohort            string                  `json:"metric_cohort,omitempty"`
	OwnQualifiedConcurrency map[string]int          `json:"own_qualified_concurrency,omitempty"`
}

type PairedMetric struct {
	Pairs               int     `json:"pairs"`
	MedianDeltaMillis   int64   `json:"median_delta_ms"`
	BootstrapLowMillis  int64   `json:"bootstrap_low_ms"`
	BootstrapHighMillis int64   `json:"bootstrap_high_ms"`
	Confidence          float64 `json:"confidence"`
}

func ComparePaired(a, b ComparisonInput) (ComparisonResult, error) {
	if a.ArmID == "" || b.ArmID == "" || a.ArmID == b.ArmID {
		return ComparisonResult{}, errors.New("comparison requires two distinct arm identities")
	}
	if err := validateComparisonTimings(a); err != nil {
		return ComparisonResult{}, err
	}
	if err := validateComparisonTimings(b); err != nil {
		return ComparisonResult{}, err
	}
	inconclusive := func(reason string) (ComparisonResult, error) {
		return ComparisonResult{Status: "INCONCLUSIVE", Reason: reason}, nil
	}
	if a.ManifestDigest == "" || a.ModelID == "" || a.TokenizerID == "" || a.RendererID == "" || a.LoadDigest == "" || a.CachePreconditionDigest == "" || a.ManifestDigest != b.ManifestDigest || a.ModelID != b.ModelID || a.TokenizerID != b.TokenizerID || a.RendererID != b.RendererID || a.LoadDigest != b.LoadDigest || a.CachePreconditionDigest != b.CachePreconditionDigest || a.QualifiedConcurrency <= 0 || b.QualifiedConcurrency <= 0 {
		return inconclusive("comparison envelope identity mismatch")
	}
	if len(a.Replicates) != 5 || len(b.Replicates) != 5 {
		return inconclusive("exactly five paired replicates are required")
	}
	left, right := orderedReplicates(a.Replicates), orderedReplicates(b.Replicates)
	orders := make([]string, 5)
	for i := 0; i < 5; i++ {
		if left[i].Pair != i+1 || right[i].Pair != i+1 || left[i].Seed != 0xA63E1001+uint64(i) || right[i].Seed != left[i].Seed || left[i].Order != right[i].Order {
			return inconclusive("paired seed/order identity mismatch")
		}
		if left[i].Order != i%2 {
			return inconclusive("arm order must alternate AB/BA")
		}
		if !left[i].SharedC2.Qualified || !right[i].SharedC2.Qualified || !left[i].OwnCapacity.Qualified || !right[i].OwnCapacity.Qualified || !left[i].SharedC2TasksAccepted || !right[i].SharedC2TasksAccepted || !left[i].OwnCapacityTasksAccepted || !right[i].OwnCapacityTasksAccepted {
			return inconclusive("both shared C2 and own-capacity runs must qualify")
		}
		if left[i].Order == 0 {
			orders[i] = a.ArmID + "," + b.ArmID
		} else {
			orders[i] = b.ArmID + "," + a.ArmID
		}
	}
	keys := commonMetricKeys(left, right)
	if len(keys) == 0 {
		return inconclusive("no matched paired metrics")
	}
	metrics := map[string]PairedMetric{}
	bootstrapDelta := make([]int64, 0, 5)
	for i := range left {
		if left[i].BootstrapKnown && right[i].BootstrapKnown {
			bootstrapDelta = append(bootstrapDelta, right[i].BootstrapMillis-left[i].BootstrapMillis)
		}
	}
	if len(bootstrapDelta) == 5 {
		metrics["bootstrap_ms"] = pairedMetric(bootstrapDelta)
	}
	for _, key := range keys {
		deltas := make([]int64, 5)
		for i := range deltas {
			deltas[i] = right[i].MetricMillis[key] - left[i].MetricMillis[key]
		}
		metrics[key] = pairedMetric(deltas)
	}
	return ComparisonResult{Status: "COMPARABLE", Reason: "five matched shared-C2 paired replicates with accepted task cohorts", Pairs: 5, ArmOrder: orders, Metrics: metrics, MetricCohort: "shared-c2", OwnQualifiedConcurrency: map[string]int{a.ArmID: a.QualifiedConcurrency, b.ArmID: b.QualifiedConcurrency}}, nil
}

func validateComparisonTimings(v ComparisonInput) error {
	for _, r := range v.Replicates {
		if r.Pair <= 0 || r.Order < 0 || r.Order > 1 || r.BootstrapMillis < 0 {
			return errors.New("comparison contains invalid replicate metadata")
		}
		for name, value := range r.MetricMillis {
			if name == "" || value < 0 {
				return fmt.Errorf("comparison contains invalid metric %q", name)
			}
		}
	}
	return nil
}
func orderedReplicates(in []ComparisonReplicate) []ComparisonReplicate {
	out := append([]ComparisonReplicate(nil), in...)
	sort.Slice(out, func(i, j int) bool { return out[i].Pair < out[j].Pair })
	return out
}
func commonMetricKeys(a, b []ComparisonReplicate) []string {
	if len(a) == 0 {
		return nil
	}
	seen := map[string]bool{}
	for k := range a[0].MetricMillis {
		seen[k] = true
	}
	for _, set := range append(a, b...) {
		for k := range seen {
			if _, ok := set.MetricMillis[k]; !ok {
				delete(seen, k)
			}
		}
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
func pairedMetric(deltas []int64) PairedMetric {
	sorted := append([]int64(nil), deltas...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	med := sorted[len(sorted)/2]
	distribution := make([]int64, 0, 3125)
	indices := make([]int, len(deltas))
	for {
		sample := make([]int64, len(deltas))
		for i, index := range indices {
			sample[i] = deltas[index]
		}
		sort.Slice(sample, func(i, j int) bool { return sample[i] < sample[j] })
		distribution = append(distribution, sample[len(sample)/2])
		p := len(indices) - 1
		for p >= 0 {
			indices[p]++
			if indices[p] < len(deltas) {
				break
			}
			indices[p] = 0
			p--
		}
		if p < 0 {
			break
		}
	}
	sort.Slice(distribution, func(i, j int) bool { return distribution[i] < distribution[j] })
	low := distribution[(len(distribution)*25)/1000]
	high := distribution[(len(distribution)*975)/1000]
	return PairedMetric{Pairs: len(deltas), MedianDeltaMillis: med, BootstrapLowMillis: low, BootstrapHighMillis: high, Confidence: .95}
}
