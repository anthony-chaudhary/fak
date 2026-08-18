package disambiguation

import "sort"

const MetricsSchemaVersion = "fak-disambiguation-metrics/1"

type MetricsCount struct {
	Key   string `json:"key"`
	Count int    `json:"count"`
}

type MetricsReport struct {
	Schema                    string         `json:"schema"`
	IndexVersion              string         `json:"index_version"`
	Total                     int            `json:"total"`
	Freshness                 []MetricsCount `json:"freshness"`
	SourceFamilies            []MetricsCount `json:"source_families"`
	Owners                    []MetricsCount `json:"owners"`
	UncoveredCandidateClasses []MetricsCount `json:"uncovered_candidate_classes"`
}

func Metrics(entries []Entry, coverage CoverageReport) MetricsReport {
	report := MetricsReport{Schema: MetricsSchemaVersion, IndexVersion: PublicIndexVersion, Total: len(entries)}
	freshness := map[string]int{"fresh": 0, "stale": 0, "unknown": 0, "invalid": 0}
	sources, owners, uncovered := map[string]int{}, map[string]int{}, map[string]int{}
	for _, entry := range entries {
		freshness[string(entry.Freshness.Verdict)]++
		owners[entry.Owner.Leaf+"@"+entry.Owner.Lane]++
		seen := map[string]bool{}
		for _, source := range entry.Sources {
			family := source.Probe
			if family == "" {
				family = string(source.Kind)
			}
			if !seen[family] {
				sources[family]++
				seen[family] = true
			}
		}
	}
	for _, finding := range coverage.Findings {
		uncovered[finding.Reason]++
	}
	report.Freshness = sortedMetrics(freshness)
	report.SourceFamilies = sortedMetrics(sources)
	report.Owners = sortedMetrics(owners)
	report.UncoveredCandidateClasses = sortedMetrics(uncovered)
	return report
}

func PublicMetrics() MetricsReport { return Metrics(publicEntries, CoverageReport{}) }

func sortedMetrics(counts map[string]int) []MetricsCount {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]MetricsCount, 0, len(keys))
	for _, key := range keys {
		out = append(out, MetricsCount{Key: key, Count: counts[key]})
	}
	return out
}

func SumMetrics(counts []MetricsCount) int {
	total := 0
	for _, count := range counts {
		total += count.Count
	}
	return total
}
