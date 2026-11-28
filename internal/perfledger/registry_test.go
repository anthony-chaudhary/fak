// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"strings"
	"testing"
)

func TestRegistryOrder(t *testing.T) {
	t.Run("valid RECORD/AGGREGATE/DERIVED graph resolves in topological order", func(t *testing.T) {
		reg, err := NewMetricRegistry(
			MetricSpec{
				Tag:             "e2e_interactivity",
				Kind:            MetricDerived,
				RequiredMetrics: []string{"request_latency", "output_sequence_length"},
				OptionalMetrics: []string{"absent_optional_metric"},
			},
			MetricSpec{
				Tag:             "inter_token_latency",
				Kind:            MetricRecord,
				RequiredMetrics: []string{"request_latency", "time_to_first_token"},
				OptionalMetrics: []string{"first_chunk_tokens"},
			},
			MetricSpec{
				Tag:             "total_osl",
				Kind:            MetricAggregate,
				RequiredMetrics: []string{"output_sequence_length"},
			},
			MetricSpec{
				Tag:             "goodput",
				Kind:            MetricDerived,
				RequiredMetrics: []string{"total_osl", "inter_token_latency"},
			},
			MetricSpec{Tag: "request_latency", Kind: MetricRecord},
			MetricSpec{Tag: "time_to_first_token", Kind: MetricRecord},
			MetricSpec{Tag: "output_sequence_length", Kind: MetricRecord},
			MetricSpec{Tag: "first_chunk_tokens", Kind: MetricRecord},
		)
		if err != nil {
			t.Fatalf("NewMetricRegistry: %v", err)
		}

		tags, err := reg.ComputeOrderTags()
		if err != nil {
			t.Fatalf("ComputeOrderTags: %v", err)
		}
		if len(tags) != 8 {
			t.Fatalf("ComputeOrderTags len = %d, want 8", len(tags))
		}
		pos := map[string]int{}
		for i, tag := range tags {
			pos[tag] = i
		}
		// Every required and present optional dependency must precede its dependent.
		edges := [][2]string{
			{"request_latency", "inter_token_latency"},
			{"time_to_first_token", "inter_token_latency"},
			{"first_chunk_tokens", "inter_token_latency"},
			{"output_sequence_length", "total_osl"},
			{"request_latency", "e2e_interactivity"},
			{"output_sequence_length", "e2e_interactivity"},
			{"total_osl", "goodput"},
			{"inter_token_latency", "goodput"},
		}
		for _, e := range edges {
			if pos[e[0]] >= pos[e[1]] {
				t.Errorf("dependency %q (pos %d) did not precede dependent %q (pos %d)", e[0], pos[e[0]], e[1], pos[e[1]])
			}
		}
	})

	t.Run("rejects dependency-shape violations across kinds", func(t *testing.T) {
		badShapes := []struct {
			name  string
			specs []MetricSpec
		}{
			{
				name: "RECORD cannot depend on AGGREGATE",
				specs: []MetricSpec{
					{Tag: "sum_tokens", Kind: MetricAggregate},
					{Tag: "bad_rec", Kind: MetricRecord, RequiredMetrics: []string{"sum_tokens"}},
				},
			},
			{
				name: "RECORD cannot depend on DERIVED via optional",
				specs: []MetricSpec{
					{Tag: "ratio", Kind: MetricDerived},
					{Tag: "bad_rec", Kind: MetricRecord, OptionalMetrics: []string{"ratio"}},
				},
			},
			{
				name: "AGGREGATE cannot depend on DERIVED",
				specs: []MetricSpec{
					{Tag: "ratio", Kind: MetricDerived},
					{Tag: "bad_agg", Kind: MetricAggregate, RequiredMetrics: []string{"ratio"}},
				},
			},
		}
		for _, tc := range badShapes {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := NewMetricRegistry(tc.specs...); err == nil {
					t.Fatalf("%s: expected startup validation error", tc.name)
				}
			})
		}
	})

	t.Run("rejects cycles, self-dependencies, duplicates, and missing required deps", func(t *testing.T) {
		if _, err := NewMetricRegistry(
			MetricSpec{Tag: "a", Kind: MetricDerived, RequiredMetrics: []string{"b"}},
			MetricSpec{Tag: "b", Kind: MetricDerived, RequiredMetrics: []string{"c"}},
			MetricSpec{Tag: "c", Kind: MetricDerived, RequiredMetrics: []string{"a"}},
		); err == nil || !strings.Contains(err.Error(), "cycle") {
			t.Fatalf("expected cycle error, got %v", err)
		}

		if _, err := NewMetricRegistry(
			MetricSpec{Tag: "self", Kind: MetricRecord, RequiredMetrics: []string{"self"}},
		); err == nil {
			t.Fatal("expected self-dependency error")
		}

		if _, err := NewMetricRegistry(
			MetricSpec{Tag: "dup", Kind: MetricRecord},
			MetricSpec{Tag: "dup", Kind: MetricRecord},
		); err == nil {
			t.Fatal("expected duplicate tag error")
		}

		if _, err := NewMetricRegistry(
			MetricSpec{Tag: "need", Kind: MetricDerived, RequiredMetrics: []string{"missing_dep"}},
		); err == nil {
			t.Fatal("expected missing required dependency error")
		}
	})
}
