package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestFormatCompactionSummaryExtractedSection(t *testing.T) {
	tests := []struct {
		name    string
		summary gateway.AdjudicationSummary
		want    []string
	}{
		{
			name: "disabled",
			summary: gateway.AdjudicationSummary{
				CompactionOff: 1,
			},
			want: []string{"compaction", "DISABLED (budget 0; body forwarded byte-for-byte)", "0 fired, 0 bailed, 1 off"},
		},
		{
			name: "fault bail remains prominent",
			summary: gateway.AdjudicationSummary{
				CompactionBudget:      4096,
				CompactionBailed:      1,
				CompactionBailReasons: map[string]uint64{"prefix_mismatch": 1},
			},
			want: []string{"ENABLED but idle, budget 4096 tok", "bailed: prefix_mismatch", "fak-fault"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := formatCompactionSummary(test.summary)
			for _, want := range test.want {
				if !strings.Contains(got, want) {
					t.Fatalf("formatCompactionSummary() = %q, want substring %q", got, want)
				}
			}
		})
	}
}
