package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=5ms lane=default
func TestV41CohortExpertUnionMetricsRender(t *testing.T) {
	var silent strings.Builder
	writeV41CohortExpertUnionMetrics(&silent, model.V41CohortExpertUnionCounters{})
	if silent.Len() != 0 {
		t.Fatalf("never-engaged cohort union rendered %q, want nothing", silent.String())
	}

	snap := model.V41CohortExpertUnionCounters{Selected: 7, DeclinedOverlap: 3, DeclinedCapacity: 1, DeclinedAdmission: 2}
	snap.SharedRows[0] = 2
	snap.SharedRows[6] = 5
	var b strings.Builder
	writeV41CohortExpertUnionMetrics(&b, snap)
	out := b.String()
	for _, want := range []string{
		"# TYPE fak_v41_cohort_expert_union_total counter",
		`fak_v41_cohort_expert_union_total{outcome="selected"} 7`,
		`fak_v41_cohort_expert_union_total{outcome="declined_overlap"} 3`,
		`fak_v41_cohort_expert_union_total{outcome="declined_capacity"} 1`,
		`fak_v41_cohort_expert_union_total{outcome="declined_admission"} 2`,
		"# TYPE fak_v41_cohort_expert_union_shared_rows_total counter",
		`fak_v41_cohort_expert_union_shared_rows_total{shared="0"} 2`,
		`fak_v41_cohort_expert_union_shared_rows_total{shared="6"} 5`,
		`fak_v41_cohort_expert_union_shared_rows_total{shared="1"} 0`,
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("render missing %q in:\n%s", want, out)
		}
	}
	if got := strings.Count(out, "fak_v41_cohort_expert_union_shared_rows_total{shared="); got != len(snap.SharedRows) {
		t.Fatalf("shared-rows buckets rendered = %d, want %d", got, len(snap.SharedRows))
	}
}
