package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelWeightResidencyAbsentWithoutModel(t *testing.T) {
	var nilPlanner *InKernelPlanner
	if _, ok := nilPlanner.WeightResidency(); ok {
		t.Fatal("nil planner must not report resident weights")
	}
	if _, ok := (&InKernelPlanner{}).WeightResidency(); ok {
		t.Fatal("planner without a model must not report resident weights")
	}
	var _ WeightResidencyReporter = (*InKernelPlanner)(nil)
}

func TestWeightResidencyFromReportCopiesStores(t *testing.T) {
	r := &model.ResidentReport{
		Q8Bytes: 8, Q4KBytes: 4, KQuantBytes: 3, Q2Bytes: 2, F32Bytes: 32,
		Q6KEmbedBytes: 6, Q4KEmbedBytes: 5, PQ2EmbedBytes: 1, Q2KEmbedBytes: 7,
		TiedEmbedF32Bytes: 9, TiedHeadQ8Bytes: 10, TotalResidentBytes: 68,
		DecodeBytesPerToken: 23, LMHead: "cpu-q6k",
	}
	got := WeightResidencyFromReport(r)
	want := WeightResidency{
		TotalResidentBytes: 68, F32Bytes: 32, Q8Bytes: 8, Q4KBytes: 4, KQuantBytes: 3, Q2Bytes: 2,
		Q2KEmbedBytes: 7, PQ2EmbedBytes: 1, Q4KEmbedBytes: 5, Q6KEmbedBytes: 6,
		TiedEmbedF32Bytes: 9, TiedHeadQ8Bytes: 10, DecodeBytesPerToken: 23, LMHead: "cpu-q6k",
	}
	if got != want {
		t.Fatalf("WeightResidencyFromReport = %+v, want %+v", got, want)
	}
}
