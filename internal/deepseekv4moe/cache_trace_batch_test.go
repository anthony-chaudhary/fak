package deepseekv4moe

import (
	"errors"
	"reflect"
	"testing"
)

// batchFromRoutes lifts a single-stream route list into the equivalent
// one-agent batch step list.
func batchFromRoutes(routes []ExpertRoute) []BatchStep {
	steps := make([]BatchStep, 0, len(routes))
	for _, route := range routes {
		steps = append(steps, BatchStep{Layer: route.Layer, PerAgent: [][]int{route.Experts}})
	}
	return steps
}

func TestSimulateExpertCacheBatchSingleAgentReducesToSingleStream(t *testing.T) {
	coldWarm := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
	}
	tests := []struct {
		name     string
		routes   []ExpertRoute
		capacity int
		layers   int
		experts  int
		topK     int
	}{
		{"cold", coldWarm, 12, 61, 384, 6},
		{"warm replay", append(append([]ExpertRoute{}, coldWarm...), coldWarm...), 12, 61, 384, 6},
		{"eviction churn", []ExpertRoute{
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 1, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		}, 6, 61, 384, 6},
		{"capacity one", []ExpertRoute{
			{Layer: 0, Experts: []int{1}},
			{Layer: 0, Experts: []int{1}},
			{Layer: 0, Experts: []int{2}},
		}, 1, 61, 384, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			single, err := SimulateExpertCache(tc.routes, tc.capacity, tc.layers, tc.experts, tc.topK)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := SimulateExpertCacheBatch(batchFromRoutes(tc.routes), tc.capacity, tc.layers, tc.experts, tc.topK, 1)
			if err != nil {
				t.Fatal(err)
			}
			if batch.Hits != single.Hits || batch.Misses != single.PageIns || batch.PeakResident != single.PeakResident {
				t.Fatalf("single-agent batch diverges:\nbatch  %+v\nsingle %+v", batch, single)
			}
			if batch.DistinctStreamed != batch.Hits+batch.Misses || batch.NaiveStreamed != batch.DistinctStreamed {
				t.Fatalf("single-agent stream accounting = %+v", batch)
			}
			if batch.CoalesceRatio != 1 {
				t.Fatalf("single-agent coalesce ratio = %v, want 1", batch.CoalesceRatio)
			}
		})
	}
}

func TestSimulateExpertCacheBatchFullOverlap(t *testing.T) {
	const agents = 4
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
	}
	steps := make([]BatchStep, 0, len(routes))
	for _, route := range routes {
		perAgent := make([][]int, agents)
		for b := range perAgent {
			perAgent[b] = route.Experts
		}
		steps = append(steps, BatchStep{Layer: route.Layer, PerAgent: perAgent})
	}

	single, err := SimulateExpertCache(routes, 12, 61, 384, 6)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := SimulateExpertCacheBatch(steps, 12, 61, 384, 6, agents)
	if err != nil {
		t.Fatal(err)
	}
	if batch.DistinctStreamed != single.PageIns+single.Hits {
		t.Fatalf("full-overlap DistinctStreamed = %d, want single-agent %d", batch.DistinctStreamed, single.PageIns+single.Hits)
	}
	if batch.Misses != single.PageIns || batch.Hits != single.Hits || batch.PeakResident != single.PeakResident {
		t.Fatalf("full-overlap cache behavior diverges:\nbatch  %+v\nsingle %+v", batch, single)
	}
	if batch.NaiveStreamed != int64(agents)*batch.DistinctStreamed || batch.CoalesceRatio != agents {
		t.Fatalf("full-overlap ratio = %+v, want CoalesceRatio %d", batch, agents)
	}
	if batch.PeakStepUnion != 6 {
		t.Fatalf("full-overlap PeakStepUnion = %d, want 6", batch.PeakStepUnion)
	}
}

func TestSimulateExpertCacheBatchDisjoint(t *testing.T) {
	steps := []BatchStep{
		{Layer: 0, PerAgent: [][]int{{0, 1}, {2, 3}, {4, 5}}},
		{Layer: 1, PerAgent: [][]int{{10, 11}, {12, 13}, {14, 15}}},
	}
	batch, err := SimulateExpertCacheBatch(steps, 12, 61, 384, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if batch.CoalesceRatio != 1 || batch.DistinctStreamed != batch.NaiveStreamed {
		t.Fatalf("disjoint batch coalesced: %+v", batch)
	}
	if batch.DistinctStreamed != 12 || batch.Misses != 12 || batch.Hits != 0 || batch.PeakResident != 12 || batch.PeakStepUnion != 6 {
		t.Fatalf("disjoint trace = %+v", batch)
	}
}

func TestSimulateExpertCacheBatchPartialOverlapExactRatio(t *testing.T) {
	steps := []BatchStep{
		{Layer: 0, PerAgent: [][]int{{0, 1}, {1, 2}, {2, 3}, {0, 3}}}, // union {0,1,2,3}
		{Layer: 1, PerAgent: [][]int{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}, // union {0,1}
	}
	batch, err := SimulateExpertCacheBatch(steps, 12, 61, 384, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	if batch.NaiveStreamed != 16 || batch.DistinctStreamed != 6 || batch.PeakStepUnion != 4 {
		t.Fatalf("partial-overlap stream accounting = %+v", batch)
	}
	if want := float64(16) / float64(6); batch.CoalesceRatio != want {
		t.Fatalf("partial-overlap CoalesceRatio = %v, want exactly %v", batch.CoalesceRatio, want)
	}
	if batch.Misses != 6 || batch.Hits != 0 || batch.PeakResident != 6 {
		t.Fatalf("partial-overlap trace = %+v", batch)
	}
}

func TestSimulateExpertCacheBatchWarmHitsOverDistinctGroups(t *testing.T) {
	step := BatchStep{Layer: 0, PerAgent: [][]int{{0, 1}, {1, 2}}}
	batch, err := SimulateExpertCacheBatch([]BatchStep{step, step}, 3, 61, 384, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if batch.Misses != 3 || batch.Hits != 3 || batch.PeakResident != 3 || batch.PeakStepUnion != 3 {
		t.Fatalf("warm batch trace = %+v", batch)
	}
	if batch.DistinctStreamed != 6 || batch.NaiveStreamed != 8 || batch.CoalesceRatio != float64(8)/float64(6) {
		t.Fatalf("warm batch stream accounting = %+v", batch)
	}
}

func TestSimulateExpertCacheBatchDeterministicReplay(t *testing.T) {
	steps := []BatchStep{
		{Layer: 0, PerAgent: [][]int{{0, 1}, {1, 2}, {3, 4}}},
		{Layer: 1, PerAgent: [][]int{{0, 5}, {5, 6}, {0, 6}}},
		{Layer: 0, PerAgent: [][]int{{0, 1}, {1, 2}, {3, 4}}},
	}
	first, err := SimulateExpertCacheBatch(steps, 5, 61, 384, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SimulateExpertCacheBatch(steps, 5, 61, 384, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("replay mismatch:\nfirst  %+v\nsecond %+v", first, second)
	}
}

func TestSimulateExpertCacheBatchFailsClosed(t *testing.T) {
	tests := []struct {
		name     string
		steps    []BatchStep
		capacity int
		layers   int
		experts  int
		topK     int
		agents   int
		want     error
	}{
		{"zero capacity", nil, 0, 61, 384, 6, 2, ErrInvalidTraceCapacity},
		{"bad shape", nil, 1, 0, 384, 6, 2, ErrInvalidTraceShape},
		{"zero agents", nil, 1, 61, 384, 6, 0, ErrInvalidTraceShape},
		{"wrong agent width", []BatchStep{{Layer: 0, PerAgent: [][]int{{0}}}}, 1, 61, 384, 1, 2, ErrInvalidTraceRoute},
		{"wrong top-k width", []BatchStep{{Layer: 0, PerAgent: [][]int{{0}, {0, 1}}}}, 2, 61, 384, 2, 2, ErrInvalidTraceRoute},
		{"layer out of range", []BatchStep{{Layer: 61, PerAgent: [][]int{{0}}}}, 1, 61, 384, 1, 1, ErrInvalidTraceRoute},
		{"expert out of range", []BatchStep{{Layer: 0, PerAgent: [][]int{{384}}}}, 1, 61, 384, 1, 1, ErrInvalidTraceRoute},
		{"duplicate within one agent", []BatchStep{{Layer: 0, PerAgent: [][]int{{1, 1}, {2, 3}}}}, 4, 61, 384, 2, 2, ErrDuplicateTraceExpert},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := SimulateExpertCacheBatch(tc.steps, tc.capacity, tc.layers, tc.experts, tc.topK, tc.agents)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestAdmitExpertCacheBatchTraceStepUnionBoundary(t *testing.T) {
	// totalHBM 60 / groupBytes 10 -> capacity 6 routed groups.
	fits := []BatchStep{{Layer: 0, PerAgent: [][]int{{0, 1}, {2, 3}, {4, 5}}}}
	got, err := AdmitExpertCacheBatchTrace(60, 0, 0, 10, 1, 384, 2, 3, fits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Plan.ResidentRoutedGroups != 6 || got.Trace.PeakStepUnion != 6 || got.Trace.PeakResident != 6 {
		t.Fatalf("exact union admission = %+v", got)
	}

	// A fourth disjoint agent unions one step to 8 > 6: fail closed.
	exceeds := []BatchStep{{Layer: 0, PerAgent: [][]int{{0, 1}, {2, 3}, {4, 5}, {6, 7}}}}
	_, err = AdmitExpertCacheBatchTrace(60, 0, 0, 10, 1, 384, 2, 4, exceeds)
	if !errors.Is(err, ErrRouteWorkingSetExceedsCache) {
		t.Fatalf("union-exceeds error = %v, want %v", err, ErrRouteWorkingSetExceedsCache)
	}

	// Capacity below topK cannot hold even a fully-overlapped step: static fail.
	_, err = AdmitExpertCacheBatchTrace(19, 0, 0, 10, 1, 384, 2, 4, nil)
	if !errors.Is(err, ErrRouteWorkingSetExceedsCache) {
		t.Fatalf("static top-k error = %v, want %v", err, ErrRouteWorkingSetExceedsCache)
	}
}

func TestAdmitExpertCacheBatchTracePropagatesFailures(t *testing.T) {
	_, err := AdmitExpertCacheBatchTrace(10, 10, 0, 1, 61, 384, 6, 2, nil)
	if !errors.Is(err, ErrInsufficientCacheBudget) {
		t.Fatalf("plan error = %v, want %v", err, ErrInsufficientCacheBudget)
	}
	bad := []BatchStep{{Layer: 0, PerAgent: [][]int{{1, 1}, {2, 3}}}}
	_, err = AdmitExpertCacheBatchTrace(600, 0, 0, 10, 1, 384, 2, 2, bad)
	if !errors.Is(err, ErrDuplicateTraceExpert) {
		t.Fatalf("trace error = %v, want %v", err, ErrDuplicateTraceExpert)
	}
}
