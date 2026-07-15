package deepseekv4moe

import (
	"errors"
	"reflect"
	"testing"
)

func TestSimulateExpertCacheV4ColdAndWarm(t *testing.T) {
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
	}
	cold, err := SimulateExpertCache(routes, 12, 61, 384, 6)
	if err != nil {
		t.Fatal(err)
	}
	if cold.PageIns != 12 || cold.Hits != 0 || cold.Evictions != 0 || cold.PeakResident != 12 || len(cold.FinalResident) != 12 {
		t.Fatalf("cold trace = %+v", cold)
	}

	warmRoutes := append(append([]ExpertRoute{}, routes...), routes...)
	warm, err := SimulateExpertCache(warmRoutes, 12, 61, 384, 6)
	if err != nil {
		t.Fatal(err)
	}
	if warm.PageIns != 12 || warm.Hits != 12 || warm.Evictions != 0 || warm.PeakResident != 12 {
		t.Fatalf("warm trace = %+v", warm)
	}
}

func TestSimulateExpertCacheEvictionAndReplay(t *testing.T) {
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 1, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
	}
	first, err := SimulateExpertCache(routes, 6, 61, 384, 6)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SimulateExpertCache(routes, 6, 61, 384, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("replay mismatch:\nfirst  %+v\nsecond %+v", first, second)
	}
	if first.PageIns != 18 || first.Hits != 0 || first.Evictions != 12 || first.PeakResident != 6 || len(first.FinalResident) != 6 {
		t.Fatalf("eviction trace = %+v", first)
	}
	for _, group := range first.FinalResident {
		if group.Layer != 0 {
			t.Fatalf("final resident contains stale group %+v", group)
		}
	}
}

func TestSimulateExpertCacheCapacityOne(t *testing.T) {
	trace, err := SimulateExpertCache([]ExpertRoute{{Layer: 0, Experts: []int{1}}, {Layer: 0, Experts: []int{1}}, {Layer: 0, Experts: []int{2}}}, 1, 61, 384, 1)
	if err != nil {
		t.Fatal(err)
	}
	if trace.PageIns != 2 || trace.Hits != 1 || trace.Evictions != 1 || trace.PeakResident != 1 || !reflect.DeepEqual(trace.FinalResident, []ExpertGroup{{Layer: 0, Expert: 2}}) {
		t.Fatalf("capacity-one trace = %+v", trace)
	}
}

func TestSimulateExpertCacheFailsClosed(t *testing.T) {
	tests := []struct {
		name     string
		routes   []ExpertRoute
		capacity int
		layers   int
		experts  int
		topK     int
		want     error
	}{
		{"zero capacity", nil, 0, 61, 384, 6, ErrInvalidTraceCapacity},
		{"bad shape", nil, 1, 0, 384, 6, ErrInvalidTraceShape},
		{"wrong top-k width", []ExpertRoute{{Layer: 0, Experts: []int{0}}}, 1, 61, 384, 6, ErrInvalidTraceRoute},
		{"layer out of range", []ExpertRoute{{Layer: 61, Experts: []int{0}}}, 1, 61, 384, 1, ErrInvalidTraceRoute},
		{"expert out of range", []ExpertRoute{{Layer: 0, Experts: []int{384}}}, 1, 61, 384, 1, ErrInvalidTraceRoute},
		{"duplicate expert", []ExpertRoute{{Layer: 0, Experts: []int{1, 1}}}, 2, 61, 384, 2, ErrDuplicateTraceExpert},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := SimulateExpertCache(tc.routes, tc.capacity, tc.layers, tc.experts, tc.topK)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}
