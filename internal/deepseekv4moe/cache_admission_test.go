package deepseekv4moe

import (
	"errors"
	"reflect"
	"testing"
)

func TestAdmitExpertCacheTracePinnedV4(t *testing.T) {
	const (
		totalHBM       = int64(480 * 1024 * 1024 * 1024)
		nonRouted      = int64(42_650_568_824)
		runtimeReserve = int64(64 * 1024 * 1024 * 1024)
		groupBytes     = int64(35_094_528)
	)
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
	}
	got, err := AdmitExpertCacheTrace(totalHBM, nonRouted, runtimeReserve, groupBytes, 61, 384, 6, routes)
	if err != nil {
		t.Fatal(err)
	}
	if got.Plan.ResidentRoutedGroups != 11_512 || got.Plan.UsedBytes > totalHBM || got.Trace.PeakResident > int(got.Plan.ResidentRoutedGroups) {
		t.Fatalf("invalid pinned admission: %+v", got)
	}
	if got.Trace.PageIns != 12 || got.Trace.Hits != 6 || got.Trace.Evictions != 0 {
		t.Fatalf("pinned trace = %+v", got.Trace)
	}
}

func TestAdmitExpertCacheTraceTopKBoundary(t *testing.T) {
	route := []ExpertRoute{{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}}}
	exact, err := AdmitExpertCacheTrace(60, 0, 0, 10, 1, 384, 6, route)
	if err != nil {
		t.Fatal(err)
	}
	if exact.Plan.ResidentRoutedGroups != 6 || exact.Trace.PeakResident != 6 {
		t.Fatalf("exact top-k admission = %+v", exact)
	}
	_, err = AdmitExpertCacheTrace(59, 0, 0, 10, 1, 384, 6, route)
	if !errors.Is(err, ErrRouteWorkingSetExceedsCache) {
		t.Fatalf("top-k-minus-one error = %v, want %v", err, ErrRouteWorkingSetExceedsCache)
	}
}

func TestAdmitExpertCacheTracePropagatesPlanFailure(t *testing.T) {
	_, err := AdmitExpertCacheTrace(10, 10, 0, 1, 61, 384, 6, nil)
	if !errors.Is(err, ErrInsufficientCacheBudget) {
		t.Fatalf("error = %v, want %v", err, ErrInsufficientCacheBudget)
	}
}

func TestAdmitExpertCacheTraceDeterministicWarmReplay(t *testing.T) {
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1}},
		{Layer: 0, Experts: []int{0, 1}},
	}
	first, err := AdmitExpertCacheTrace(20, 0, 0, 10, 1, 384, 2, routes)
	if err != nil {
		t.Fatal(err)
	}
	second, err := AdmitExpertCacheTrace(20, 0, 0, 10, 1, 384, 2, routes)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || first.Trace.PageIns != 2 || first.Trace.Hits != 2 {
		t.Fatalf("non-deterministic warm replay: first=%+v second=%+v", first, second)
	}
}
