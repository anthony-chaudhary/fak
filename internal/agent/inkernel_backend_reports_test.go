package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type backendReportsProbe struct {
	compute.Backend
	calls [6]int
	zero  bool
	text  string
}

func (*backendReportsProbe) Name() string { return "fixture-backend" }

type backendReportsArena struct{ p *backendReportsProbe }

func (r backendReportsArena) WeightArenaCounters() (uint64, uint64, uint64, uint64, uint64) {
	r.p.calls[0]++
	if r.p.zero {
		return 0, 0, 0, 0, 0
	}
	return 2, 3, 5, 7, 11
}

type backendReportsPaged struct{ p *backendReportsProbe }

func (r backendReportsPaged) OffloadWeightCounters() (uint64, uint64) {
	r.p.calls[1]++
	if r.p.zero {
		return 0, 0
	}
	return 13, 17
}

type backendReportsRelease struct{ p *backendReportsProbe }

func (r backendReportsRelease) RetainedReleaseSummary() string {
	r.p.calls[2]++
	return r.p.text
}

type backendReportsLegacyArena struct{ p *backendReportsProbe }

func (r backendReportsLegacyArena) VulkanWeightArenaCounters() (uint64, uint64, uint64, uint64, uint64) {
	r.p.calls[3]++
	return 19, 23, 29, 31, 37
}

type backendReportsLegacyPaged struct{ p *backendReportsProbe }

func (r backendReportsLegacyPaged) VulkanOffloadWeightCounters() (uint64, uint64) {
	r.p.calls[4]++
	return 41, 43
}

type backendReportsLegacyRelease struct{ p *backendReportsProbe }

func (r backendReportsLegacyRelease) VulkanRetainedReleaseSummary() string {
	r.p.calls[5]++
	return "legacy:" + r.p.text
}

// fak-test:runtime fast est=5ms lane=default
func TestBackendReportsOptionalReaders(t *testing.T) {
	neutral := DeviceWeightMemory{Backend: "fixture-backend", Arena: true, ArenaMemoryAllocations: 2, ArenaBufferBindings: 3, ArenaReservedBytes: 5, ArenaLiveBytes: 7, ArenaPeakReservedBytes: 11, PagedWeights: true, PagedWeightsReused: 13, PagedWeightsFreshly: 17}
	legacy := DeviceWeightMemory{Backend: "fixture-backend", Arena: true, ArenaMemoryAllocations: 19, ArenaBufferBindings: 23, ArenaReservedBytes: 29, ArenaLiveBytes: 31, ArenaPeakReservedBytes: 37, PagedWeights: true, PagedWeightsReused: 41, PagedWeightsFreshly: 43}
	for _, tc := range []struct {
		name        string
		makeBackend func(*backendReportsProbe) compute.Backend
		want        DeviceWeightMemory
		calls       [6]int
		zero        bool
	}{
		{"neutral", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsArena
				backendReportsPaged
				backendReportsRelease
			}{p, backendReportsArena{p}, backendReportsPaged{p}, backendReportsRelease{p}}
		}, neutral, [6]int{1, 1, 1, 0, 0, 0}, false},
		{"legacy", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsLegacyArena
				backendReportsLegacyPaged
				backendReportsLegacyRelease
			}{p, backendReportsLegacyArena{p}, backendReportsLegacyPaged{p}, backendReportsLegacyRelease{p}}
		}, legacy, [6]int{0, 0, 0, 1, 1, 1}, false},
		{"both", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsArena
				backendReportsPaged
				backendReportsRelease
				backendReportsLegacyArena
				backendReportsLegacyPaged
				backendReportsLegacyRelease
			}{p, backendReportsArena{p}, backendReportsPaged{p}, backendReportsRelease{p}, backendReportsLegacyArena{p}, backendReportsLegacyPaged{p}, backendReportsLegacyRelease{p}}
		}, neutral, [6]int{1, 1, 1, 0, 0, 0}, false},
		{"neither", func(p *backendReportsProbe) compute.Backend { return p }, DeviceWeightMemory{Backend: "fixture-backend"}, [6]int{}, false},
		{"nil backend", func(*backendReportsProbe) compute.Backend { return nil }, DeviceWeightMemory{}, [6]int{}, false},
		{"arena only", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsArena
			}{p, backendReportsArena{p}}
		}, DeviceWeightMemory{Backend: "fixture-backend", Arena: true, ArenaMemoryAllocations: 2, ArenaBufferBindings: 3, ArenaReservedBytes: 5, ArenaLiveBytes: 7, ArenaPeakReservedBytes: 11}, [6]int{1, 0, 0, 0, 0, 0}, false},
		{"paged only", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsPaged
			}{p, backendReportsPaged{p}}
		}, DeviceWeightMemory{Backend: "fixture-backend", PagedWeights: true, PagedWeightsReused: 13, PagedWeightsFreshly: 17}, [6]int{0, 1, 0, 0, 0, 0}, false},
		{"mixed", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsArena
				backendReportsLegacyPaged
			}{p, backendReportsArena{p}, backendReportsLegacyPaged{p}}
		}, DeviceWeightMemory{Backend: "fixture-backend", Arena: true, ArenaMemoryAllocations: 2, ArenaBufferBindings: 3, ArenaReservedBytes: 5, ArenaLiveBytes: 7, ArenaPeakReservedBytes: 11, PagedWeights: true, PagedWeightsReused: 41, PagedWeightsFreshly: 43}, [6]int{1, 0, 0, 0, 1, 0}, false},
		{"supported zero", func(p *backendReportsProbe) compute.Backend {
			return struct {
				*backendReportsProbe
				backendReportsArena
				backendReportsPaged
				backendReportsLegacyArena
				backendReportsLegacyPaged
			}{p, backendReportsArena{p}, backendReportsPaged{p}, backendReportsLegacyArena{p}, backendReportsLegacyPaged{p}}
		}, DeviceWeightMemory{Backend: "fixture-backend", Arena: true, PagedWeights: true}, [6]int{1, 1, 0, 0, 0, 0}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, text := range []string{"status=unavailable", "status=invalid", "status=stale-generation", ""} {
				p := &backendReportsProbe{zero: tc.zero, text: text}
				be := tc.makeBackend(p)
				planner := &InKernelPlanner{backend: be}
				if p.calls != ([6]int{}) {
					t.Fatal("eager report query")
				}
				got, ok := planner.DeviceWeightMemory()
				if got != tc.want || ok != (tc.want.Arena || tc.want.PagedWeights) {
					t.Fatalf("memory = %+v, %t; want %+v", got, ok, tc.want)
				}
				gotText, supported := planner.RetainedReleaseSummary()
				wantText := ""
				wantSupported := tc.calls[2] != 0 || tc.calls[5] != 0
				if tc.calls[2] != 0 {
					wantText = text
				} else if tc.calls[5] != 0 {
					wantText = "legacy:" + text
				}
				if gotText != wantText || supported != wantSupported || p.calls != tc.calls {
					t.Fatalf("text = %q, support = %t, calls = %v; want %q, %t, %v", gotText, supported, p.calls, wantText, wantSupported, tc.calls)
				}
			}
		})
	}
	var nilPlanner *InKernelPlanner
	if got, ok := nilPlanner.DeviceWeightMemory(); ok || got != (DeviceWeightMemory{}) {
		t.Fatalf("nil planner memory = %+v, %t", got, ok)
	}
	if text, ok := nilPlanner.RetainedReleaseSummary(); ok || text != "" {
		t.Fatalf("nil planner release = %q, %t", text, ok)
	}
}
