package systembaseline

import (
	"testing"
	"time"
)

func TestBuildReportsAdaptiveCensusAndInvalidatesLimitedCoverage(t *testing.T) {
	base := time.Unix(100, 0)
	snap := func(at time.Duration, limited bool) Snapshot {
		return Snapshot{At: base.Add(at), CensusWallNS: int64(time.Millisecond), CensusStages: map[string]int64{"process_census": int64(time.Millisecond)}, EffectiveCadenceNS: int64(50 * time.Millisecond), StableCacheHits: 2, StableCacheMisses: 1, CoverageLimited: limited, ProcessEnumerationOK: true, Host: HostSample{CPUAvailable: true, TotalCPUNS: uint64(at), BusyCPUNS: uint64(at / 2)}, Processes: []ProcessSample{{PID: 7, StartID: 1, CPUAvailable: true, CPUNS: uint64(at)}}}
	}
	baseline := []Snapshot{snap(0, false), snap(time.Second, false)}
	command := []Snapshot{snap(2*time.Second, true), snap(3*time.Second, true)}
	r := Build(baseline, command, 7, 50*time.Millisecond, DefaultPolicy(), 0, false)
	if r.Verdict != VerdictInvalid {
		t.Fatalf("verdict=%s findings=%+v", r.Verdict, r.Findings)
	}
	if !r.CommandSampler.CoverageLimited || r.CommandSampler.EffectiveCadenceNS != int64(50*time.Millisecond) {
		t.Fatalf("sampler=%+v", r.CommandSampler)
	}
	if r.CommandSampler.Stages["process_census"] != int64(time.Millisecond) || r.CommandSampler.CacheHits != 2 || r.CommandSampler.CacheMisses != 1 {
		t.Fatalf("stage/cache=%+v", r.CommandSampler)
	}
}
