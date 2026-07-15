package model

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4ExpertStatsDoesNotInitializeRuntime(t *testing.T) {
	var nilSession *Session
	if got := nilSession.V4ExpertStats(); got != (V4ExpertStats{}) {
		t.Fatalf("nil stats=%+v", got)
	}
	m := &Model{Cfg: pinnedV4RuntimeConfig()}
	s := m.NewSession()
	if got := s.V4ExpertStats(); got != (V4ExpertStats{}) {
		t.Fatalf("uninitialized stats=%+v", got)
	}
	if s.v4Expert != nil {
		t.Fatal("stats initialized the live expert runtime")
	}
}

func TestV4ExpertStatsCapturedLiveRuntimeCacheWitness(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, _ := writeV4RuntimeFixture(t)
	be := compute.Default()
	runtime, err := newV4LiveExpert(dir, pinnedV4RuntimeConfig(), be)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	s := &Session{v4Expert: runtime}

	xHost := make([]float32, 32)
	for i := range xHost {
		xHost[i] = float32((i%5)-2) / 2
	}
	logits := make([]float32, 384)
	for i := range logits {
		logits[i] = -10
	}
	for i := 0; i < 6; i++ {
		logits[i] = float32(6 - i)
	}
	if _, err := runtime.forward(3, 0, xHost, logits, nil); err != nil {
		t.Fatal(err)
	}
	cold := s.V4ExpertStats()
	if cold.SourceReads == 0 || cold.SourceBytes == 0 || cold.Misses == 0 || cold.PageIns != cold.Misses {
		t.Fatalf("cold counters do not capture source misses: %+v", cold)
	}
	if cold.ResidentBytes <= 0 || cold.PeakResident <= 0 || cold.ResidentBytes > cold.RingBudget || cold.PeakResident > cold.RingBudget {
		t.Fatalf("cold counters do not prove ring bound: %+v", cold)
	}

	if _, err := runtime.forward(3, 0, xHost, logits, nil); err != nil {
		t.Fatal(err)
	}
	warm := s.V4ExpertStats()
	if warm.Hits <= cold.Hits || warm.SourceReads != cold.SourceReads || warm.SourceBytes != cold.SourceBytes || warm.Misses != cold.Misses {
		t.Fatalf("second identical access was not a source-free cache hit: cold=%+v warm=%+v", cold, warm)
	}
	if warm.CacheAccesses() != warm.Hits+warm.Misses || warm.HitRate() <= 0 || warm.HitRate() > 1 {
		t.Fatalf("cache denominator/rate invalid: %+v rate=%v", warm, warm.HitRate())
	}
	captured, err := json.Marshal(warm)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"source_reads"`, `"source_bytes"`, `"hits"`, `"misses"`, `"peak_resident_bytes"`, `"ring_budget_bytes"`} {
		if !json.Valid(captured) || !containsBytes(captured, []byte(field)) {
			t.Fatalf("captured JSON %s missing %s", captured, field)
		}
	}
}

func containsBytes(haystack, needle []byte) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		match := true
		for j := range needle {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
