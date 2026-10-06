package gateway

import (
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func scrapeCounter(t *testing.T, text, name string) uint64 {
	t.Helper()
	for _, line := range strings.Split(text, "\n") {
		if v, ok := strings.CutPrefix(line, name+" "); ok {
			n, err := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
			if err != nil {
				t.Fatalf("parse %s=%q: %v", name, v, err)
			}
			return n
		}
	}
	t.Fatalf("/metrics is missing %s", name)
	return 0
}

// The live-engine KV cache-event stream must reach a real server's /metrics by
// default: a restore MISS / FAULT recorded by any engine.CacheEventRecorder (not one
// threaded to the gateway) moves fak_engine_cache_restore_{miss,fault}_total, and
// the observed gauge is always present.
func TestMetricsRenderEngineCacheEventStream(t *testing.T) {
	srv := newTestServer(t)
	pre := srv.renderMetrics()
	if !strings.Contains(pre, "# TYPE fak_engine_cache_events_observed gauge") {
		t.Fatalf("fak_engine_cache_events_observed missing from /metrics")
	}
	miss0 := scrapeCounter(t, pre, "fak_engine_cache_restore_miss_total")
	fault0 := scrapeCounter(t, pre, "fak_engine_cache_restore_fault_total")

	rec := engine.NewCacheEventRecorder()
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferMissed})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, ToTier: cachemeta.TierHBM, Outcome: cachemeta.KVTransferFault, FaultReason: "page-in EIO"})

	post := srv.renderMetrics()
	if d := scrapeCounter(t, post, "fak_engine_cache_restore_miss_total") - miss0; d != 1 {
		t.Errorf("restore_miss delta on /metrics = %d, want 1", d)
	}
	if d := scrapeCounter(t, post, "fak_engine_cache_restore_fault_total") - fault0; d != 1 {
		t.Errorf("restore_fault delta on /metrics = %d, want 1", d)
	}
	if scrapeCounter(t, post, "fak_engine_cache_events_observed") != 1 {
		t.Error("observed gauge must be 1 once the stream is fed")
	}
}
