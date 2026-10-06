package engine_test

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// Every recorder — not just one the gateway was handed — must fold into the
// process-global DefaultCacheEvents, so a restore miss/fault from any live adapter
// reaches /metrics by default. Deltas, because other tests in the binary also feed
// the global.
func TestDefaultCacheEventsFoldsEveryRecorder(t *testing.T) {
	before := engine.DefaultCacheEvents.Snapshot()

	a := engine.NewCacheEventRecorder()
	b := engine.NewCacheEventRecorder()
	a.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferMissed})
	b.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, ToTier: cachemeta.TierHBM, Outcome: cachemeta.KVTransferFault, FaultReason: "page-in EIO"})
	b.Record(engine.CacheEvent{Direction: cachemeta.KVOffload, ToTier: cachemeta.TierDRAM, BytesMoved: 4096, Tokens: 16, Outcome: cachemeta.KVTransferOK})

	after := engine.DefaultCacheEvents.Snapshot()
	if d := after.RestoreMiss - before.RestoreMiss; d != 1 {
		t.Errorf("global restore_miss delta = %d, want 1", d)
	}
	if d := after.RestoreFault - before.RestoreFault; d != 1 {
		t.Errorf("global restore_fault delta = %d, want 1", d)
	}
	if d := after.Events - before.Events; d != 3 {
		t.Errorf("global events delta = %d, want 3", d)
	}
	if d := after.BytesMoved - before.BytesMoved; d != 4096 {
		t.Errorf("global bytes_moved delta = %d, want 4096", d)
	}
	// Per-recorder surfaces stay independent of each other.
	if s := a.Metrics().Snapshot(); s.Events != 1 || s.RestoreMiss != 1 {
		t.Errorf("recorder a snapshot = %+v, want exactly its own restore miss", s)
	}
	if !after.Observed() {
		t.Error("DefaultCacheEvents must report observed once fed")
	}
	if !strings.Contains(after.Prometheus(), "fak_engine_cache_events_observed 1\n") {
		t.Errorf("fed stream must render observed=1:\n%s", after.Prometheus())
	}
}

// An unfed surface renders every family at zero with observed=0, so a dark sensor
// is visible on the board instead of reading as "no misses".
func TestCacheEventsUnfedRendersObservedZero(t *testing.T) {
	prom := engine.NewCacheEventMetrics().Snapshot().Prometheus()
	for _, want := range []string{
		"fak_engine_cache_events_observed 0\n",
		"fak_engine_cache_restore_miss_total 0\n",
		"fak_engine_cache_restore_fault_total 0\n",
		"fak_engine_cache_hits_total 0\n",
	} {
		if !strings.Contains(prom, want) {
			t.Errorf("unfed stream missing %q:\n%s", want, prom)
		}
	}
}
