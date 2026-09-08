package metrics

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

func (c *Collector) writeStartup(b *strings.Builder) {
	b.WriteString("\n\n# ---- startup progress --------------------------------------------\n\n")

	b.WriteString("# HELP l3_startup_shards_ready Shards that have completed initialization.\n")
	b.WriteString("# TYPE l3_startup_shards_ready gauge\n")
	fmt.Fprintf(b, "l3_startup_shards_ready %d\n", c.startup.ShardsReady.Load())

	b.WriteString("\n# HELP l3_startup_shards_total Total shards to initialize.\n")
	b.WriteString("# TYPE l3_startup_shards_total gauge\n")
	fmt.Fprintf(b, "l3_startup_shards_total %d\n", c.startup.ShardsTotal.Load())

	b.WriteString("\n# HELP l3_startup_mem_reserved_bytes Memory reserved during startup in bytes.\n")
	b.WriteString("# TYPE l3_startup_mem_reserved_bytes gauge\n")
	fmt.Fprintf(b, "l3_startup_mem_reserved_bytes %d\n", c.startup.MemReservedBytes.Load())

	b.WriteString("\n# HELP l3_startup_mem_total_bytes Total memory target in bytes.\n")
	b.WriteString("# TYPE l3_startup_mem_total_bytes gauge\n")
	fmt.Fprintf(b, "l3_startup_mem_total_bytes %d\n", c.startup.MemTotalBytes.Load())
}

// slabReady returns true when all shards have finished slab detection (no shard
// is in "warming_up" status). Returns true if ShardDetectionFunc is not wired up.
func (c *Collector) slabReady() (ready bool, pending int, total int) {
	if c.ShardDetectionFunc == nil {
		return true, 0, 0
	}
	statuses := c.ShardDetectionFunc()
	total = len(statuses)
	for _, s := range statuses {
		if s == "warming_up" {
			pending++
		}
	}
	return pending == 0, pending, total
}

// ReadyHandler returns 200 with {"status":"ready"} when the server is ready,
// or 503 with JSON progress details otherwise. HTTP status codes are unchanged
// for backward compatibility with k8s probes.
//
// When ?slab=true is present, also checks that all shards have completed slab
// detection (no shard in "warming_up" status). Without ?slab=true, behavior is
// identical to previous versions.
func (c *Collector) ReadyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if c.startup.Ready.Load() {
		// Base server is ready; optionally check slab detection status.
		if r.URL.Query().Get("slab") == "true" {
			if ok, pending, total := c.slabReady(); !ok {
				w.WriteHeader(http.StatusServiceUnavailable)
				json.NewEncoder(w).Encode(map[string]interface{}{
					"status":         "slab_not_ready",
					"shards_pending": pending,
					"shards_total":   total,
				})
				return
			}
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ready"}` + "\n"))
		return
	}

	shardsReady := c.startup.ShardsReady.Load()
	shardsTotal := c.startup.ShardsTotal.Load()
	memReserved := c.startup.MemReservedBytes.Load()
	memTotal := c.startup.MemTotalBytes.Load()

	phaseLabel := "init"
	if v := c.startup.PhaseLabel.Load(); v != nil {
		phaseLabel = v.(string)
	}

	status := "not_ready"
	if c.startup.AcceptingConnections.Load() {
		status = "accepting"
	}
	resp := map[string]interface{}{
		"status":          status,
		"phase":           phaseLabel,
		"shards_ready":    shardsReady,
		"shards_total":    shardsTotal,
		"mem_reserved_gb": float64(memReserved) / (1 << 30),
		"mem_total_gb":    float64(memTotal) / (1 << 30),
	}

	if shardsTotal > 0 {
		pct := float64(shardsReady) / float64(shardsTotal) * 100
		resp["percent"] = pct

		avgNs := c.startup.AvgShardNanos.Load()
		if avgNs > 0 && shardsReady < shardsTotal {
			remaining := shardsTotal - shardsReady
			etaSeconds := float64(remaining) * float64(avgNs) / 1e9
			resp["eta_seconds"] = etaSeconds
		}
	}

	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(resp)
}
