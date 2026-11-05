package dispatch

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=50ms lane=default
// Estimate only: tiny snapshot/one 1 KiB logical slab; one deliberate 5ms timeout.
func TestRestoreRequiresSuccessfulFlush(t *testing.T) {
	t.Parallel()
	for _, running := range []bool{false, true} {
		name := "flush_timeout"
		if running {
			name = "successful_flush_and_restore"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			entries := [][]snapshot.KVEntry{{{Key: []byte("new"), Value: []byte("value")}}}
			if err := snapshot.Save(dir, "test", "slab", entries); err != nil {
				t.Fatal(err)
			}
			timeoutMs := int64(5)
			if running {
				timeoutMs = 1000
			}
			mgr, err := shard.NewManager(shard.ManagerConfig{
				NumShards: 1, MaxMemoryGB: 0, IndexCapacity: 1024, DispatchTimeoutMs: timeoutMs,
				EvictionPolicy: "wtinylfu", SlabDistribution: "dedicated", ModelPageBytes: 256,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(mgr.Stop)
			if running {
				mgr.Start()
				if !mgr.Set([]byte("old"), []byte("prior"), 0) {
					t.Fatal("could not populate the live control")
				}
			}
			// Call the handler directly to inject an allocated but unstarted
			// shard. The public startup gate normally rejects this fixture.
			d := &Dispatcher{Manager: mgr, SnapshotDir: dir}
			resp := d.HandleRestore(protocol.Message{Header: protocol.Header{OpCode: protocol.OpRestore, RequestID: 41}})
			if !running {
				if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 41 || string(resp.Body) != "restore flush shard 0: shard dispatch timeout" {
					t.Fatalf("failed flush response: header=%+v body=%q", resp.Header, resp.Body)
				}
				// The timeout can win before enqueue (zero pending) or while
				// waiting for its result (one pending). Check only the upper bound.
				_, _, _, _, _, _, _, _, _, _, _, _, pending, _ := mgr.Shard(0).OpLatencySnapshot()
				if pending > 1 {
					t.Fatalf("pending operations=%d, want at most one", pending)
				}
				return
			}
			var result struct {
				Keys int64  `json:"keys"`
				Dir  string `json:"dir"`
			}
			if err := json.Unmarshal(resp.Body, &result); err != nil {
				t.Fatal(err)
			}
			if resp.Header.OpCode != protocol.RespOK || resp.Header.RequestID != 41 || result.Keys != 1 || result.Dir != dir {
				t.Fatalf("success response: header=%+v result=%+v", resp.Header, result)
			}
			if value, found := mgr.Get([]byte("new")); !found || string(value) != "value" {
				t.Fatalf("snapshot value: found=%t value=%q", found, value)
			}
			if _, found := mgr.Get([]byte("old")); found {
				t.Fatal("successful restore did not flush the previous entry")
			}
		})
	}
}
