package dispatch

import (
	"bytes"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=50ms lane=default
// Estimate only: one shard with a 1 KiB logical region, no listener or sleeps.
func TestMSetReplicatesOnlySuccessfulKeys(t *testing.T) {
	t.Parallel()
	var pressure atomic.Bool
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards: 1, MaxMemoryGB: 0, IndexCapacity: 1024, DispatchTimeoutMs: 1000,
		EvictionPolicy: "wtinylfu", SlabDistribution: "dedicated", ModelPageBytes: 256,
		SystemOOMFlag: &pressure,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)
	ring := cluster.NewRing()
	ring.AddNode(&cluster.Node{ID: "local", IsLocal: true, Alive: true})
	// The unstarted worker leaves replication queue depth deterministic.
	replicator := cluster.NewReplicator(ring, "local", 2)
	d := &Dispatcher{Ring: ring, Replicator: replicator, LocalID: "local"}
	d.SetManager(mgr)

	for _, tc := range []struct {
		name     string
		keys     [][]byte
		values   [][]byte
		pressure bool
		flags    uint8
		statuses []byte
		queued   int
		stored   string
	}{
		{"whole_shard_refused", [][]byte{[]byte("blocked")}, [][]byte{[]byte("v")}, true, 0, []byte{1}, 0, ""},
		{"mixed", [][]byte{[]byte("oversize"), []byte("good")}, [][]byte{bytes.Repeat([]byte("x"), 257), []byte("v1")}, false, 0, []byte{1, 0}, 1, "v1"},
		{"success", [][]byte{[]byte("good")}, [][]byte{[]byte("v2")}, false, 0, nil, 2, "v2"},
		{"already_replicated", [][]byte{[]byte("good")}, [][]byte{[]byte("v3")}, false, protocol.FlagReplicated, nil, 2, "v3"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pressure.Store(tc.pressure)
			resp := d.Dispatch(protocol.Message{
				Header: protocol.Header{OpCode: protocol.OpMSet, Flags: tc.flags, RequestID: 17},
				Body:   protocol.EncodeMSetBody(tc.keys, tc.values),
			})
			wantCode := uint8(protocol.RespOK)
			if tc.statuses != nil {
				wantCode = protocol.RespMSetResult
				statuses, err := protocol.DecodeMSetResultResponse(resp.Body)
				if err != nil || !bytes.Equal(statuses, tc.statuses) {
					t.Fatalf("statuses=%v err=%v, want %v", statuses, err, tc.statuses)
				}
			}
			if resp.Header.OpCode != wantCode || resp.Header.RequestID != 17 {
				t.Fatalf("response header=%+v, want opcode=%#x and request 17", resp.Header, wantCode)
			}
			if got := replicator.QueueDepth(); got != tc.queued {
				t.Fatalf("queued replication=%d, want %d", got, tc.queued)
			}
			got, found := mgr.Get([]byte("good"))
			if found != (tc.stored != "") || string(got) != tc.stored {
				t.Fatalf("local success: found=%t value=%q, want %q", found, got, tc.stored)
			}
			for _, key := range []string{"blocked", "oversize"} {
				if value, found := mgr.Get([]byte(key)); found {
					t.Fatalf("failed key %q was stored: %q", key, value)
				}
			}
		})
	}
}
