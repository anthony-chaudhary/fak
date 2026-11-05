package dispatch

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=50ms lane=default
// Estimate only: one shard with a 1 KiB logical slab region, no listener or sleeps.
func TestReplicatedWritesApplyLocally(t *testing.T) {
	t.Parallel()
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards: 1, MaxMemoryGB: 0, IndexCapacity: 1024, DispatchTimeoutMs: 1000,
		EvictionPolicy: "wtinylfu", SlabDistribution: "dedicated", ModelPageBytes: 256,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const primaryAddr = "missing-port"
	ring := cluster.NewRing()
	ring.AddNode(&cluster.Node{ID: "primary", Addr: primaryAddr, Alive: true})
	// No Start: the queue witnesses re-replication without any peer I/O.
	replicator := cluster.NewReplicator(ring, "secondary", 2)
	d := &Dispatcher{Ring: ring, Replicator: replicator, LocalID: "secondary"}
	d.SetManager(mgr)
	key, value := []byte("replica-key"), []byte("replica-value")

	for _, tc := range []struct {
		name       string
		op         uint8
		flags      uint8
		local      bool
		wantFound  bool
		wantQueued int
	}{
		{"ordinary_set_routes", protocol.OpSet, 0, false, false, 0},
		{"replicated_set_stays", protocol.OpSet, protocol.FlagReplicated | protocol.FlagWithTTL, true, true, 0},
		{"ordinary_delete_routes", protocol.OpDelete, 0, false, true, 0},
		{"replicated_delete_stays", protocol.OpDelete, protocol.FlagReplicated, true, false, 0},
		{"primary_set_replicates", protocol.OpSet, 0, true, true, 1},
		{"primary_delete_replicates", protocol.OpDelete, 0, true, false, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.wantQueued > 0 {
				// Ordinary writes at the primary must still enqueue replication.
				ring.AddNode(&cluster.Node{ID: "primary", Addr: primaryAddr, Alive: true, IsLocal: true})
			}
			body := protocol.EncodeKeyBody(key)
			if tc.op == protocol.OpSet {
				body = protocol.EncodeKVBody(key, value, 0, tc.flags)
			}
			resp := d.Dispatch(protocol.Message{
				Header: protocol.Header{OpCode: tc.op, Flags: tc.flags, RequestID: 17}, Body: body,
			})
			if tc.local {
				if resp.Header.OpCode != protocol.RespOK || resp.Header.RequestID != 17 {
					t.Fatalf("local write: header=%+v body=%q", resp.Header, resp.Body)
				}
			} else if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 17 || !strings.HasPrefix(string(resp.Body), "proxy connect to "+primaryAddr+": ") {
				// The missing-port address fails parsing without DNS or a connection.
				t.Fatalf("ordinary write bypassed primary routing: header=%+v body=%q", resp.Header, resp.Body)
			}
			got, found := mgr.Get(key)
			if found != tc.wantFound || (found && string(got) != string(value)) {
				t.Fatalf("local data: found=%t value=%q, want found=%t value=%q", found, got, tc.wantFound, value)
			}
			if got := replicator.QueueDepth(); got != tc.wantQueued {
				t.Fatalf("queued replication: got %d, want %d", got, tc.wantQueued)
			}
		})
	}
}
