package cluster

import "testing"

// fak-test:runtime fast est=5ms lane=default
// Estimate only: tiny owned queues; no Start, connection or background consumer.
func TestReplicationFullQueueSkipsCopies(t *testing.T) {
	// AllocsPerRun controls GOMAXPROCS, so this test is deliberately not parallel.
	for _, op := range []string{"set", "delete"} {
		t.Run(op, func(t *testing.T) {
			r := &Replicator{queue: make(chan replicateOp, 1)}
			retained := replicateOp{Key: []byte("queued-key"), Value: []byte("queued-value"), TTLMs: 17, Op: "set"}
			r.queue <- retained
			key, value := []byte("new-key"), []byte("new-value")
			enqueue := func() {
				if op == "set" {
					r.ReplicateSet(key, value, 123)
				} else {
					r.ReplicateDelete(key)
				}
			}
			if got := testing.AllocsPerRun(100, enqueue); got != 0 {
				t.Fatalf("full %s queue allocated %g times per dropped operation", op, got)
			}
			if r.QueueDepth() != 1 {
				t.Fatalf("full queue depth=%d, want 1", r.QueueDepth())
			}
			got := <-r.queue
			if &got.Key[0] != &retained.Key[0] || &got.Value[0] != &retained.Value[0] ||
				string(got.Key) != "queued-key" || string(got.Value) != "queued-value" || got.TTLMs != 17 || got.Op != "set" {
				t.Fatal("dropped operations changed the queued entry")
			}

			// Capacity becoming available must still admit an owned copy.
			enqueue()
			if r.QueueDepth() != 1 {
				t.Fatal("available queue did not admit the operation")
			}
			key[0], value[0] = 'X', 'Y'
			got = <-r.queue
			if string(got.Key) != "new-key" || got.Op != op {
				t.Fatalf("admitted key/op was not preserved: %+v", got)
			}
			if op == "set" {
				if string(got.Value) != "new-value" || got.TTLMs != 123 {
					t.Fatalf("admitted SET value/TTL was not preserved: %+v", got)
				}
			} else if got.Value != nil || got.TTLMs != 0 {
				t.Fatalf("DELETE unexpectedly acquired SET fields: %+v", got)
			}
		})
	}
}
