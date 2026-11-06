package cluster

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestReplicationOwnsQueuedPayload(t *testing.T) {
	t.Parallel()
	for _, op := range []string{"set", "delete"} {
		t.Run(op, func(t *testing.T) {
			r := &Replicator{queue: make(chan replicateOp, 1)}
			body := []byte("keyvalue")
			if op == "set" {
				r.ReplicateSet(body[:3], body[3:], 123)
			} else {
				r.ReplicateDelete(body[:3])
			}
			if len(r.queue) != 1 {
				t.Fatal("operation was not queued")
			}
			// Model request-buffer reuse after dispatch, before replication reads it.
			for i := range body {
				body[i] = '?'
			}
			got := <-r.queue
			if got.Op != op || string(got.Key) != "key" {
				t.Fatalf("queued operation changed: op=%q key=%q", got.Op, got.Key)
			}
			if op == "set" {
				if string(got.Value) != "value" || got.TTLMs != 123 {
					t.Fatalf("queued SET changed: value=%q TTL=%d", got.Value, got.TTLMs)
				}
				got.Value[0] = '!'
			} else if got.Value != nil || got.TTLMs != 0 {
				t.Fatal("DELETE gained a value or TTL")
			}
			got.Key[0] = '!'
			if !bytes.Equal(body, []byte("????????")) {
				t.Fatal("queued payload still aliases caller storage")
			}
		})
	}
	for _, input := range [][]byte{nil, {}} {
		r := &Replicator{queue: make(chan replicateOp, 1)}
		r.ReplicateSet(input, input, 0)
		if len(r.queue) != 1 {
			t.Fatal("empty SET was not queued")
		}
		got := <-r.queue
		if got.Op != "set" || got.TTLMs != 0 || len(got.Key) != 0 || len(got.Value) != 0 || (got.Key == nil) != (input == nil) || (got.Value == nil) != (input == nil) {
			t.Fatal("nil or empty payload semantics changed")
		}
	}
	t.Run("full queue still drops", func(t *testing.T) {
		r := &Replicator{queue: make(chan replicateOp, 1)}
		r.ReplicateSet([]byte("first"), []byte("retained"), 9)
		r.ReplicateSet([]byte("second"), []byte("dropped"), 10)
		r.ReplicateDelete([]byte("third"))
		if len(r.queue) != 1 {
			t.Fatal("full-queue drop changed capacity")
		}
		got := <-r.queue
		if got.Op != "set" || string(got.Key) != "first" || string(got.Value) != "retained" || got.TTLMs != 9 {
			t.Fatal("full-queue drop replaced the accepted operation")
		}
	})
}
