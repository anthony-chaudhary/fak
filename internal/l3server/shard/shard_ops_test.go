package shard

import (
	"bytes"
	"testing"
	"time"
)

// TestExpiredKeyReclaimsSlot is the regression witness for #13517: a directly
// observed TTL expiry on the read paths (GET/MGET/EXISTS) must free the slab
// key/value slots, not merely drop the index and eviction entries. Before the
// fix the allocator's AllocatedBytes stayed elevated after the expiry was
// observed, orphaning the slab slot forever -- the periodic TTL sweep can no
// longer see the entry (it was already removed from the index), so the RSS
// growth is permanent for any TTL workload.
func TestExpiredKeyReclaimsSlot(t *testing.T) {
	cases := []struct {
		name string
		op   func(key []byte, hash uint64) ShardOp
	}{
		{"GET", func(key []byte, hash uint64) ShardOp {
			return ShardOp{Type: OpGet, Key: key, KeyHash: hash}
		}},
		{"GETAlloc", func(key []byte, hash uint64) ShardOp {
			return ShardOp{Type: OpGetAlloc, Key: key, KeyHash: hash}
		}},
		{"MGET", func(key []byte, hash uint64) ShardOp {
			return ShardOp{Type: OpMGet, Keys: [][]byte{key}, KeyHashes: []uint64{hash}}
		}},
		{"MGETWithAlloc", func(key []byte, hash uint64) ShardOp {
			return ShardOp{Type: OpMGetWithAlloc, Keys: [][]byte{key}, KeyHashes: []uint64{hash}}
		}},
		{"EXISTS", func(key []byte, hash uint64) ShardOp {
			return ShardOp{Type: OpTest, Key: key, KeyHash: hash}
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestShard(t, 0, 0, 0, false)
			s.Start()
			defer func() {
				s.Stop()
				<-s.Done()
			}()

			a := s.Allocator()
			baseline := a.AllocatedBytes()

			key := []byte("expire-me")
			val := bytes.Repeat([]byte{0xAB}, 1024)
			if r := s.Submit(ShardOp{
				Type:    OpSet,
				Key:     key,
				KeyHash: 1,
				Value:   val,
				TTLMs:   1,
				Result:  make(chan OpResult, 1),
			}); r.Err != nil {
				t.Fatalf("SET: %v", r.Err)
			}
			if after := a.AllocatedBytes(); after <= baseline {
				t.Fatalf("SET allocated nothing: baseline=%d after=%d", baseline, after)
			}

			time.Sleep(20 * time.Millisecond)

			op := tc.op(key, 1)
			op.Result = make(chan OpResult, 1)
			res := s.Submit(op)
			if res.Found {
				t.Fatalf("%s: expired entry still reported as found", tc.name)
			}

			if got := a.AllocatedBytes(); got != baseline {
				t.Fatalf("%s: slab slot not reclaimed after observed expiry: allocated=%d, want baseline %d (orphaned %d bytes)",
					tc.name, got, baseline, got-baseline)
			}
		})
	}
}
