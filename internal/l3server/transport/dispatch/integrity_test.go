package dispatch

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// newSmallDispatcher creates a dispatcher with a single shard and small index
// capacity, suitable for triggering evictions quickly.
func newSmallDispatcher(t *testing.T) *Dispatcher {
	t.Helper()
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      1,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	d := &Dispatcher{
		ConnReg:   metrics.NewConnRegistry(),
		StartedAt: time.Now(),
	}
	d.SetManager(mgr)
	return d
}

// setKeyBytes is a helper that SETs a key with a raw byte value.
func setKeyBytes(t *testing.T, d *Dispatcher, key string, value []byte) {
	t.Helper()
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagNone},
		Body:   protocol.EncodeKVBody([]byte(key), value, 0, protocol.FlagNone),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("SET %q failed: opcode=%#x", key, resp.Header.OpCode)
	}
}

// getKey is a helper that GETs a key and returns (value, found).
func getKey(t *testing.T, d *Dispatcher, key string) ([]byte, bool) {
	t.Helper()
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpGet},
		Body:   protocol.EncodeKeyBody([]byte(key)),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespValue {
		t.Fatalf("GET %q: unexpected opcode %#x", key, resp.Header.OpCode)
	}
	val, found, err := protocol.DecodeValueResponse(resp.Body)
	if err != nil {
		t.Fatalf("GET %q: decode error: %v", key, err)
	}
	return val, found
}

// TestSetOverwriteProtocol verifies SET-overwrite through the full protocol stack.
func TestSetOverwriteProtocol(t *testing.T) {
	d := newTestDispatcher(t)

	valA := bytes.Repeat([]byte{0xAA}, 1024)
	valB := bytes.Repeat([]byte{0xBB}, 1024)
	setKeyBytes(t, d, "ow-key", valA)
	setKeyBytes(t, d, "ow-key", valB)

	got, found := getKey(t, d, "ow-key")
	if !found {
		t.Fatal("ow-key: not found after overwrite")
	}
	if !bytes.Equal(got, valB) {
		t.Errorf("ow-key: got value A or corrupted data, want value B")
	}

	// Test size change: grow then shrink
	valBig := bytes.Repeat([]byte{0xCC}, 4096)
	setKeyBytes(t, d, "ow-key", valBig)
	got, found = getKey(t, d, "ow-key")
	if !found {
		t.Fatal("ow-key: not found after grow")
	}
	if !bytes.Equal(got, valBig) {
		t.Error("ow-key: value corrupted after grow")
	}

	valSmall := bytes.Repeat([]byte{0xDD}, 256)
	setKeyBytes(t, d, "ow-key", valSmall)
	got, found = getKey(t, d, "ow-key")
	if !found {
		t.Fatal("ow-key: not found after shrink")
	}
	if !bytes.Equal(got, valSmall) {
		t.Error("ow-key: value corrupted after shrink")
	}
}

// TestFlushThenGetMiss verifies FLUSH through the protocol makes all keys miss.
func TestFlushThenGetMiss(t *testing.T) {
	d := newTestDispatcher(t)

	numKeys := 10
	for i := 0; i < numKeys; i++ {
		setKey(t, d, fmt.Sprintf("fk-%04d", i), "some-value")
	}

	// FLUSH
	flushMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpFlush},
	}
	resp := d.Dispatch(flushMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("FLUSH: opcode=%#x", resp.Header.OpCode)
	}

	// All GETs should miss
	for i := 0; i < numKeys; i++ {
		_, found := getKey(t, d, fmt.Sprintf("fk-%04d", i))
		if found {
			t.Errorf("fk-%04d: still found after FLUSH", i)
		}
	}
}

// TestEvictionSurvivorsIntegrity inserts more keys than IndexCapacity through
// the protocol layer and verifies surviving values are byte-for-byte correct.
func TestEvictionSurvivorsIntegrity(t *testing.T) {
	d := newSmallDispatcher(t) // IndexCapacity=1024, single shard

	total := 1200
	valSize := 512
	type entry struct {
		key    string
		marker byte
	}

	entries := make([]entry, total)
	for i := 0; i < total; i++ {
		key := fmt.Sprintf("es-%05d", i)
		marker := byte(i % 256)
		val := bytes.Repeat([]byte{marker}, valSize)
		entries[i] = entry{key: key, marker: marker}
		setKeyBytes(t, d, key, val)
	}

	found, missing, corrupted := 0, 0, 0
	for _, e := range entries {
		val, ok := getKey(t, d, e.key)
		if !ok {
			missing++
			continue
		}
		found++
		if len(val) != valSize {
			t.Errorf("%s: length %d, want %d", e.key, len(val), valSize)
			corrupted++
			continue
		}
		for j, b := range val {
			if b != e.marker {
				t.Errorf("%s: byte[%d]=%d, want %d", e.key, j, b, e.marker)
				corrupted++
				break
			}
		}
	}

	t.Logf("found=%d missing=%d corrupted=%d total=%d", found, missing, corrupted, total)
	if found == 0 {
		t.Error("no keys survived â€” broken")
	}
	if missing == 0 {
		t.Error("no evictions â€” increase total or decrease IndexCapacity")
	}
	if corrupted > 0 {
		t.Errorf("%d values corrupted after eviction", corrupted)
	}
}

// TestTTLKeyIntegrity verifies TTL behavior through the protocol layer.
func TestTTLKeyIntegrity(t *testing.T) {
	d := newTestDispatcher(t)

	// SET with long TTL
	longVal := bytes.Repeat([]byte{0xEE}, 512)
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagWithTTL},
		Body:   protocol.EncodeKVBody([]byte("ttl-long"), longVal, 60000, protocol.FlagWithTTL),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("SET ttl-long: opcode=%#x", resp.Header.OpCode)
	}

	// SET with very short TTL
	shortVal := bytes.Repeat([]byte{0xFF}, 512)
	msg = protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagWithTTL},
		Body:   protocol.EncodeKVBody([]byte("ttl-short"), shortVal, 1, protocol.FlagWithTTL),
	}
	resp = d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("SET ttl-short: opcode=%#x", resp.Header.OpCode)
	}

	// Immediate GET â€” long TTL should be present
	got, found := getKey(t, d, "ttl-long")
	if !found {
		t.Fatal("ttl-long: not found immediately")
	}
	if !bytes.Equal(got, longVal) {
		t.Error("ttl-long: value corrupted")
	}

	// Wait for short TTL to expire
	time.Sleep(20 * time.Millisecond)

	_, found = getKey(t, d, "ttl-short")
	if found {
		t.Error("ttl-short: still found after expiry")
	}

	// Long TTL should still be intact
	got, found = getKey(t, d, "ttl-long")
	if !found {
		t.Fatal("ttl-long: not found after neighbor expiry")
	}
	if !bytes.Equal(got, longVal) {
		t.Error("ttl-long: value corrupted after neighbor expiry")
	}
}
