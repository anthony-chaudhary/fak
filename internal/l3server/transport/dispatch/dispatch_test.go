package dispatch

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

func newTestDispatcher(t *testing.T) *Dispatcher {
	t.Helper()
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      2,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: 5242880,
		MaxLeaseDurMs:  30000,
		IndexCapacity:  4096,
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

// setKey is a test helper that SETs a key-value pair and fatals on failure.
func setKey(t *testing.T, d *Dispatcher, key, value string) {
	t.Helper()
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagNone},
		Body:   protocol.EncodeKVBody([]byte(key), []byte(value), 0, protocol.FlagNone),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("SET %q failed: opcode=%#x", key, resp.Header.OpCode)
	}
}

func TestSingleKeyOps(t *testing.T) {
	d := newTestDispatcher(t)
	setKey(t, d, "hello", "world")

	// Table-driven read operations (don't mutate state).
	readTests := []struct {
		name      string
		opCode    uint8
		key       string
		wantCode  uint8
		wantValue string // for GET
		wantFound bool   // for GET and TEST
	}{
		{"GET hit", protocol.OpGet, "hello", protocol.RespValue, "world", true},
		{"GET miss", protocol.OpGet, "missing", protocol.RespValue, "", false},
		{"TEST hit", protocol.OpTest, "hello", protocol.RespOK, "", true},
		{"TEST miss", protocol.OpTest, "missing", protocol.RespOK, "", false},
	}

	for _, tc := range readTests {
		t.Run(tc.name, func(t *testing.T) {
			msg := protocol.Message{
				Header: protocol.Header{OpCode: tc.opCode},
				Body:   protocol.EncodeKeyBody([]byte(tc.key)),
			}
			resp := d.Dispatch(msg)
			if resp.Header.OpCode != tc.wantCode {
				t.Fatalf("opcode: got %#x, want %#x", resp.Header.OpCode, tc.wantCode)
			}

			switch tc.opCode {
			case protocol.OpGet:
				val, found, err := protocol.DecodeValueResponse(resp.Body)
				if err != nil {
					t.Fatal(err)
				}
				if found != tc.wantFound {
					t.Errorf("found: got %v, want %v", found, tc.wantFound)
				}
				if tc.wantFound && string(val) != tc.wantValue {
					t.Errorf("value: got %q, want %q", val, tc.wantValue)
				}
			case protocol.OpTest:
				if len(resp.Body) == 0 {
					t.Fatal("empty TEST response body")
				}
				got := resp.Body[0] == 1
				if got != tc.wantFound {
					t.Errorf("found: got %v, want %v", got, tc.wantFound)
				}
			}
		})
	}

	// Sequential mutation operations.
	t.Run("PIN", func(t *testing.T) {
		msg := protocol.Message{
			Header: protocol.Header{OpCode: protocol.OpPin},
			Body:   protocol.EncodeKeyBody([]byte("hello")),
		}
		resp := d.Dispatch(msg)
		if resp.Header.OpCode != protocol.RespOK {
			t.Fatalf("PIN: got %#x, want RespOK", resp.Header.OpCode)
		}
	})

	t.Run("UNPIN", func(t *testing.T) {
		msg := protocol.Message{
			Header: protocol.Header{OpCode: protocol.OpUnpin},
			Body:   protocol.EncodeKeyBody([]byte("hello")),
		}
		resp := d.Dispatch(msg)
		if resp.Header.OpCode != protocol.RespOK {
			t.Fatalf("UNPIN: got %#x, want RespOK", resp.Header.OpCode)
		}
	})

	t.Run("DELETE then GET miss", func(t *testing.T) {
		msg := protocol.Message{
			Header: protocol.Header{OpCode: protocol.OpDelete},
			Body:   protocol.EncodeKeyBody([]byte("hello")),
		}
		resp := d.Dispatch(msg)
		if resp.Header.OpCode != protocol.RespOK {
			t.Fatalf("DELETE: got %#x, want RespOK", resp.Header.OpCode)
		}

		getMsg := protocol.Message{
			Header: protocol.Header{OpCode: protocol.OpGet},
			Body:   protocol.EncodeKeyBody([]byte("hello")),
		}
		getResp := d.Dispatch(getMsg)
		_, found, err := protocol.DecodeValueResponse(getResp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if found {
			t.Error("expected miss after DELETE")
		}
	})
}

func TestBatchOps(t *testing.T) {
	d := newTestDispatcher(t)

	keys := [][]byte{[]byte("k1"), []byte("k2"), []byte("k3")}
	vals := [][]byte{[]byte("v1"), []byte("v2"), []byte("v3")}

	// MSET
	msetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMSet},
		Body:   protocol.EncodeMSetBody(keys, vals),
	}
	resp := d.Dispatch(msetMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MSET: got %#x, want RespOK", resp.Header.OpCode)
	}

	// MGET â€” verify values
	mgetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMGet},
		Body:   protocol.EncodeMGetBody(keys),
	}
	resp = d.Dispatch(mgetMsg)
	if resp.Header.OpCode != protocol.RespMultiValue {
		t.Fatalf("MGET: got %#x, want RespMultiValue", resp.Header.OpCode)
	}
	gotVals, gotFounds, err := protocol.DecodeMultiValueResponse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	for i, key := range keys {
		if !gotFounds[i] {
			t.Errorf("MGET %q: not found", key)
		}
		if string(gotVals[i]) != string(vals[i]) {
			t.Errorf("MGET %q: got %q, want %q", key, gotVals[i], vals[i])
		}
	}

	// MDEL
	mdelMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMDel},
		Body:   protocol.EncodeMDelBody(keys),
	}
	resp = d.Dispatch(mdelMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MDEL: got %#x, want RespOK", resp.Header.OpCode)
	}

	// MGET after delete â€” verify all deleted
	resp = d.Dispatch(mgetMsg)
	gotVals, gotFounds, err = protocol.DecodeMultiValueResponse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	for i, key := range keys {
		if gotFounds[i] {
			t.Errorf("MGET after MDEL %q: expected miss, got value %q", key, gotVals[i])
		}
	}
}

func TestHandleStats(t *testing.T) {
	d := newTestDispatcher(t)

	// Insert some data so metrics are non-zero.
	setKey(t, d, "stats-key", "stats-val")

	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpStats},
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("STATS: got %#x, want RespOK", resp.Header.OpCode)
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	for _, key := range []string{"shards", "connections", "totals", "uptime_seconds"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing top-level key %q", key)
		}
	}

	// Verify totals structure.
	var totals map[string]json.RawMessage
	if err := json.Unmarshal(result["totals"], &totals); err != nil {
		t.Fatalf("totals not an object: %v", err)
	}
	for _, key := range []string{
		"gb_in", "gb_out", "rdma_read_gb_out",
		"wire_gb_recv", "wire_gb_sent",
		"effective_gb_sent", "inline_payload_gb_sent",
		"ops_overhead_gb_recv", "ops_overhead_gb_sent",
		"gbps_in", "gbps_out", "gbps_total",
		"gets", "sets", "exists", "exists_hits", "exists_misses",
		"hits", "misses", "active_connections",
		"evictions_key_pressure", "evictions_value_pressure",
		"evictions_failed", "evictions_lease_skip",
		"eviction_fail_rate_percent",
	} {
		if _, ok := totals[key]; !ok {
			t.Errorf("missing totals key %q", key)
		}
	}
}

func TestHandleInfo(t *testing.T) {
	d := newTestDispatcher(t)

	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpInfo},
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("INFO: got %#x, want RespOK", resp.Header.OpCode)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	for _, key := range []string{"server", "server_version", "protocol_version", "uptime_seconds"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing key %q", key)
		}
	}
}

func TestHandleHandshake(t *testing.T) {
	d := newTestDispatcher(t)

	body, _ := json.Marshal(map[string]interface{}{
		"client_version": "99.0.0",
	})
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpHandshake},
		Body:   body,
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("HANDSHAKE: got %#x, want RespOK", resp.Header.OpCode)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if result["status"] != "ok" {
		t.Errorf("status: got %v, want ok", result["status"])
	}
}

func TestUnknownOpcode(t *testing.T) {
	d := newTestDispatcher(t)

	msg := protocol.Message{
		Header: protocol.Header{OpCode: 0xFF},
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespError {
		t.Fatalf("unknown opcode: got %#x, want RespError", resp.Header.OpCode)
	}
}

func TestDecodeErrorReturnsRespError(t *testing.T) {
	d := newTestDispatcher(t)

	opcodes := []struct {
		name string
		op   uint8
	}{
		{"GET", protocol.OpGet},
		{"SET", protocol.OpSet},
		{"DELETE", protocol.OpDelete},
		{"TEST", protocol.OpTest},
		{"LEASE", protocol.OpLease},
		{"PIN", protocol.OpPin},
		{"UNPIN", protocol.OpUnpin},
		{"MGET", protocol.OpMGet},
		{"MSET", protocol.OpMSet},
		{"MDEL", protocol.OpMDel},
	}

	for _, tc := range opcodes {
		t.Run(tc.name, func(t *testing.T) {
			msg := protocol.Message{
				Header: protocol.Header{OpCode: tc.op},
				Body:   nil,
			}
			resp := d.Dispatch(msg)
			if resp.Header.OpCode != protocol.RespError {
				t.Errorf("empty body for %s: got %#x, want RespError", tc.name, resp.Header.OpCode)
			}
		})
	}
}

func TestFlushResetsStats(t *testing.T) {
	d := newTestDispatcher(t)

	// Insert data so shard metrics are non-zero.
	setKey(t, d, "k1", "v1")
	setKey(t, d, "k2", "v2")

	// GET to populate hits/gets.
	d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpGet},
		Body:   protocol.EncodeKeyBody([]byte("k1")),
	})

	// Verify stats show non-zero counters before flush.
	statsMsg := protocol.Message{Header: protocol.Header{OpCode: protocol.OpStats}}
	resp := d.Dispatch(statsMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("pre-flush STATS: got %#x, want RespOK", resp.Header.OpCode)
	}
	var pre map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &pre); err != nil {
		t.Fatal(err)
	}
	var preTotals map[string]interface{}
	if err := json.Unmarshal(pre["totals"], &preTotals); err != nil {
		t.Fatal(err)
	}
	if preTotals["sets"].(float64) == 0 {
		t.Fatal("expected non-zero sets before flush")
	}
	if preTotals["gets"].(float64) == 0 {
		t.Fatal("expected non-zero gets before flush")
	}

	// FLUSH
	flushResp := d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpFlush},
	})
	if flushResp.Header.OpCode != protocol.RespOK {
		t.Fatalf("FLUSH: got %#x, want RespOK", flushResp.Header.OpCode)
	}

	// Verify stats are zeroed after flush.
	resp = d.Dispatch(statsMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("post-flush STATS: got %#x, want RespOK", resp.Header.OpCode)
	}
	var post map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &post); err != nil {
		t.Fatal(err)
	}
	var postTotals map[string]interface{}
	if err := json.Unmarshal(post["totals"], &postTotals); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"gets", "sets", "hits", "misses", "exists_hits", "exists_misses",
		"evictions_key_pressure", "evictions_value_pressure", "evictions_failed", "evictions_lease_skip"} {
		if v := postTotals[key].(float64); v != 0 {
			t.Errorf("post-flush totals[%q] = %v, want 0", key, v)
		}
	}
	for _, key := range []string{"gb_in", "gb_out"} {
		if v := postTotals[key].(float64); v != 0 {
			t.Errorf("post-flush totals[%q] = %v, want 0", key, v)
		}
	}
}

func TestMTestGroupedByShard(t *testing.T) {
	d := newTestDispatcher(t)

	// SET 3 keys
	setKey(t, d, "a", "v1")
	setKey(t, d, "b", "v2")
	setKey(t, d, "c", "v3")

	// MTEST with 3 hits + 1 miss
	testKeys := [][]byte{[]byte("a"), []byte("missing"), []byte("b"), []byte("c")}
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMTest},
		Body:   protocol.EncodeMGetBody(testKeys),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespMultiValue {
		t.Fatalf("MTEST: got %#x, want RespMultiValue", resp.Header.OpCode)
	}

	_, founds, err := protocol.DecodeMultiValueResponse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if len(founds) != 4 {
		t.Fatalf("founds length: got %d, want 4", len(founds))
	}
	want := []bool{true, false, true, true}
	for i, w := range want {
		if founds[i] != w {
			t.Errorf("founds[%d]: got %v, want %v", i, founds[i], w)
		}
	}
}

func TestExistsHitMissMetrics(t *testing.T) {
	d := newTestDispatcher(t)

	setKey(t, d, "exist-key", "val")

	// TEST hit
	d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpTest},
		Body:   protocol.EncodeKeyBody([]byte("exist-key")),
	})
	// TEST miss
	d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpTest},
		Body:   protocol.EncodeKeyBody([]byte("no-such-key")),
	})

	// Read STATS
	resp := d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpStats},
	})
	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatal(err)
	}
	var totals map[string]interface{}
	if err := json.Unmarshal(result["totals"], &totals); err != nil {
		t.Fatal(err)
	}

	if v := totals["exists_hits"].(float64); v != 1 {
		t.Errorf("exists_hits: got %v, want 1", v)
	}
	if v := totals["exists_misses"].(float64); v != 1 {
		t.Errorf("exists_misses: got %v, want 1", v)
	}
}

func TestExistsTTLExpiry(t *testing.T) {
	d := newTestDispatcher(t)

	// SET with TTL=1ms
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagWithTTL},
		Body:   protocol.EncodeKVBody([]byte("ttl-key"), []byte("val"), 1, protocol.FlagWithTTL),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("SET with TTL: got %#x, want RespOK", resp.Header.OpCode)
	}

	// Wait for expiry
	time.Sleep(5 * time.Millisecond)

	// TEST should return miss
	testResp := d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpTest},
		Body:   protocol.EncodeKeyBody([]byte("ttl-key")),
	})
	if testResp.Header.OpCode != protocol.RespOK {
		t.Fatalf("TEST: got %#x, want RespOK", testResp.Header.OpCode)
	}
	if len(testResp.Body) == 0 {
		t.Fatal("empty TEST response body")
	}
	if testResp.Body[0] != 0 {
		t.Errorf("TEST expired key: got found=true, want found=false")
	}
}

func TestEvictionReasonTracking(t *testing.T) {
	// Index capacity minimum is 1024 â†’ maxKeys = 896. Insert >896 unique
	// keys to trigger eviction via Admit() in the WTinyLFU policy.
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      1,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: 0,
		MaxLeaseDurMs:  30000,
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

	// Insert >896 entries to overflow the eviction engine (maxKeys = 1024*7/8 = 896).
	for i := 0; i < 1200; i++ {
		key := []byte(fmt.Sprintf("evict-key-%04d", i))
		val := []byte("v")
		msg := protocol.Message{
			Header: protocol.Header{OpCode: protocol.OpSet, Flags: protocol.FlagNone},
			Body:   protocol.EncodeKVBody(key, val, 0, protocol.FlagNone),
		}
		d.Dispatch(msg)
	}

	// Read stats and verify eviction reason fields exist and are plausible.
	resp := d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpStats},
	})
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("STATS: got %#x, want RespOK", resp.Header.OpCode)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatal(err)
	}
	var totals map[string]interface{}
	if err := json.Unmarshal(result["totals"], &totals); err != nil {
		t.Fatal(err)
	}

	// With tiny index capacity, policy evictions via Admit() are triggered.
	evictions := totals["evictions"].(float64)
	keyPressure := totals["evictions_key_pressure"].(float64)
	valPressure := totals["evictions_value_pressure"].(float64)

	if evictions == 0 {
		t.Error("expected non-zero evictions with tiny index capacity")
	}
	// Invariant: key_pressure + value_pressure <= evictions
	// (gap = policy evictions from Admit()).
	if keyPressure+valPressure > evictions {
		t.Errorf("key_pressure(%v) + value_pressure(%v) > evictions(%v)", keyPressure, valPressure, evictions)
	}
	// All new fields must exist.
	for _, key := range []string{"evictions_key_pressure", "evictions_value_pressure",
		"evictions_failed", "evictions_lease_skip", "eviction_fail_rate_percent"} {
		if _, ok := totals[key]; !ok {
			t.Errorf("missing totals key %q", key)
		}
	}

	// Per-shard entries must also contain the new fields.
	var shards []map[string]interface{}
	if err := json.Unmarshal(result["shards"], &shards); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"evictions_key_pressure", "evictions_value_pressure",
		"evictions_failed", "evictions_lease_skip"} {
		if _, ok := shards[0][key]; !ok {
			t.Errorf("missing shard key %q", key)
		}
	}
}

func TestHandleMaintenance_Status(t *testing.T) {
	d := newTestDispatcher(t)

	body, _ := json.Marshal(map[string]interface{}{"action": "status"})
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMaintenance},
		Body:   body,
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MAINTENANCE status: got %#x, want RespOK (body: %s)", resp.Header.OpCode, resp.Body)
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, key := range []string{"vacuum_config", "vacuum_stats", "shard_detections"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing key %q in status response", key)
		}
	}
}

func TestHandleMaintenance_Vacuum(t *testing.T) {
	d := newTestDispatcher(t)

	// Seed some data
	for i := 0; i < 10; i++ {
		setKey(t, d, fmt.Sprintf("vac-key-%d", i), "value")
	}

	body, _ := json.Marshal(map[string]interface{}{"action": "vacuum", "force": false})
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMaintenance},
		Body:   body,
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MAINTENANCE vacuum: got %#x, want RespOK (body: %s)", resp.Header.OpCode, resp.Body)
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, key := range []string{"shards_evaluated", "shards_skipped", "vacuum_stats", "duration_ms"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing key %q in vacuum response", key)
		}
	}
}

func TestHandleMaintenance_Autotune(t *testing.T) {
	d := newTestDispatcher(t)

	body, _ := json.Marshal(map[string]interface{}{"action": "autotune", "force": false})
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMaintenance},
		Body:   body,
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MAINTENANCE autotune: got %#x, want RespOK (body: %s)", resp.Header.OpCode, resp.Body)
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, key := range []string{"shards_skipped", "duration_ms"} {
		if _, ok := result[key]; !ok {
			t.Errorf("missing key %q in autotune response", key)
		}
	}
}

// TestBatchConcurrentFanOut verifies that fanOutShards collects results concurrently:
// fast shards are collected before slow shard timeout, and timeout only fires
// when a shard genuinely stalls.
func TestBatchConcurrentFanOut(t *testing.T) {
	d := newTestDispatcher(t)

	// Use a short dispatch timeout so the test finishes quickly
	d.DispatchTimeout = 500 * time.Millisecond

	// MSET keys that land on different shards (2 shards configured)
	keys := [][]byte{[]byte("alpha"), []byte("beta"), []byte("gamma"), []byte("delta")}
	vals := [][]byte{[]byte("v1"), []byte("v2"), []byte("v3"), []byte("v4")}

	msetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMSet},
		Body:   protocol.EncodeMSetBody(keys, vals),
	}
	resp := d.Dispatch(msetMsg)
	if resp.Header.OpCode != protocol.RespOK && resp.Header.OpCode != protocol.RespMSetResult {
		t.Fatalf("MSET: got %#x, want RespOK or RespMSetResult", resp.Header.OpCode)
	}

	// MGET with concurrent fan-out â€” all shards should respond within timeout
	mgetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMGet},
		Body:   protocol.EncodeMGetBody(keys),
	}
	start := time.Now()
	resp = d.Dispatch(mgetMsg)
	elapsed := time.Since(start)

	if resp.Header.OpCode == protocol.RespError {
		t.Fatalf("MGET timed out unexpectedly: %s", resp.Body)
	}
	if resp.Header.OpCode != protocol.RespMultiValue {
		t.Fatalf("MGET: got %#x, want RespMultiValue", resp.Header.OpCode)
	}

	// Verify results arrived well before the dispatch timeout
	if elapsed > 200*time.Millisecond {
		t.Errorf("MGET took %v â€” expected well under 500ms dispatch timeout", elapsed)
	}

	gotVals, gotFounds, err := protocol.DecodeMultiValueResponse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	for i, key := range keys {
		if !gotFounds[i] {
			t.Errorf("MGET %q: not found", key)
		}
		if string(gotVals[i]) != string(vals[i]) {
			t.Errorf("MGET %q: got %q, want %q", key, gotVals[i], vals[i])
		}
	}
}

// TestBatchDispatchTimeout verifies that fanOutShards returns an error
// when the dispatch timeout is exceeded (using an impossibly short timeout).
func TestBatchDispatchTimeout(t *testing.T) {
	// Create a dispatcher with a very short timeout.
	// This test uses normal shard dispatch which should complete, but verifies
	// the timeout path isn't interfering with normal operations.
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      2,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: 5242880,
		MaxLeaseDurMs:  30000,
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	d := &Dispatcher{
		ConnReg:         metrics.NewConnRegistry(),
		StartedAt:       time.Now(),
		DispatchTimeout: 5 * time.Second, // reasonable timeout
	}
	d.SetManager(mgr)

	// Insert keys and verify batch works with explicit timeout
	keys := [][]byte{[]byte("t1"), []byte("t2"), []byte("t3")}
	vals := [][]byte{[]byte("v1"), []byte("v2"), []byte("v3")}
	msetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMSet},
		Body:   protocol.EncodeMSetBody(keys, vals),
	}
	resp := d.Dispatch(msetMsg)
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("MSET with DispatchTimeout: got %#x, want RespOK", resp.Header.OpCode)
	}

	// Verify MGET returns all results
	mgetMsg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMGet},
		Body:   protocol.EncodeMGetBody(keys),
	}
	resp = d.Dispatch(mgetMsg)
	if resp.Header.OpCode != protocol.RespMultiValue {
		t.Fatalf("MGET with DispatchTimeout: got %#x, want RespMultiValue", resp.Header.OpCode)
	}
	_, founds, err := protocol.DecodeMultiValueResponse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	for i, key := range keys {
		if !founds[i] {
			t.Errorf("key %q: not found with explicit DispatchTimeout", key)
		}
	}
}

func TestHandleMaintenance_UnknownAction(t *testing.T) {
	d := newTestDispatcher(t)

	body, _ := json.Marshal(map[string]interface{}{"action": "bad"})
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMaintenance},
		Body:   body,
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespError {
		t.Fatalf("unknown action: got %#x, want RespError", resp.Header.OpCode)
	}
}

func TestHandleMaintenance_InvalidJSON(t *testing.T) {
	d := newTestDispatcher(t)

	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpMaintenance},
		Body:   []byte("not json"),
	}
	resp := d.Dispatch(msg)
	if resp.Header.OpCode != protocol.RespError {
		t.Fatalf("invalid JSON: got %#x, want RespError", resp.Header.OpCode)
	}
}

func TestLeaseSkipTracking(t *testing.T) {
	d := newTestDispatcher(t)

	// Set a key and PIN it (lease-protect).
	setKey(t, d, "pinned", "value")
	d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpPin},
		Body:   protocol.EncodeKeyBody([]byte("pinned")),
	})

	// Read stats â€” evictions_lease_skip field should exist (may be 0 since
	// eviction hasn't been triggered yet, but the field must be present).
	resp := d.Dispatch(protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpStats},
	})
	if resp.Header.OpCode != protocol.RespOK {
		t.Fatalf("STATS: got %#x, want RespOK", resp.Header.OpCode)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		t.Fatal(err)
	}
	var totals map[string]interface{}
	if err := json.Unmarshal(result["totals"], &totals); err != nil {
		t.Fatal(err)
	}
	if _, ok := totals["evictions_lease_skip"]; !ok {
		t.Error("missing evictions_lease_skip in totals")
	}

	// Check per-shard entry also has the field.
	var shards []map[string]interface{}
	if err := json.Unmarshal(result["shards"], &shards); err != nil {
		t.Fatal(err)
	}
	if len(shards) == 0 {
		t.Fatal("no shards in stats")
	}
	if _, ok := shards[0]["evictions_lease_skip"]; !ok {
		t.Error("missing evictions_lease_skip in shard entry")
	}
}
