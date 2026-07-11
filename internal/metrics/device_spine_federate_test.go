package metrics

import (
	"strings"
	"testing"
)

// TestParsePeerSnapshotRoundTrip proves federation reuses the wire schema
// RenderJSON emits: a peer's rendered snapshot parses back into the SAME
// struct list, every row marked Remote with the peer origin, metric values
// and the null-on-error nils intact.
func TestParsePeerSnapshotRoundTrip(t *testing.T) {
	peerSnap := []DeviceMetrics{
		{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(42), UtilizationRatio: f(0.9)},
		{Backend: "engine", DeviceID: "vllm0", QueueDepth: f(3)},
	}
	wire, err := RenderJSON(peerSnap)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}

	got, err := ParsePeerSnapshot(wire, "10.0.0.2")
	if err != nil {
		t.Fatalf("ParsePeerSnapshot: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 rows, got %d: %v", len(got), got)
	}
	for i, r := range got {
		if !r.Remote || r.Peer != "10.0.0.2" {
			t.Fatalf("row %d not marked federated: remote=%v peer=%q", i, r.Remote, r.Peer)
		}
	}
	if v, ok := deref(got[0].TokensPerSecond); !ok || v != 42 {
		t.Fatalf("tokens_per_second lost in transit: %v %v", v, ok)
	}
	if got[0].QueueDepth != nil {
		t.Fatalf("unread metric should stay nil across the wire, got %v", *got[0].QueueDepth)
	}
	if v, ok := deref(got[1].QueueDepth); !ok || v != 3 {
		t.Fatalf("queue_depth lost in transit: %v %v", v, ok)
	}
}

// TestParsePeerSnapshotMultiHopAndUnknownFields proves a row that arrives
// already carrying a Peer origin keeps it (multi-hop federation preserves the
// true origin) and that unknown JSON fields from a newer peer are ignored, not
// fatal — ZML's ignore_unknown_fields.
func TestParsePeerSnapshotMultiHopAndUnknownFields(t *testing.T) {
	wire := `[
		{"backend":"nvml","device":"gpu0","remote":true,"peer":"10.0.0.9","power_watts":120,
		 "some_future_metric":7,"nested_future":{"a":1}}
	]`
	got, err := ParsePeerSnapshot([]byte(wire), "10.0.0.2")
	if err != nil {
		t.Fatalf("ParsePeerSnapshot with unknown fields: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 row, got %d", len(got))
	}
	if got[0].Peer != "10.0.0.9" {
		t.Fatalf("multi-hop origin clobbered: peer=%q, want 10.0.0.9", got[0].Peer)
	}
	if !got[0].Remote {
		t.Fatalf("row not marked remote")
	}
	if v, ok := deref(got[0].PowerWatts); !ok || v != 120 {
		t.Fatalf("known field lost next to unknown ones: %v %v", v, ok)
	}
}

// TestParsePeerSnapshotErrors proves malformed peer bytes fail with the peer
// named in the error (skip-not-fatal lives in the caller: one bad peer must be
// identifiable and droppable), while empty and JSON-null inputs are benign.
func TestParsePeerSnapshotErrors(t *testing.T) {
	if _, err := ParsePeerSnapshot([]byte("{not json"), "peer-a"); err == nil {
		t.Fatalf("malformed input should error")
	} else if !strings.Contains(err.Error(), "peer-a") {
		t.Fatalf("error should name the peer, got: %v", err)
	}
	for _, benign := range [][]byte{nil, {}, []byte("null")} {
		got, err := ParsePeerSnapshot(benign, "peer-a")
		if err != nil || len(got) != 0 {
			t.Fatalf("benign input %q: got %v, %v", benign, got, err)
		}
	}
}

// TestFederatePaneOfGlass proves Federate builds the one pane-of-glass list —
// local rows first then peers in order, inputs never aliased — and that the
// existing renderers consume the federated list with no new surface: the
// Prometheus exposition grows remote/peer labels on federated samples only.
func TestFederatePaneOfGlass(t *testing.T) {
	local := []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", PowerWatts: f(120)}}
	peer := []DeviceMetrics{{Backend: "engine", DeviceID: "vllm0", Remote: true, Peer: "10.0.0.2", PowerWatts: f(80)}}

	merged := Federate(local, peer)
	if len(merged) != 2 || merged[0].DeviceID != "gpu0" || merged[1].DeviceID != "vllm0" {
		t.Fatalf("pane-of-glass order wrong: %v", merged)
	}
	merged[0].Backend = "mutated"
	if local[0].Backend != "nvml" {
		t.Fatalf("Federate aliased its local input")
	}
	merged[0].Backend = "nvml"

	prom, err := RenderProm(merged)
	if err != nil {
		t.Fatalf("RenderProm federated: %v", err)
	}
	text := string(prom)
	if !strings.Contains(text, `peer="10.0.0.2"`) || !strings.Contains(text, `remote="true"`) {
		t.Fatalf("federated sample missing remote/peer labels:\n%s", text)
	}
	if !strings.Contains(text, `fak_device_power_watts{backend="nvml",device="gpu0"} 120`) {
		t.Fatalf("local sample must render without federation labels:\n%s", text)
	}
}
