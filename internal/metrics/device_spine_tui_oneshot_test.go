package metrics

import (
	"bytes"
	"testing"
)

// TestFleetViewOneShotSharesRenderText proves the TUI's non-interactive fallback
// is the SAME one-shot the spine already has — byte-for-byte RenderText over the
// identical snapshot, never a second aligned-table renderer. This is the
// "sharing RenderText's one-shot for a `fak status` pretty-print" contract.
func TestFleetViewOneShotSharesRenderText(t *testing.T) {
	snap := Federate(
		[]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(42)}},
		[]DeviceMetrics{{Backend: "engine", DeviceID: "vllm0", Remote: true, Peer: "10.0.0.2", QueueDepth: f(3)}},
	)
	v := NewFleetView()
	v.Observe(snap)
	if got, want := v.OneShot(snap), RenderText(snap); !bytes.Equal(got, want) {
		t.Fatalf("OneShot must delegate to RenderText byte-for-byte:\n got %q\nwant %q", got, want)
	}
	// The one-shot is stateless w.r.t. the view's sparkline history: observing
	// more polls does not change the one-shot bytes for the same snapshot.
	v.Observe(snap)
	if got, want := v.OneShot(snap), RenderText(snap); !bytes.Equal(got, want) {
		t.Fatalf("OneShot must stay a stateless delegate to RenderText")
	}
}
