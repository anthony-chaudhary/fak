package gateway

// metrics_cache_stream_test.go — the unified cache-stream family (fak_cache_*) on
// /metrics, fed live by the vDSO tier-2 cache-event sink that New subscribes. Before
// this wiring the vDSO already emitted first-class cachemeta lifecycle events
// (fill/hit/evict/revoke) but no serving-path consumer rendered them, so the
// strongest local cache's behavior was invisible to a scrape. The test drives a real
// fill + hit through the process-global vDSO and asserts both the in-process fold and
// the rendered exposition observe them.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func TestMetricsExposesCacheStream(t *testing.T) {
	srv := newTestServer(t)
	defer srv.Close() // detaches the sink this server installed on the global vDSO

	// A unique read so the tier-2 key cannot collide with a sibling test's entry: one
	// fill (the Emit) then one hit (the Lookup), both folded into srv.cacheStream by
	// the sink New installed on the process-global vDSO.
	call := &abi.ToolCall{
		Tool: "fak_cache_stream_probe",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"cache-stream-witness"}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
	res := &abi.Result{Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"ok":true}`)}}

	vdso.Default.Emit(abi.Event{Kind: abi.EvComplete, Call: call, Result: res}) // -> CacheFill
	if _, ok := vdso.Default.Lookup(context.Background(), call); !ok {          // -> CacheHit
		t.Fatalf("expected a tier-2 hit after the fill (no hit event would be folded)")
	}

	// In-process fold: the live producer reached srv.cacheStream.
	snap := srv.cacheStream.Snapshot()
	kinds := map[string]uint64{}
	for _, r := range snap.Rows {
		if r.Plane == "tool_result" && r.Tier == "dram" {
			kinds[r.Kind] += r.Count
		}
	}
	if kinds["fill"] < 1 {
		t.Fatalf("cache stream missing a tool_result/dram fill event: %+v", snap.Rows)
	}
	if kinds["hit"] < 1 {
		t.Fatalf("cache stream missing a tool_result/dram hit event: %+v", snap.Rows)
	}

	// Rendered exposition: the family and the labeled fill breakdown reach /metrics.
	text := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_cache_events_total counter",
		"fak_cache_events_total ",
		"# TYPE fak_cache_event_breakdown_total counter",
		`plane="tool_result",tier="dram",kind="fill"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}
}
