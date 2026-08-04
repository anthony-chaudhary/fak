package gateway

// debug_vars_defer_shed_test.go — the #3647 done-condition witness: the cold-tool-defer shed's
// COUNT and its DEFERRED-TOOL LIST reach an operator on the served /debug/vars document, and read
// there as their own mechanism rather than as the compaction shed.
//
// The existing served-wire coverage next door (defer_inert_http_test.go) belongs to #3621 and
// decodes exactly three keys — fak_defer_finding, fak_defer_stand_down_turns, fak_defer_cold_count.
// That is the armed-but-INERT alarm, whose whole subject is a shed that never happened. It cannot
// witness this issue's ask, which is the opposite session: a shed that DID happen, and specifically
// the tool-NAME list, the half of "#3647 shed count + deferred-tool list" that no served-wire test
// asserts today. The producer-level unit test (debug_cache_attribution_test.go) hands
// cacheAttributionVars an AdjudicationSummary with DeferColdToolNames already populated by literal,
// so it proves the block COPIES a name list it was given — never that the live path PRODUCES one.
// A toolDeferNamesSnapshot that returned nil on real traffic would leave both of those tests green
// while the operator saw a bare count with no answer to "which tools went cold", which is the exact
// question the issue exists to answer.
//
// So the replay here is end to end on the real seam: maybeDeferColdTools → the metrics
// accumulators → adjudicationSummary → the /debug/vars document → an HTTP GET off the real handler.
// It decodes into guardvars.CacheAttributionVars — the SHARED wire shape `fak info` decodes
// (cmd/fak/info.go aliases the same type) — so one witness covers both surfaces the done condition
// names, and the pane's Cache tab cannot be reading a different block than this one.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// servedCacheAttributionBlock GETs /debug/vars off a real httptest server bound to srv's real
// handler and decodes the cache_attribution block into the SHARED guardvars shape. Returns the raw
// document alongside it: some assertions below are about the wire KEYS an operator or a scraper
// greps for, which a Go-struct decode would silently satisfy even if the tag drifted.
func servedCacheAttributionBlock(t *testing.T, srv *Server) (string, *guardvars.CacheAttributionVars) {
	t.Helper()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/debug/vars")
	if err != nil {
		t.Fatalf("GET /debug/vars: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /debug/vars status = %d, want 200", resp.StatusCode)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read /debug/vars body: %v", err)
	}
	var doc struct {
		CacheAttribution *guardvars.CacheAttributionVars `json:"cache_attribution"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("decode /debug/vars document: %v\nbody: %s", err, raw)
	}
	return string(raw), doc.CacheAttribution
}

// TestServedDebugVarsCarriesDeferColdToolList is the #3647 done-condition witness on the served
// wire: a defer-ON session's /debug/vars reports how many cold defs were deferred over how many
// turns AND names which tools went cold — and none of it is priced or reported as the compaction
// shed, so the two mechanisms stay separately addressable by anyone scraping the document.
func TestServedDebugVarsCarriesDeferColdToolList(t *testing.T) {
	const turns = 3
	srv := armedHTTPDeferServer(t)
	sum := runDeferTurns(srv, deferBody, turns)
	if sum.DeferColdCount == 0 {
		t.Fatalf("replay deferred nothing — the fixture no longer exercises the lever")
	}

	raw, block := servedCacheAttributionBlock(t, srv)
	if block == nil {
		t.Fatalf("served /debug/vars omitted cache_attribution on a session with a real defer shed\nbody: %s", raw)
	}

	// The COUNT half. deferBody advertises two cold mcp__ tools, so each replayed turn defers two
	// defs: the count is the per-turn total summed across turns, NOT the size of the distinct name
	// set below. Asserting both against the same replay is what keeps that distinction honest — a
	// producer that reported len(names) as the count would pass a bare non-zero check.
	if block.FakDeferColdTurns != turns {
		t.Errorf("served fak_defer_cold_turns = %d, want %d", block.FakDeferColdTurns, turns)
	}
	if want := uint64(2 * turns); block.FakDeferColdCount != want {
		t.Errorf("served fak_defer_cold_count = %d, want %d (2 cold defs × %d turns)",
			block.FakDeferColdCount, want, turns)
	}

	// The LIST half — the part of #3647 with no served-wire witness before this test. Distinct and
	// sorted, so the operator render is stable turn-over-turn rather than a reshuffling sample.
	want := []string{"mcp__dos__dos_verify", "mcp__fak__fak_syscall"}
	got := block.FakDeferColdToolNames
	if len(got) != len(want) {
		t.Fatalf("served fak_defer_cold_tool_names = %v, want %v (distinct, not once per turn)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("served fak_defer_cold_tool_names = %v, want %v (sorted)", got, want)
		}
	}

	// The hot core must never appear in the list: a deferred built-in would be a behavior bug, and
	// naming one here would tell the operator the lever went after tools it must leave eager.
	for _, hot := range []string{"Read", "Bash"} {
		for _, n := range got {
			if n == hot {
				t.Errorf("hot built-in %q reported as deferred; the hot core must stay eager", hot)
			}
		}
	}

	// The CONFLATION guard, which is the reason the issue was filed. Defer shrinks no request bytes
	// and buys no token-equiv — every def still ships and the reduction is provider-side — so on a
	// session where the ONLY mechanism that fired is defer, every token-priced field must stay 0.
	// If a future change ever priced the defer shed into the fak slice, this session would start
	// reporting a compaction shed that never happened, and the three mechanisms would be conflated
	// on the wire exactly as they were on the pane before this leaf.
	if block.FakCompactionShedTokens != 0 {
		t.Errorf("served fak_compaction_shed_tokens = %d on a defer-only session, want 0 — "+
			"the defer shed must not be reported as a compaction shed", block.FakCompactionShedTokens)
	}
	if block.FakTokenEquiv != 0 {
		t.Errorf("served fak_token_equiv = %v on a defer-only session, want 0 — "+
			"defer prices no tokens", block.FakTokenEquiv)
	}

	// A healthy shed raises no watchdog: the #3621 finding keys on the OPPOSITE session, so its
	// token must be absent here or the two readings of "defer" collide on one document.
	if block.FakDeferFinding != "" {
		t.Errorf("served fak_defer_finding = %q on a healthy defer session, want it absent", block.FakDeferFinding)
	}

	// Operators and scrapers grep the JSON NAME, not the Go field, so pin the wire keys themselves.
	for _, key := range []string{"fak_defer_cold_tool_names", "fak_defer_cold_count", "fak_defer_cold_turns"} {
		if !strings.Contains(raw, key) {
			t.Errorf("served /debug/vars body is missing wire key %q: %s", key, raw)
		}
	}

	// The captured block, so `go test -run TestServedDebugVarsCarriesDeferColdToolList -v` is itself
	// the captured-output witness the issue asks for alongside the Cache tab render.
	captured, _ := json.MarshalIndent(block, "", "  ")
	t.Logf("captured /debug/vars cache_attribution (defer-on session):\n%s", captured)
}
