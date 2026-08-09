package gateway

// defer_inert_http_test.go — the #3621 httptest REPLAY witness, the artifact the issue's
// Witness line names ("captured live-session artifact (or httptest replay) showing the
// finding"). Its sibling defer_inert_test.go proves the FOLD: that an armed-but-inert
// session raises DEFER_ENABLED_BUT_INERT and that cacheAttributionVars renders it. What no
// test proved is that the finding survives the trip an operator actually takes — a GET of
// /debug/vars against a live gateway — because those tests call cacheAttributionVars
// directly, while the served document builds its block deep inside Server.debugVars
// (debug.go: CacheAttribution: cacheAttributionVars(m.adjudicationSummary(), c.VDSOHits,
// m.servedInlineSnapshot())). A change that dropped the block from the document, gated it
// behind another emit condition, renamed the wire key, or lost the auth-path plumbing would
// leave every existing #3621 test green while the alarm never reached a single operator.
//
// So this file replays turns through the REAL transform seam on a REAL server, serves that
// server's REAL handler over httptest, and reads the finding out of the decoded HTTP
// response body. Nothing is hand-set: the only way the token appears is if the transform
// ran, stood down, and the whole /debug/vars pipeline carried the verdict to the wire.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// armedHTTPDeferServer builds the FULL server /debug/vars is served off (unlike
// armedDefaultDeferServer's bare struct, which has no handler), armed exactly the way
// `fak serve --defer-cold-tools` arms it: the lever on plus an Anthropic passthrough wire,
// the two halves of maybeDeferColdTools' eligibility gate.
func armedHTTPDeferServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	srv.deferColdTools = true
	srv.planner = &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}
	return srv
}

// debugVarsCacheAttribution GETs /debug/vars off a real httptest server bound to srv's real
// handler and returns the raw response body plus the decoded cache_attribution block (nil
// when the document omitted it). Decoding by WIRE KEY, not by Go struct, is the point: the
// operator pane and `fak info` read these JSON names.
func debugVarsCacheAttribution(t *testing.T, srv *Server) (string, *struct {
	Finding        string `json:"fak_defer_finding"`
	StandDownTurns uint64 `json:"fak_defer_stand_down_turns"`
	ColdCount      uint64 `json:"fak_defer_cold_count"`
}) {
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
		CacheAttribution *struct {
			Finding        string `json:"fak_defer_finding"`
			StandDownTurns uint64 `json:"fak_defer_stand_down_turns"`
			ColdCount      uint64 `json:"fak_defer_cold_count"`
		} `json:"cache_attribution"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("decode /debug/vars document: %v\nbody: %s", err, raw)
	}
	return string(raw), doc.CacheAttribution
}

// The replay: an armed session whose every turn stands down to identity surfaces
// DEFER_ENABLED_BUT_INERT in the body an operator GETs from a running gateway. This is the
// issue's done-condition read end to end — transform seam → counters → summary → document →
// HTTP wire — with the denominator (stand-down turns) carried alongside so the operator can
// see HOW MANY eligible turns produced nothing.
func TestDeferEnabledButInertSurfacesOnServedDebugVars(t *testing.T) {
	srv := armedHTTPDeferServer(t)
	if got := runDeferTurns(srv, allHotBody, deferInertMinTurns); !got.DeferEnabledButInert() {
		t.Fatalf("replay did not reach the inert state (cold=%d, standdown=%d)", got.DeferColdCount, got.DeferStandDownTurns)
	}

	raw, block := debugVarsCacheAttribution(t, srv)
	if block == nil {
		t.Fatalf("served /debug/vars omitted cache_attribution on an inert defer session — the alarm has no carrier\nbody: %s", raw)
	}
	if block.Finding != guardvars.FindingDeferEnabledButInert {
		t.Errorf("served cache_attribution.fak_defer_finding = %q, want %q", block.Finding, guardvars.FindingDeferEnabledButInert)
	}
	if block.ColdCount != 0 {
		t.Errorf("served fak_defer_cold_count = %d, want 0 — an inert session deferred nothing", block.ColdCount)
	}
	if block.StandDownTurns != uint64(deferInertMinTurns) {
		t.Errorf("served fak_defer_stand_down_turns = %d, want %d — the denominator must ride with the finding",
			block.StandDownTurns, deferInertMinTurns)
	}
	if !strings.Contains(raw, guardvars.FindingDeferEnabledButInert) {
		t.Errorf("served /debug/vars body does not contain the finding token anywhere: %s", raw)
	}
}

// The negative control on the same wire: a HEALTHY defer session must return a document with
// no finding token in it at all. The field is omitempty, so its ABSENCE from the served bytes
// is the healthy signal — an operator (or a scraper) keying on the token must never see it
// fire on a lever that is working.
func TestServedDebugVarsQuietOnHealthyDeferSession(t *testing.T) {
	srv := armedHTTPDeferServer(t)
	sum := runDeferTurns(srv, deferBody, deferInertMinTurns+2)
	if sum.DeferColdCount == 0 {
		t.Fatalf("healthy replay deferred nothing — fixture no longer exercises the lever")
	}

	raw, block := debugVarsCacheAttribution(t, srv)
	if block == nil {
		t.Fatalf("served /debug/vars omitted cache_attribution on a session with a real defer shed\nbody: %s", raw)
	}
	if block.Finding != "" {
		t.Errorf("healthy session served fak_defer_finding = %q, want it absent", block.Finding)
	}
	if block.ColdCount == 0 {
		t.Errorf("served fak_defer_cold_count = 0 on a healthy defer session; the shed must be visible")
	}
	if strings.Contains(raw, guardvars.FindingDeferEnabledButInert) {
		t.Errorf("healthy session's served /debug/vars carries the finding token: %s", raw)
	}
}

// FAIL-SAFE on the wire: a session whose lever was never armed can never serve the finding,
// no matter how many turns run a body that WOULD have deferred. Nothing accrues past the
// eligibility gate, so the watchdog cannot invent an alarm about a lever nobody turned on —
// and an operator who left --defer-cold-tools off sees an unchanged document.
func TestServedDebugVarsNeverRaisesInertWhenLeverOff(t *testing.T) {
	srv := armedHTTPDeferServer(t)
	srv.deferColdTools = false
	sum := runDeferTurns(srv, deferBody, deferInertMinTurns+2)
	if sum.DeferAttempts() != 0 {
		t.Fatalf("lever-off session booked %d eligible turn(s); want 0", sum.DeferAttempts())
	}

	raw, block := debugVarsCacheAttribution(t, srv)
	if block != nil && block.Finding != "" {
		t.Errorf("lever-off session served fak_defer_finding = %q", block.Finding)
	}
	if strings.Contains(raw, guardvars.FindingDeferEnabledButInert) {
		t.Errorf("lever-off session's served /debug/vars carries the finding token: %s", raw)
	}
}
