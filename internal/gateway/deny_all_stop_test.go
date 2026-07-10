package gateway

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRecordAdjudicationOutcomeCountsAndResets pins the pure accumulator: a deny-all turn
// bumps both the cumulative count and the consecutive run; any non-deny-all turn resets the
// consecutive run to 0 while leaving the cumulative count intact.
func TestRecordAdjudicationOutcomeCountsAndResets(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))

	m.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "fp-a")
	m.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "fp-a")
	if stops, consec := m.denyAllSnapshot(); stops != 2 || consec != 2 {
		t.Fatalf("after two deny-all turns: stops=%d consec=%d, want 2/2", stops, consec)
	}
	// Same fingerprint both turns -> the same-issue run climbs alongside the blind one.
	if same := m.denyAllSameSnapshot(); same != 2 {
		t.Fatalf("after two identical deny-all turns: same=%d, want 2", same)
	}

	// A non-deny-all turn (a survivor, or a pure-text turn) resets the consecutive run but not
	// the cumulative total.
	m.recordAdjudicationOutcome(adjudicationOutcomeReset, "")
	if stops, consec := m.denyAllSnapshot(); stops != 2 || consec != 0 {
		t.Fatalf("after reset turn: stops=%d consec=%d, want 2/0", stops, consec)
	}
	if same := m.denyAllSameSnapshot(); same != 0 {
		t.Fatalf("after reset turn: same=%d, want 0", same)
	}

	// A fresh deny-all run starts both consecutive counts over from 1.
	m.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "fp-a")
	if stops, consec := m.denyAllSnapshot(); stops != 3 || consec != 1 {
		t.Fatalf("after new deny-all: stops=%d consec=%d, want 3/1", stops, consec)
	}
	if same := m.denyAllSameSnapshot(); same != 1 {
		t.Fatalf("after new deny-all: same=%d, want 1", same)
	}

	// The render surfaces both series, and the summary carries the cumulative count.
	var b strings.Builder
	m.writeDenyAllMetrics(&b)
	out := b.String()
	if !strings.Contains(out, "fak_guard_deny_all_stops_total 3") {
		t.Fatalf("metrics missing stops_total: %s", out)
	}
	if !strings.Contains(out, "fak_guard_deny_all_consecutive 1") {
		t.Fatalf("metrics missing consecutive gauge: %s", out)
	}
	if !strings.Contains(out, "fak_guard_deny_all_same_consecutive 1") {
		t.Fatalf("metrics missing same-issue gauge: %s", out)
	}
	if got := m.adjudicationSummary().DenyAllStops; got != 3 {
		t.Fatalf("summary DenyAllStops = %d, want 3", got)
	}
}

// TestDenyAllSameConsecutiveFold pins the same-issue accumulator distinct from the blind one:
// the SAME fingerprint turn after turn climbs, a CHANGED fingerprint re-seeds to 1 (so a varied
// session never accumulates toward a stop) while the blind run keeps climbing, an empty
// fingerprint fails open (never climbs past 1), and any non-deny-all turn zeroes it.
func TestDenyAllSameConsecutiveFold(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))

	// Six identical refusals in a row -> same-issue run reaches 6 (the default give-up depth).
	for i := 0; i < 6; i++ {
		m.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "Write\x1fSELF_MODIFY")
	}
	if same := m.denyAllSameSnapshot(); same != 6 {
		t.Fatalf("six identical deny-all turns: same=%d, want 6", same)
	}
	if _, consec := m.denyAllSnapshot(); consec != 6 {
		t.Fatalf("six deny-all turns: blind consec=%d, want 6", consec)
	}

	// A DIFFERENT issue every turn: the blind run keeps climbing but the same-issue run is pinned
	// at 1 — the whole point, so an exploring session is never given up.
	m2 := newGatewayMetrics(time.Unix(0, 0))
	for i, fp := range []string{"A\x1fMISROUTE", "B\x1fLEASE_HELD", "C\x1fSELF_MODIFY", "D\x1fMISROUTE"} {
		m2.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, fp)
		if same := m2.denyAllSameSnapshot(); same != 1 {
			t.Fatalf("varied turn %d (%s): same=%d, want 1 (a varied session must never climb)", i, fp, same)
		}
	}
	if _, consec := m2.denyAllSnapshot(); consec != 4 {
		t.Fatalf("four varied deny-all turns: blind consec=%d, want 4 (blind still climbs)", consec)
	}

	// An empty fingerprint (nothing fingerprintable) fails open: it never climbs past 1, even
	// when repeated, so an unidentifiable turn cannot drive a give-up.
	m3 := newGatewayMetrics(time.Unix(0, 0))
	m3.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "")
	m3.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "")
	if same := m3.denyAllSameSnapshot(); same != 1 {
		t.Fatalf("two empty-fingerprint deny-all turns: same=%d, want 1 (fail-open, never accumulates)", same)
	}

	// A tool-feedback turn zeroes the same-issue run just like the blind one.
	m.recordAdjudicationOutcome(adjudicationOutcomeToolFeedback, "")
	if same := m.denyAllSameSnapshot(); same != 0 {
		t.Fatalf("after tool-feedback turn: same=%d, want 0", same)
	}
}

// TestDenyAllFingerprintIdentity pins the fingerprint's identity rules: it is order- and
// duplicate-insensitive over (tool, reason) pairs, distinguishes a different tool or reason,
// excludes admitted/allowed entries, and is empty when nothing is fingerprintable.
func TestDenyAllFingerprintIdentity(t *testing.T) {
	deny := func(tool, reason string) ToolAdjudication {
		return ToolAdjudication{Tool: tool, Admitted: false, Verdict: WireVerdict{Kind: "DENY", Reason: reason}}
	}

	// Order- and duplicate-insensitive: same set of (tool, reason) pairs -> same fingerprint.
	a := denyAllFingerprint([]ToolAdjudication{deny("Write", "SELF_MODIFY"), deny("Bash", "LEASE_HELD")})
	b := denyAllFingerprint([]ToolAdjudication{deny("Bash", "LEASE_HELD"), deny("Write", "self_modify"), deny("Write", "SELF_MODIFY")})
	if a == "" || a != b {
		t.Fatalf("fingerprint not order/dup/case-insensitive: a=%q b=%q", a, b)
	}

	// A different reason (or tool) is a different issue.
	if denyAllFingerprint([]ToolAdjudication{deny("Write", "SELF_MODIFY")}) ==
		denyAllFingerprint([]ToolAdjudication{deny("Write", "MISROUTE")}) {
		t.Fatal("different reason must yield a different fingerprint")
	}

	// Admitted / allowed entries do not contribute to the issue identity.
	admitted := ToolAdjudication{Tool: "Read", Admitted: true, Verdict: WireVerdict{Kind: "ALLOW"}}
	if got := denyAllFingerprint([]ToolAdjudication{admitted, deny("Write", "SELF_MODIFY")}); got != denyAllFingerprint([]ToolAdjudication{deny("Write", "SELF_MODIFY")}) {
		t.Fatalf("admitted entry leaked into fingerprint: %q", got)
	}

	// Nothing fingerprintable -> empty (fail-open at the fold).
	if got := denyAllFingerprint([]ToolAdjudication{{Tool: "", Verdict: WireVerdict{Kind: "DENY"}}}); got != "" {
		t.Fatalf("unfingerprintable turn: got %q, want empty", got)
	}
}

func TestRecordAdjudicationOutcomeSeparatesRetryableToolFeedback(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))

	for i := 0; i < 4; i++ {
		m.recordAdjudicationOutcome(adjudicationOutcomeToolFeedback, "")
	}
	if stops, consec := m.denyAllSnapshot(); stops != 0 || consec != 0 {
		t.Fatalf("malformed feedback counted as hard deny-all: stops=%d consec=%d, want 0/0", stops, consec)
	}
	if turns, consec := m.toolFeedbackSnapshot(); turns != 4 || consec != 4 {
		t.Fatalf("tool feedback snapshot = %d/%d, want 4/4", turns, consec)
	}

	var b strings.Builder
	m.writeDenyAllMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_guard_deny_all_stops_total 0",
		"fak_guard_deny_all_consecutive 0",
		"fak_guard_tool_feedback_turns_total 4",
		"fak_guard_tool_feedback_consecutive 4",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing %q:\n%s", want, out)
		}
	}
}

func TestMalformedToolCallsDriveFeedbackNotDenyAllStop(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))
	malformed := []ToolAdjudication{{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "MALFORMED", Disposition: "RETRYABLE"},
	}}

	for i := 0; i < 4; i++ {
		m.recordAdjudicationOutcome(adjudicationOutcomeForTurn(malformed, 0, 0), "")
	}
	if stops, consec := m.denyAllSnapshot(); stops != 0 || consec != 0 {
		t.Fatalf("four malformed tool-call turns became session-stop signal: denyAll=%d/%d, want 0/0", stops, consec)
	}
	if turns, consec := m.toolFeedbackSnapshot(); turns != 4 || consec != 4 {
		t.Fatalf("four malformed tool-call turns did not stay feedback: feedback=%d/%d, want 4/4", turns, consec)
	}
}

// TestNilMetricsRecordAdjudicationOutcomeNoPanic guards the nil-receiver contract the other
// observe methods hold: a Server built without metrics must not panic on the hot path.
func TestNilMetricsRecordAdjudicationOutcomeNoPanic(t *testing.T) {
	var m *gatewayMetrics
	m.recordAdjudicationOutcome(adjudicationOutcomeDenyAll, "fp") // must be a no-op, not a nil deref
	if stops, consec := m.denyAllSnapshot(); stops != 0 || consec != 0 {
		t.Fatalf("nil snapshot = %d/%d, want 0/0", stops, consec)
	}
	if same := m.denyAllSameSnapshot(); same != 0 {
		t.Fatalf("nil same snapshot = %d, want 0", same)
	}
}

// TestAnthropicStreamDenyAllCountsDenyAllStop is the end-to-end witness: an all-denied
// STREAMED turn (the flagship passthrough path) both rewrites stop_reason to end_turn AND
// records exactly one deny-all stop in the adjudication summary — so the otherwise-invisible
// "fak ended the turn" is counted, which is what the guard Stop-hook resumes the agent past.
func TestAnthropicStreamDenyAllCountsDenyAllStop(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,` +
		`"tools":[{"name":"deny_b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"go"}]}`)

	upstreamSSE := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":2,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"d1","name":"deny_b","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, upstreamSSE)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	// Drain the stream so the turn fully completes (the deny-all fold happens at message_delta).
	frames := readAnthropicSSE(t, httpResp.Body)
	_ = httpResp.Body.Close()
	if len(frames) == 0 {
		t.Fatalf("no SSE frames")
	}

	if got := srv.AdjudicationSummary().DenyAllStops; got != 1 {
		t.Fatalf("DenyAllStops = %d, want 1 (the all-denied streamed turn must count exactly one deny-all stop)", got)
	}
	// And the gauges the Stop-hook polls read 1 after the single deny-all turn: the blind
	// consecutive AND the same-issue gauge it now keys its give-up on.
	rendered := srv.renderMetrics()
	if !strings.Contains(rendered, "fak_guard_deny_all_consecutive 1") {
		t.Fatalf("metrics did not show one consecutive deny-all turn")
	}
	if !strings.Contains(rendered, "fak_guard_deny_all_same_consecutive 1") {
		t.Fatalf("metrics did not show one same-issue deny-all turn")
	}
}
