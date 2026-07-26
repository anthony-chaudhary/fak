package gateway

// defer_inert_test.go — the #3621 witness for the DEFER_ENABLED_BUT_INERT live monitor.
//
// Every case here drives the REAL server seam (Server.maybeDeferColdTools) turn by turn — the
// same call messages_transform.go makes on a live request — and then reads the finding off the
// two surfaces the issue's done-condition names: the session summary the guard exit banner folds
// (AdjudicationSummary) and the /debug/vars cache_attribution block (cacheAttributionVars). No
// counter is hand-set, so the test cannot pass on a transform that never ran.
//
// The property under test is the one the pre-#3621 counters could not express: cold_total==0 is
// AMBIGUOUS on its own — a lever left off and a lever armed-but-never-biting both render a flat
// zero. The stand-down denominator disambiguates them, and only the armed-and-inert half raises.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// allHotBody is a Claude-Code-shaped body whose every advertised tool is in defaultHotToolSet —
// the transform runs and stands down with reason "no_cold_tools". This is the BENIGN inert case.
const allHotBody = `{"model":"claude-x","system":"SYS-PROMPT","messages":[{"role":"user","content":"hello"}],` +
	`"tools":[` +
	`{"name":"Read","description":"read a file","input_schema":{"type":"object"}},` +
	`{"name":"Bash","description":"run a command","input_schema":{"type":"object"}}` +
	`]}`

// preDeferredBody already carries a tool_search_tool, so the idempotent stand-down fires with
// reason "already_deferred" — the Claude Code ENABLE_TOOL_SEARCH case, where fak's own lever is
// armed and contributes nothing.
const preDeferredBody = `{"model":"claude-x","system":"SYS-PROMPT","messages":[{"role":"user","content":"hello"}],` +
	`"tools":[` +
	`{"name":"Read","description":"read a file","input_schema":{"type":"object"}},` +
	`{"name":"mcp__fak__fak_syscall","description":"adjudicate","input_schema":{"type":"object"},"defer_loading":true},` +
	`{"type":"tool_search_tool_20250917","name":"tool_search_tool"}` +
	`]}`

// runDeferTurns replays n turns of body through the real seam on srv and returns the resulting
// session summary — the exact object the guard exit banner and the ledger row fold.
func runDeferTurns(srv *Server, body string, n int) AdjudicationSummary {
	for i := 0; i < n; i++ {
		req := &agent.AnthropicMessagesRequest{Model: "claude-x", Raw: []byte(body)}
		srv.maybeDeferColdTools(req, "trace-inert")
	}
	return srv.metrics.adjudicationSummary()
}

// deferFindingOnDebugVars renders the /debug/vars cache_attribution block for sum and returns the
// decoded fak_defer_finding wire field, proving the alarm survives JSON encoding (an operator and
// the `fak info` pane read the field, not the Go struct). "" means the block was omitted entirely.
func deferFindingOnDebugVars(t *testing.T, sum AdjudicationSummary) string {
	t.Helper()
	block := cacheAttributionVars(sum, 0, 0)
	if block == nil {
		return ""
	}
	raw, err := json.Marshal(block)
	if err != nil {
		t.Fatalf("marshal cache_attribution block: %v", err)
	}
	var decoded struct {
		Finding string `json:"fak_defer_finding"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode cache_attribution block: %v", err)
	}
	return decoded.Finding
}

// The core done-condition: an armed session that defers 0 cold tools across enough turns raises
// DEFER_ENABLED_BUT_INERT, on the summary AND on /debug/vars, with the stand-down reason named.
func TestDeferEnabledButInertRaisesOnArmedZeroDeferSession(t *testing.T) {
	srv := armedDefaultDeferServer()
	sum := runDeferTurns(srv, allHotBody, deferInertMinTurns)

	if sum.DeferColdCount != 0 {
		t.Fatalf("all-hot body deferred %d cold def(s); want 0", sum.DeferColdCount)
	}
	if got := sum.DeferStandDownTurns; got != uint64(deferInertMinTurns) {
		t.Fatalf("stand-down turns=%d, want %d — the denominator must accrue past the eligibility gate", got, deferInertMinTurns)
	}
	if got := sum.DeferStandDownReasons["no_cold_tools"]; got != uint64(deferInertMinTurns) {
		t.Errorf("stand-down reason no_cold_tools=x%d, want x%d (breakdown: %v)", got, deferInertMinTurns, sum.DeferStandDownReasons)
	}
	if got := sum.DeferAttempts(); got != uint64(deferInertMinTurns) {
		t.Errorf("DeferAttempts()=%d, want %d", got, deferInertMinTurns)
	}
	if !sum.DeferEnabledButInert() {
		t.Fatalf("armed session with 0 cold defers across %d eligible turns did not raise the finding", deferInertMinTurns)
	}
	if got := deferFindingOnDebugVars(t, sum); got != guardvars.FindingDeferEnabledButInert {
		t.Errorf("/debug/vars cache_attribution.fak_defer_finding=%q, want %q", got, guardvars.FindingDeferEnabledButInert)
	}
}

// The other half of the done-condition: a HEALTHY defer session must stay quiet.
func TestDeferEnabledButInertQuietOnHealthyDeferSession(t *testing.T) {
	srv := armedDefaultDeferServer()
	sum := runDeferTurns(srv, deferBody, deferInertMinTurns+2)

	if sum.DeferColdCount == 0 {
		t.Fatalf("healthy defer body deferred nothing — fixture no longer exercises the lever")
	}
	if sum.DeferStandDownTurns != 0 {
		t.Errorf("healthy session booked %d stand-down turn(s); want 0", sum.DeferStandDownTurns)
	}
	if sum.DeferEnabledButInert() {
		t.Fatalf("healthy defer session (cold=%d over %d turns) raised DEFER_ENABLED_BUT_INERT",
			sum.DeferColdCount, sum.DeferColdTurns)
	}
	if got := deferFindingOnDebugVars(t, sum); got != "" {
		t.Errorf("healthy session carries fak_defer_finding=%q; the field's presence is the alarm, so it must be absent", got)
	}
}

// A single fired deferral clears the finding for the session's lifetime, even when stand-downs
// dominate: the lever demonstrably BIT, so it is not inert — only unlucky on the other turns.
func TestDeferEnabledButInertClearedByOneFiredDeferral(t *testing.T) {
	srv := armedDefaultDeferServer()
	runDeferTurns(srv, allHotBody, deferInertMinTurns+3)
	sum := runDeferTurns(srv, deferBody, 1)

	if sum.DeferColdCount == 0 {
		t.Fatalf("mixed session never fired; fixture broken")
	}
	if sum.DeferAttempts() <= uint64(deferInertMinTurns) {
		t.Fatalf("DeferAttempts()=%d must exceed the floor for this case to be meaningful", sum.DeferAttempts())
	}
	if sum.DeferEnabledButInert() {
		t.Fatalf("one fired deferral did not clear the finding (cold=%d, standdown=%d)",
			sum.DeferColdCount, sum.DeferStandDownTurns)
	}
}

// Below the eligible-turn floor a zero-defer session is merely SHORT, not inert. A watchdog that
// fired on turn one would alarm on every cold start.
func TestDeferEnabledButInertQuietBelowTurnFloor(t *testing.T) {
	srv := armedDefaultDeferServer()
	sum := runDeferTurns(srv, allHotBody, deferInertMinTurns-1)

	if sum.DeferStandDownTurns != uint64(deferInertMinTurns-1) {
		t.Fatalf("stand-down turns=%d, want %d", sum.DeferStandDownTurns, deferInertMinTurns-1)
	}
	if sum.DeferEnabledButInert() {
		t.Fatalf("a %d-turn session tripped the finding; the floor is %d", deferInertMinTurns-1, deferInertMinTurns)
	}
}

// The already-deferred (client ENABLE_TOOL_SEARCH) path is inert for FAK's lever too, and names
// itself in the breakdown so an operator can tell it apart from a body fak could not prove.
func TestDeferEnabledButInertNamesAlreadyDeferredStandDown(t *testing.T) {
	srv := armedDefaultDeferServer()
	sum := runDeferTurns(srv, preDeferredBody, deferInertMinTurns)

	if !sum.DeferEnabledButInert() {
		t.Fatalf("armed session standing down on an already-deferred body did not raise the finding")
	}
	if got := sum.DeferStandDownReasons["already_deferred"]; got != uint64(deferInertMinTurns) {
		t.Errorf("reason already_deferred=x%d, want x%d (breakdown: %v)", got, deferInertMinTurns, sum.DeferStandDownReasons)
	}
}

// FAIL-SAFE: the finding is unreachable on any session where the lever was not actually armed on
// a live Anthropic wire. Each of these stands down BEFORE the eligibility gate, so no denominator
// accrues and the watchdog can never invent an alarm about a lever nobody turned on.
func TestDeferEnabledButInertUnreachableWhenLeverNotArmed(t *testing.T) {
	turns := deferInertMinTurns + 2
	cases := []struct {
		name string
		srv  func(t *testing.T) *Server
	}{
		{
			name: "lever off (--defer-cold-tools=false)",
			srv: func(t *testing.T) *Server {
				s := armedDefaultDeferServer()
				s.deferColdTools = false
				return s
			},
		},
		{
			name: "ablation arm (FAK_ABLATE_DEFER_TOOLS=1)",
			srv: func(t *testing.T) *Server {
				t.Setenv("FAK_ABLATE_DEFER_TOOLS", "1")
				return armedDefaultDeferServer()
			},
		},
		{
			name: "non-Anthropic wire",
			srv: func(t *testing.T) *Server {
				s := armedDefaultDeferServer()
				s.planner = &agent.HTTPPlanner{Provider: agent.ProviderOpenAI}
				return s
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Use the body that WOULD defer, so only the gate can explain a zero.
			sum := runDeferTurns(tc.srv(t), deferBody, turns)
			if sum.DeferStandDownTurns != 0 {
				t.Errorf("booked %d stand-down turn(s) before the eligibility gate; the denominator must stay 0", sum.DeferStandDownTurns)
			}
			if sum.DeferAttempts() != 0 {
				t.Errorf("DeferAttempts()=%d, want 0 — nothing was eligible", sum.DeferAttempts())
			}
			if sum.DeferEnabledButInert() {
				t.Fatalf("raised DEFER_ENABLED_BUT_INERT on a session where the lever was never armed")
			}
			if got := deferFindingOnDebugVars(t, sum); got != "" {
				t.Errorf("/debug/vars carries fak_defer_finding=%q on an unarmed session", got)
			}
		})
	}
}

// The /debug/vars block must RENDER on an inert session. Its pre-#3621 emit gate suppressed the
// block whenever there was no token slice, no VDSO call and no defer shed — which is exactly the
// shape of an inert defer session, so the alarm would have been swallowed by its own carrier.
func TestDeferInertBlockRendersOnOtherwiseQuietSession(t *testing.T) {
	srv := armedDefaultDeferServer()
	sum := runDeferTurns(srv, allHotBody, deferInertMinTurns)

	block := cacheAttributionVars(sum, 0, 0)
	if block == nil {
		t.Fatalf("cache_attribution block omitted on an inert defer session — the finding has no carrier")
	}
	if block.FakDeferStandDownTurns != uint64(deferInertMinTurns) {
		t.Errorf("block fak_defer_stand_down_turns=%d, want %d", block.FakDeferStandDownTurns, deferInertMinTurns)
	}
	raw, err := json.Marshal(block)
	if err != nil {
		t.Fatalf("marshal block: %v", err)
	}
	if !strings.Contains(string(raw), guardvars.FindingDeferEnabledButInert) {
		t.Errorf("rendered block does not carry the finding token: %s", raw)
	}
}

// The scrape surface carries the same denominator, so the finding is reproducible from /metrics
// alone (cold_total==0 while standdown_turns_total climbs) and not only from the two Go surfaces.
func TestDeferStandDownCounterRendersOnMetrics(t *testing.T) {
	srv := armedDefaultDeferServer()
	runDeferTurns(srv, allHotBody, deferInertMinTurns)

	turns, reasons := srv.metrics.toolDeferStandDownSnapshot()
	if turns != uint64(deferInertMinTurns) {
		t.Fatalf("snapshot turns=%d, want %d", turns, deferInertMinTurns)
	}
	// The snapshot must hand back a COPY: mutating it can never corrupt the live accumulator.
	reasons["no_cold_tools"] = 999
	if _, again := srv.metrics.toolDeferStandDownSnapshot(); again["no_cold_tools"] != uint64(deferInertMinTurns) {
		t.Errorf("mutating the snapshot leaked into the live reason map: %v", again)
	}
}
