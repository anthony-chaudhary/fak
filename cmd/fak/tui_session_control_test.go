package main

// tui_session_control_test.go — the render-witness for the OOB control keybindings
// (#2763). The witness the issue names is "a keypress -> the emitted control request,
// captured, not live-dialed": planTUISessionControlKey is the pure seam, so the core
// tests feed a key and assert plan.Request WITHOUT any gateway. A second layer drives the
// full `fak console sessions --press` path against the in-package stubGateway to prove the
// keybinding actually issues the control op over the session-control route (verb + run
// captured server-side), and that the destructive-confirmation gate holds end to end.

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func controlRowFixture() tuiSessionRow {
	return tuiSessionRow{TraceID: "urgent", Run: "running", Rev: 3}
}

// TestTUISessionControlKeyEmitsRunRequest is the core witness: each non-destructive key
// maps to the exact control request it emits, captured from the pure planner, no dialing.
func TestTUISessionControlKeyEmitsRunRequest(t *testing.T) {
	row := controlRowFixture()
	cases := []struct {
		key      rune
		wantRun  string
		wantReas string
	}{
		{'p', "paused", ""},
		{'r', "running", ""},
		{'t', "throttled", "operator-tui-throttle"},
	}
	for _, tc := range cases {
		plan, ok := planTUISessionControlKey(row, tc.key, false)
		if !ok {
			t.Fatalf("key %q not bound", string(tc.key))
		}
		if !plan.Emit {
			t.Fatalf("key %q did not emit (plan=%+v)", string(tc.key), plan)
		}
		if plan.Verb != "run" {
			t.Fatalf("key %q verb = %q, want run", string(tc.key), plan.Verb)
		}
		if plan.Request.Run != tc.wantRun {
			t.Fatalf("key %q run = %q, want %q", string(tc.key), plan.Request.Run, tc.wantRun)
		}
		if plan.Request.Reason != tc.wantReas {
			t.Fatalf("key %q reason = %q, want %q", string(tc.key), plan.Request.Reason, tc.wantReas)
		}
		if plan.TraceID != row.TraceID {
			t.Fatalf("key %q trace = %q, want %q", string(tc.key), plan.TraceID, row.TraceID)
		}
	}
}

// TestTUISessionControlDrainRequiresConfirm proves the destructive-op gate: an unconfirmed
// drain is captured but WITHHELD (no request emitted); a confirmed drain emits run=draining.
func TestTUISessionControlDrainRequiresConfirm(t *testing.T) {
	row := controlRowFixture()

	unconfirmed, ok := planTUISessionControlKey(row, 'd', false)
	if !ok {
		t.Fatal("drain key not bound")
	}
	if !unconfirmed.Destructive || !unconfirmed.NeedsConfirm {
		t.Fatalf("drain unconfirmed = %+v, want destructive+needs-confirm", unconfirmed)
	}
	if unconfirmed.Emit || unconfirmed.Request.Run != "" {
		t.Fatalf("drain unconfirmed emitted a request %+v — must be withheld", unconfirmed.Request)
	}

	confirmed, ok := planTUISessionControlKey(row, 'd', true)
	if !ok {
		t.Fatal("drain key not bound (confirmed)")
	}
	if !confirmed.Emit || confirmed.Request.Run != "draining" {
		t.Fatalf("drain confirmed = %+v, want emit run=draining", confirmed)
	}
	if confirmed.Request.Reason != "operator-tui-drain" {
		t.Fatalf("drain reason = %q, want operator-tui-drain", confirmed.Request.Reason)
	}
}

// TestTUISessionControlRedirectDeferred proves the redirect key is declared but inert until
// its sibling op lands: it is a known binding that emits nothing and names the blocker.
func TestTUISessionControlRedirectDeferred(t *testing.T) {
	plan, ok := planTUISessionControlKey(controlRowFixture(), '>', true)
	if !ok {
		t.Fatal("redirect key not bound")
	}
	if plan.Emit {
		t.Fatalf("redirect emitted a request %+v — must be deferred", plan.Request)
	}
	if plan.Deferred == "" || !strings.Contains(plan.Deferred, "#2753") {
		t.Fatalf("redirect deferred reason = %q, want it to name the blocking child", plan.Deferred)
	}
}

// TestTUISessionControlUnboundKey proves an unbound key does nothing (fail-closed).
func TestTUISessionControlUnboundKey(t *testing.T) {
	if _, ok := planTUISessionControlKey(controlRowFixture(), 'z', true); ok {
		t.Fatal("key 'z' should not be bound")
	}
}

// TestTUISessionControlDispatchWiresRoute proves the emit-ready plan is dispatched through
// the real session-control client route (captured server-side by the stub, not a mock of
// the client): pressing p POSTs {id}/run with run=paused.
func TestTUISessionControlDispatchWiresRoute(t *testing.T) {
	g := &stubGateway{curRev: 3}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	c := &sessionClient{base: ts.URL, hc: ts.Client()}
	plan, _ := planTUISessionControlKey(controlRowFixture(), 'p', false)
	st, err := dispatchTUISessionControl(c, plan)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if g.lastVerb != "run" {
		t.Fatalf("route verb = %q, want run", g.lastVerb)
	}
	if g.lastBody.Run != "paused" {
		t.Fatalf("route run = %q, want paused", g.lastBody.Run)
	}
	if st.Run != "paused" {
		t.Fatalf("returned state run = %q, want paused", st.Run)
	}
}

// TestTUISessionControlDispatchRefusesWithheldPlan proves dispatch never re-decides the
// gate: a needs-confirm plan is refused without any POST.
func TestTUISessionControlDispatchRefusesWithheldPlan(t *testing.T) {
	g := &stubGateway{curRev: 3}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	c := &sessionClient{base: ts.URL, hc: ts.Client()}
	plan, _ := planTUISessionControlKey(controlRowFixture(), 'd', false) // unconfirmed drain
	if _, err := dispatchTUISessionControl(c, plan); err == nil {
		t.Fatal("dispatch of an unconfirmed drain must refuse, got nil error")
	}
	if len(g.verbs) != 0 {
		t.Fatalf("a withheld plan POSTed to the route: %v", g.verbs)
	}
}

// TestTUISessionControlKeyEndToEnd drives the operator surface — `fak console sessions
// --press p --session urgent` — against the stub gateway, proving the CLI resolves the
// session, plans, and dispatches the right control op over the live route.
func TestTUISessionControlKeyEndToEnd(t *testing.T) {
	g := &stubGateway{curRev: 3}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	var stdout, stderr strings.Builder
	code := runTUI(&stdout, &stderr, []string{
		"sessions", "--addr", ts.URL, "--press", "p", "--session", "urgent",
	})
	if code != 0 {
		t.Fatalf("runTUI sessions --press p code=%d stderr=%s", code, stderr.String())
	}
	if g.lastVerb != "run" || g.lastBody.Run != "paused" {
		t.Fatalf("end-to-end emitted verb=%q run=%q, want run/paused", g.lastVerb, g.lastBody.Run)
	}
	if !strings.Contains(stdout.String(), "pause -> urgent") {
		t.Fatalf("output missing control confirmation:\n%s", stdout.String())
	}
}

// TestTUISessionControlKeyDrainGateEndToEnd proves the confirmation gate holds through the
// CLI: `--press d` without --confirm refuses (no POST); with --confirm it drains.
func TestTUISessionControlKeyDrainGateEndToEnd(t *testing.T) {
	g := &stubGateway{curRev: 3}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	var out1, err1 strings.Builder
	code := runTUI(&out1, &err1, []string{
		"sessions", "--addr", ts.URL, "--press", "d", "--session", "urgent",
	})
	if code == 0 {
		t.Fatalf("drain without --confirm should refuse, got code 0:\n%s", out1.String())
	}
	if len(g.verbs) != 0 {
		t.Fatalf("drain without --confirm POSTed to the route: %v", g.verbs)
	}
	if !strings.Contains(err1.String(), "--confirm") {
		t.Fatalf("refusal should name --confirm, got: %s", err1.String())
	}

	var out2, err2 strings.Builder
	code = runTUI(&out2, &err2, []string{
		"sessions", "--addr", ts.URL, "--press", "d", "--session", "urgent", "--confirm",
	})
	if code != 0 {
		t.Fatalf("drain with --confirm code=%d stderr=%s", code, err2.String())
	}
	if g.lastVerb != "run" || g.lastBody.Run != "draining" {
		t.Fatalf("confirmed drain emitted verb=%q run=%q, want run/draining", g.lastVerb, g.lastBody.Run)
	}
}

// TestTUISessionControlLegendRendersInPane proves the pane advertises the control surface:
// the rendered sessions view carries the keybinding legend.
func TestTUISessionControlLegendRendersInPane(t *testing.T) {
	at, err := time.Parse(time.RFC3339, "2026-06-25T12:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	report := buildTUISessionReport(gateway.SessionListResponse{
		Count:    1,
		Sessions: []gateway.SessionState{{TraceID: "urgent", Run: "running", Budget: gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1}, Rev: 1}},
	}, "fixture", at)
	out := renderTUISessions(report, 25, 120)
	for _, want := range []string{"Controls", "p pause", "d drain (confirm)", "> redirect (pending)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("legend missing %q:\n%s", want, out)
		}
	}
}
