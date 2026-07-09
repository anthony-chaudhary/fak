package microagent_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// countingGateway is a pure in-process base gateway for the SessionGateway tests:
// it counts dials and returns a completion carrying a fixed Usage. It opens NO
// socket and speaks NO HTTP — the direct-call contrast to microagent_test.go's
// chatPlanner, which dials an httptest server. Every dial here is a plain Go method
// call, so a turn the session gate REFUSES costs zero dials, observably.
type countingGateway struct {
	dials atomic.Int64
	usage agent.Usage
}

func (p *countingGateway) Model() string { return "count" }

func (p *countingGateway) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.dials.Add(1)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, Usage: p.usage}, nil
}

// tracedTurnAgent finishes after `turns` model turns, each taken through the SHARED
// gateway the host hands it — tagging the call context with its own id (== its
// session.Table TraceID) so the session-adjudicating gateway drives THIS agent's
// row. It holds no gateway of its own.
type tracedTurnAgent struct {
	id    string
	turns int
	took  int
}

func (a *tracedTurnAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.took++
	ctx = microagent.WithTrace(ctx, a.id)
	msg := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("agent %s turn %d", a.id, a.took)}}
	if _, err := gw.Complete(ctx, msg, nil); err != nil {
		return false, err
	}
	return a.took >= a.turns, nil
}

// TestSessionGatewayGatesTurnBudgetBeforeDial pins #2005 scope 1: the shared
// session control plane gates a turn at its boundary (session.Table.Decide) BEFORE
// the base gateway is dialed. A trace with a 2-turn budget dials twice and then, on
// the third turn, is refused with ErrSessionGated carrying the closed
// BUDGET_TURNS_EXHAUSTED reason — and the base gateway is NEVER dialed for the
// refused turn (a stopped/exhausted session costs zero model calls).
func TestSessionGatewayGatesTurnBudgetBeforeDial(t *testing.T) {
	tbl := session.NewTable()
	base := &countingGateway{}
	sg := microagent.NewSessionGateway(base, tbl)

	const trace = "gated"
	tbl.SetBudget(trace, session.Budget{TurnsLeft: 2, TokensLeft: session.Unbounded})
	ctx := microagent.WithTrace(context.Background(), trace)
	msg := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}

	// Two proceeding turns: both dial the base gateway.
	for i := 1; i <= 2; i++ {
		if _, err := sg.Complete(ctx, msg, nil); err != nil {
			t.Fatalf("turn %d: unexpected error %v", i, err)
		}
	}
	if got := base.dials.Load(); got != 2 {
		t.Fatalf("base gateway dialed %d times after 2 budgeted turns, want 2", got)
	}

	// Third turn: the budget is exhausted, so Decide refuses and the base gateway is
	// NOT dialed.
	_, err := sg.Complete(ctx, msg, nil)
	if !errors.Is(err, microagent.ErrSessionGated) {
		t.Fatalf("exhausted turn error = %v, want ErrSessionGated", err)
	}
	if !strings.Contains(err.Error(), session.ReasonBudgetTurns) {
		t.Errorf("gate error %q does not carry the %s reason", err.Error(), session.ReasonBudgetTurns)
	}
	if got := base.dials.Load(); got != 2 {
		t.Errorf("base gateway dialed %d times after the refused turn, want 2 (a gated turn costs zero dials)", got)
	}
}

// TestSessionGatewayDebitsUsageIntoSharedTable pins the debit half of the control
// plane (the debitSession twin): the usage a completion reports is recorded into
// the SAME shared Table via DebitUsage, so a LATER turn's Decide observes the token
// budget draining. Two 50-token turns spend a 100-token budget down to exactly 0
// (output-token exhaustion is observed AT the zero boundary — a remaining axis is
// "unbounded" only at the -1 sentinel, so the debit is sized to land on 0, not
// overshoot); the third turn is refused with BUDGET_TOKENS_EXHAUSTED — proving the
// post-turn usage actually flowed back into the shared control plane, not just the
// gate.
func TestSessionGatewayDebitsUsageIntoSharedTable(t *testing.T) {
	tbl := session.NewTable()
	base := &countingGateway{usage: agent.Usage{CompletionTokens: 50}}
	sg := microagent.NewSessionGateway(base, tbl)

	const trace = "debited"
	tbl.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: 100})
	ctx := microagent.WithTrace(context.Background(), trace)
	msg := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}

	// Turn 1: TokensLeft 100 > 0 → proceed + dial, then debit 50 → 50 left.
	// Turn 2: TokensLeft 50 > 0 → proceed + dial, then debit 50 → 0 left.
	for i := 1; i <= 2; i++ {
		if _, err := sg.Complete(ctx, msg, nil); err != nil {
			t.Fatalf("turn %d: unexpected error %v", i, err)
		}
	}
	if got := tbl.Get(trace).Budget.TokensLeft; got != 0 {
		t.Fatalf("TokensLeft after two 50-token turns = %d, want 0 (DebitUsage did not reach the shared Table)", got)
	}

	// Turn 3: the debited budget is negative → Decide drains and refuses before dial.
	_, err := sg.Complete(ctx, msg, nil)
	if !errors.Is(err, microagent.ErrSessionGated) || !strings.Contains(err.Error(), session.ReasonBudgetTokens) {
		t.Fatalf("post-debit turn error = %v, want ErrSessionGated/%s", err, session.ReasonBudgetTokens)
	}
	if got := base.dials.Load(); got != 2 {
		t.Errorf("base gateway dialed %d times, want 2 (the token-exhausted turn is not dialed)", got)
	}
}

// TestSessionGatewayNilTablePassThrough pins the nil-permissive contract: a
// SessionGateway wrapping a nil *session.Table degrades to a plain pass-through
// (Decide proceeds unbounded, DebitUsage is a no-op) rather than panicking — the
// "a loop with no table wired behaves byte-identically" discipline the session
// package documents, so the wrap is safe to install unconditionally.
func TestSessionGatewayNilTablePassThrough(t *testing.T) {
	base := &countingGateway{}
	sg := microagent.NewSessionGateway(base, nil)
	ctx := microagent.WithTrace(context.Background(), "whatever")
	msg := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}
	for i := 0; i < 3; i++ {
		if _, err := sg.Complete(ctx, msg, nil); err != nil {
			t.Fatalf("nil-table turn %d: unexpected error %v", i, err)
		}
	}
	if got := base.dials.Load(); got != 3 {
		t.Errorf("base gateway dialed %d times with a nil table, want 3 (pass-through)", got)
	}
}

// TestSessionGatewayConcurrentLoadOneTableRaceClean is the #2005 acceptance
// witness: N microagents run as goroutines in ONE process, every model turn routed
// through EXACTLY ONE SessionGateway over EXACTLY ONE session.Table — the SAME
// Table the host uses for per-agent lifecycle — with the whole session drive
// (Decide + DebitUsage) exercised concurrently, race-detector clean, and ZERO
// per-agent listener or socket opened (the base gateway is a direct in-process
// call, not the #2002 smoke test's httptest hop). Run under `go test -race`.
func TestSessionGatewayConcurrentLoadOneTableRaceClean(t *testing.T) {
	tbl := session.NewTable()
	base := &countingGateway{}
	// ONE shared, session-adjudicating gateway for the whole host.
	gw := microagent.NewSessionGateway(base, tbl)

	sink := &countingSink{}
	// The SAME Table drives both the gateway's per-turn session control plane and
	// the host's per-agent lifecycle entry — one control plane, not two.
	h, err := microagent.NewHost(gw, microagent.Config{Workers: 16, Queue: 256, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	const agents, turns = 120, 3
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("sg-%03d", i)
		if err := h.Spawn(id, &tracedTurnAgent{id: id, turns: turns}); err != nil {
			t.Fatalf("Spawn(%s): %v", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v (live=%d)", err, h.Live())
	}

	rs := h.Reap()
	if len(rs) != agents {
		t.Fatalf("reaped %d results, want %d", len(rs), agents)
	}
	for _, r := range rs {
		if !r.Done || r.Err != nil || r.Steps != turns {
			t.Fatalf("agent %s: done=%v err=%v steps=%d, want done/nil/%d", r.ID, r.Done, r.Err, r.Steps, turns)
		}
	}

	// ONE session Table: exactly one entry per agent, all Stopped/"done" — the drive
	// plane and the lifecycle plane agree on the same rows.
	if got := tbl.Len(); got != agents {
		t.Errorf("shared session table has %d entries, want %d", got, agents)
	}
	for _, st := range tbl.Snapshot() {
		if st.Run != session.Stopped || st.Reason != "done" {
			t.Errorf("session %s: run=%v reason=%q, want Stopped/done", st.TraceID, st.Run, st.Reason)
		}
	}

	// EXACTLY ONE gateway carried every turn, by direct call — no per-agent socket.
	if got, want := base.dials.Load(), int64(agents*turns); got != want {
		t.Errorf("the one session gateway dialed the base %d times, want %d", got, want)
	}
	if got := sink.kind(microagent.EventDone); got != agents {
		t.Errorf("audit sink saw %d dones, want %d", got, agents)
	}
}
