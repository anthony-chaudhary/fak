package gateway

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore_planner_test.go — contract test for the ctxview-elision restore source (#3062).
//
// Slice 1 served exactly one dropped-span source (the compaction-tombstone stash). This proves the
// generalization: when the per-trace stash MISSES, a fak_context_restore(id) whose digest addresses a
// span the trace's retained SessionPlanner still holds — a span the planned-view rewrite ELIDED from
// the passthrough but the lossless store kept — pages back in by its content-address, WITNESSED, with
// no new routing in restoreContext (the planner is a ctxplan.Store via its Spans+Materialize pair).
// The miss case proves the source falls THROUGH rather than fabricating a hit.

// injectPlanner ingests msgs into a fresh SessionPlanner and retains it under trace on srv, exactly
// as a served turn's sessionPlannerFor would, returning the planner so the caller can address its
// spans by content-address. The resident budget is deliberately tiny so later spans elide from the
// planned view while the store stays lossless — the elided-then-restored path the issue targets.
func injectPlanner(t *testing.T, srv *Server, trace string, msgs []agent.Message) *agent.SessionPlanner {
	t.Helper()
	planner := agent.NewSessionPlanner(64)
	planner.RenderTurn(context.Background(), msgs) // lowers every message into the lossless store
	srv.sessionPlannerMu.Lock()
	if srv.sessionPlanners == nil {
		srv.sessionPlanners = make(map[string]*agent.SessionPlanner)
	}
	srv.sessionPlanners[trace] = planner
	srv.sessionPlannerMu.Unlock()
	return planner
}

func TestRestoreContext_CtxviewElisionSource(t *testing.T) {
	srv := newTestServer(t)
	trace := srv.traceFor("t-ctxview")
	const elided = "the ORIGINATING task the planned-view rewrite elided from the passthrough"
	injectPlanner(t, srv, trace, []agent.Message{
		{Role: agent.RoleSystem, Content: "system preamble"},
		{Role: agent.RoleUser, Content: elided},
		{Role: agent.RoleUser, Content: "a later, resident turn"},
	})

	// The digest is computed the SAME way the store does (ctxplan.Digest over the raw content), so a
	// caller holding only the content-address — as a resuming model does — restores by that address.
	digest := ctxplan.Digest([]byte(elided))
	got, err := srv.restoreContext("", ContextRestoreRequest{ID: digest, TraceID: trace})
	if err != nil {
		t.Fatalf("restoreContext ctxview source: unexpected err %v", err)
	}
	if got.Bytes != elided {
		t.Fatalf("restored bytes = %q, want the verbatim elided span %q", got.Bytes, elided)
	}
	if got.Provenance != "WITNESSED" {
		t.Fatalf("provenance = %q, want WITNESSED (fak authored the drop, never a guess)", got.Provenance)
	}
	if got.ID != digest {
		t.Fatalf("echoed id = %q, want %q", got.ID, digest)
	}
}

// TestRestoreContext_CtxviewMissFallsThrough: a digest no source holds is a plain miss. The planner
// source must fall THROUGH (not error, not fabricate), and with no recall image named restore ends at
// ErrRestoreMiss — the "never had it" answer, distinct from a trust-gate refusal.
func TestRestoreContext_CtxviewMissFallsThrough(t *testing.T) {
	srv := newTestServer(t)
	trace := srv.traceFor("t-ctxview-miss")
	injectPlanner(t, srv, trace, []agent.Message{
		{Role: agent.RoleUser, Content: "only span"},
	})
	unknown := ctxplan.Digest([]byte("a span no store holds"))
	_, err := srv.restoreContext("", ContextRestoreRequest{ID: unknown, TraceID: trace})
	if !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("unknown digest: err = %v, want ErrRestoreMiss", err)
	}
}

// TestRestoreContext_NoPlannerIsMiss: a trace that never planned a turn has no retained planner, so
// existingSessionPlanner returns nil and the source is skipped cleanly (a plain miss, not a panic on
// a nil map / freshly-minted empty planner).
func TestRestoreContext_NoPlannerIsMiss(t *testing.T) {
	srv := newTestServer(t)
	trace := srv.traceFor("t-no-planner")
	digest := ctxplan.Digest([]byte("anything"))
	_, err := srv.restoreContext("", ContextRestoreRequest{ID: digest, TraceID: trace})
	if !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("no-planner trace: err = %v, want ErrRestoreMiss", err)
	}
}
