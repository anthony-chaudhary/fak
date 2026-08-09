package gateway

// ctxadvice_test.go — the #2424 witnesses for ctxadvice.go: the step_advice verdict is
// PUSHED to the model once per pressure-state entry and reads identically to the pulled
// /v1/fak/ctxvalue report for the same trace, and a window that refills to the compaction
// limit turn after turn takes the closed COMPACTION_THRASH verdict at the session gate and
// shows up in the CompactionBailReasons breakdown.

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// wireStepAdvice reads the verdict the PULL surface serves for one trace — the live
// GET /v1/fak/ctxvalue wire, decoded, not a re-derivation — so a test comparing the pushed
// advisory against it is comparing two independent renderings of one decision.
func wireStepAdvice(t *testing.T, s *Server, trace string) CtxStepAdvice {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleFakCtxValue(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/ctxvalue?trace="+trace, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/fak/ctxvalue = %d, want 200", rec.Code)
	}
	var snap CtxValueSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &snap); err != nil {
		t.Fatalf("decode ctxvalue snapshot: %v\n%s", err, rec.Body.String())
	}
	if len(snap.Sessions) != 1 {
		t.Fatalf("ctxvalue snapshot carried %d sessions, want exactly the one filtered trace", len(snap.Sessions))
	}
	return snap.Sessions[0].StepAdvice
}

// TestCtxValueAdviceSpliced is the #2424 push witness. A synthetic near-full session gets
// EXACTLY ONE checkpoint advisory, its text string-equal to the rendering of the very
// verdict /v1/fak/ctxvalue serves for the same trace — the property that makes the pushed
// line trustworthy: it is not a second heuristic that can drift, it is the same decision
// delivered a second way.
func TestCtxValueAdviceSpliced(t *testing.T) {
	const trace = "tr-near-full"
	s := &Server{compactHistoryBudget: 1000, metrics: newGatewayMetrics(time.Now())}

	// A session cruising at half the window says nothing: the advice is readable on request,
	// but bounded/any are not worth spending a model-facing block on.
	s.observeCtxValue(trace, 500, 0, 0, 20, false)
	if got := wireStepAdvice(t, s, trace).StepClass; got != StepClassBounded {
		t.Fatalf("mid-window step_class = %q, want %q (fixture no longer models the state it claims)", got, StepClassBounded)
	}
	if note := s.ctxAdviceNoteOnce(trace); note != "" {
		t.Fatalf("bounded state pushed an advisory %q — only checkpoint/rebuild are worth the model's attention", note)
	}

	// The window fills to 90% of the budget: the verdict crosses into checkpoint.
	s.observeCtxValue(trace, 900, 0, 0, 20, false)
	wire := wireStepAdvice(t, s, trace)
	if wire.StepClass != StepClassCheckpoint {
		t.Fatalf("near-full step_class = %q, want %q", wire.StepClass, StepClassCheckpoint)
	}

	note := s.ctxAdviceNoteOnce(trace)
	if note == "" {
		t.Fatal("entering checkpoint pushed no advisory — the verdict stayed pull-only, which is the #2424 defect")
	}
	if want := ctxAdviceNote(wire); note != want {
		t.Fatalf("pushed advisory and the /v1/fak/ctxvalue verdict disagree for one trace:\n push: %q\n wire: %q", note, want)
	}
	if !strings.Contains(note, string(StepClassCheckpoint)) || !strings.Contains(note, wire.Reason) {
		t.Fatalf("advisory %q drops the step class or the deciding numbers the verdict named", note)
	}

	// EXACTLY ONE: sitting in the same state says nothing more, however many turns pass.
	if again := s.ctxAdviceNoteOnce(trace); again != "" {
		t.Fatalf("checkpoint advisory repeated within one state: %q", again)
	}
	s.observeCtxValue(trace, 950, 0, 0, 20, false)
	if again := s.ctxAdviceNoteOnce(trace); again != "" {
		t.Fatalf("still checkpoint, advisory fired twice for one state entry: %q", again)
	}

	// Once per STATE ENTRY, not once per session: a compaction rewrites the window, the
	// verdict becomes rebuild, and the model is told about the NEW state.
	s.observeCtxValue(trace, 200, 0, 0, 20, true)
	rebuilt := wireStepAdvice(t, s, trace)
	if rebuilt.StepClass != StepClassRebuild {
		t.Fatalf("post-compaction step_class = %q, want %q", rebuilt.StepClass, StepClassRebuild)
	}
	post := s.ctxAdviceNoteOnce(trace)
	if post == "" {
		t.Fatal("entering rebuild pushed nothing — the dedup is per SESSION, not per state entry")
	}
	if want := ctxAdviceNote(rebuilt); post != want {
		t.Fatalf("rebuild advisory and the wire verdict disagree:\n push: %q\n wire: %q", post, want)
	}
	if post == note {
		t.Fatalf("rebuild re-emitted the checkpoint text %q — the model is being told the wrong state", post)
	}
}

// TestCtxValueAdviceNoPhantom: a trace the gateway has never served a turn for gets no
// advisory. The push must never invent pressure for a session it has not observed — the
// same no-phantom invariant the multi-session snapshot holds.
func TestCtxValueAdviceNoPhantom(t *testing.T) {
	s := &Server{compactHistoryBudget: 1000}
	for _, trace := range []string{"", "   ", "never-served"} {
		if note := s.ctxAdviceNoteOnce(trace); note != "" {
			t.Fatalf("trace %q with no served turn pushed %q", trace, note)
		}
	}
	var nilServer *Server
	if note := nilServer.ctxAdviceNoteOnce("tr"); note != "" {
		t.Fatalf("nil server pushed %q", note)
	}
}

// TestCompactionThrashVerdict is the #2424 telemetry witness. Three consecutive
// refill-to-limit turns end the session with COMPACTION_THRASH at the served session gate,
// and the verdict surfaces as a CompactionBailReasons row plus its own /metrics counter.
func TestCompactionThrashVerdict(t *testing.T) {
	const trace = "tr-thrash"
	t.Setenv(envCompactionThrashStop, "1")
	s := &Server{compactHistoryBudget: 1000, metrics: newGatewayMetrics(time.Now())}
	ctx := context.Background()

	// Two back-to-back context events is an ordinary burst, not thrash: the session still
	// admits. A stop that fired here would kill healthy sessions.
	for i := 0; i < ctxThrashConsecutiveRefills-1; i++ {
		s.observeCtxValue(trace, 1000, 0, 0, 20, true)
	}
	if _, ok, _ := s.beginServedSessionTurn(ctx, trace); !ok {
		t.Fatalf("session refused after only %d consecutive refills — the run threshold is %d",
			ctxThrashConsecutiveRefills-1, ctxThrashConsecutiveRefills)
	}
	if got := s.metrics.adjudicationSummary().CompactionBailReasons[ReasonCompactionThrash]; got != 0 {
		t.Fatalf("thrash counted at %d consecutive refills = %d, want 0", ctxThrashConsecutiveRefills-1, got)
	}

	// The third consecutive refill is the verdict.
	s.observeCtxValue(trace, 1000, 0, 0, 20, true)
	turn, ok, canceled := s.beginServedSessionTurn(ctx, trace)
	if ok || canceled {
		t.Fatalf("after %d consecutive refills the turn was admitted (ok=%v canceled=%v) — no verdict was taken",
			ctxThrashConsecutiveRefills, ok, canceled)
	}
	if turn.state.Reason != ReasonCompactionThrash {
		t.Fatalf("refusal reason = %q, want %q", turn.state.Reason, ReasonCompactionThrash)
	}
	if turn.state.TraceID != trace {
		t.Fatalf("refusal state lost the trace: %+v", turn.state)
	}
	// It must NOT read as a budget drain: a thrashing session is not continued by minting a
	// continuation id, it is stopped.
	if isBudgetResetReason(turn.state) {
		t.Fatalf("COMPACTION_THRASH graded as a budget-reset reason (%+v) — it would silently re-arm the same traffic in a fresh window", turn.state)
	}

	// The refusal names itself on the wire rather than blaming an operator who never acted.
	rec := httptest.NewRecorder()
	writeSessionRefusal(rec, turn.state)
	if rec.Code != http.StatusConflict {
		t.Fatalf("thrash refusal status = %d, want 409", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "session_compaction_thrash") || !strings.Contains(body, ReasonCompactionThrash) {
		t.Fatalf("thrash refusal body does not carry its own code/reason:\n%s", body)
	}
	if strings.Contains(body, "operator control") {
		t.Fatalf("thrash refusal blamed operator control; nobody paused this session:\n%s", body)
	}

	// /metrics: the CompactionBailReasons row an operator reads to answer "why is compaction
	// not holding this session", plus the dedicated counter.
	sum := s.metrics.adjudicationSummary()
	if got := sum.CompactionBailReasons[ReasonCompactionThrash]; got != 1 {
		t.Fatalf("CompactionBailReasons[%s] = %d, want 1 (rows: %v)", ReasonCompactionThrash, got, sum.CompactionBailReasons)
	}
	if sum.CompactionBailed != 0 {
		t.Fatalf("CompactionBailed = %d — a thrash is not a bail (compaction FIRED every one of those turns); folding it into the lump corrupts the alertable rate", sum.CompactionBailed)
	}
	var b strings.Builder
	s.metrics.writeCompactionMetrics(&b)
	if !strings.Contains(b.String(), "fak_gateway_compaction_thrash_sessions_total 1") {
		t.Fatalf("/metrics missing the thrash counter:\n%s", b.String())
	}
	if strings.Contains(b.String(), `fak_gateway_compaction_bail_reason_total{reason="`+ReasonCompactionThrash) {
		t.Fatal("thrash was emitted as a bail_reason label — that HELP claims a CLOSED set owned by the compactor's own vocabulary")
	}

	// ONE verdict per thrashing stretch: a fourth refill does not re-count.
	s.observeCtxValue(trace, 1000, 0, 0, 20, true)
	if got := s.metrics.adjudicationSummary().CompactionBailReasons[ReasonCompactionThrash]; got != 1 {
		t.Fatalf("thrash counted %d times for one stretch, want 1 — the counter measures sessions, not turns", got)
	}

	// A turn that fires no context event breaks the run: the window stopped refilling.
	s.observeCtxValue(trace, 300, 0, 0, 20, false)
	if streak := s.compactionThrashStreak(trace); streak != 0 {
		t.Fatalf("streak = %d after a non-event turn, want 0", streak)
	}
	if _, ok, _ := s.beginServedSessionTurn(ctx, trace); !ok {
		t.Fatal("session still refused after the refill run broke — the verdict must track a RUN, not latch forever")
	}
}

// TestCompactionThrashStopIsOptIn: with FAK_COMPACTION_THRASH_STOP unset the served path is
// byte-for-byte historical — a thrashing session is COUNTED but never refused. Detection
// ships on; enforcement soaks first.
func TestCompactionThrashStopIsOptIn(t *testing.T) {
	const trace = "tr-thrash-unarmed"
	t.Setenv(envCompactionThrashStop, "")
	s := &Server{compactHistoryBudget: 1000, metrics: newGatewayMetrics(time.Now())}
	for i := 0; i < ctxThrashConsecutiveRefills+1; i++ {
		s.observeCtxValue(trace, 1000, 0, 0, 20, true)
	}
	if ref := s.compactionThrashRefusal(trace); ref != nil {
		t.Fatalf("unarmed deploy refused a thrashing session: %+v", ref)
	}
	if _, ok, _ := s.beginServedSessionTurn(context.Background(), trace); !ok {
		t.Fatal("unarmed deploy refused the turn at the session gate")
	}
	if got := s.metrics.adjudicationSummary().CompactionBailReasons[ReasonCompactionThrash]; got != 1 {
		t.Fatalf("unarmed deploy counted %d thrash verdicts, want 1 — the telemetry is the part that is NOT opt-in", got)
	}
}

// TestCompactionThrashIsARegisteredReason: the verdict is not free text. It has to resolve
// in the repo's refusal vocabulary (dos.toml — what `dos check-reason COMPACTION_THRASH`
// reads), so the token the wire carries is the token an operator can look up.
func TestCompactionThrashIsARegisteredReason(t *testing.T) {
	toml, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	if !bytes.Contains(toml, []byte("[reasons."+ReasonCompactionThrash+"]")) {
		t.Fatalf("%q is absent from dos.toml — a refusal reason the kernel cannot resolve is free text", ReasonCompactionThrash)
	}
}
