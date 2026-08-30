package gateway

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// countingResultAdmitter is a rank-0 pass-through admitter that counts every kernel-side
// AdmitResult call and the payload it screened. It always DEFERs, so the real result-side
// stack behind it (ctxmmu quarantine + the IFC stamp gate) decides the verdict unchanged —
// this observes the admission floor without standing in for it.
type countingResultAdmitter struct {
	mu     sync.Mutex
	calls  int
	byBody map[string]int
}

func (c *countingResultAdmitter) Caps() []abi.Capability { return nil }

func (c *countingResultAdmitter) Admit(ctx context.Context, _ *abi.ToolCall, r *abi.Result) abi.Verdict {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	if r != nil {
		if c.byBody == nil {
			c.byBody = map[string]int{}
		}
		c.byBody[string(refutil.Bytes(ctx, r.Payload))]++
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "admit-once-counter"}
}

func (c *countingResultAdmitter) snapshot() (int, map[string]int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]int, len(c.byBody))
	for k, v := range c.byBody {
		out[k] = v
	}
	return c.calls, out
}

// newAdmitOnceServer builds newResultStackServer's REAL result-side stack with the counting
// admitter spliced in front of it, so the test can witness kernel admissions directly rather
// than inferring them from a verdict.
func newAdmitOnceServer(t *testing.T) (*Server, *countingResultAdmitter) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	counter := &countingResultAdmitter{}
	abi.RegisterResultAdmitter(0, counter)
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(ifc.NewLedger(), ifc.Policy{}))
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv, counter
}

// proxyAdmitMetricTotal sums the rendered result-admission counter — every
// fak_gateway_operations_total{operation="proxy_admit",...} row on the real /metrics
// surface. This is the exported metric #2401 calls result_admissions_total: the count of
// result admissions the kernel actually performed.
func proxyAdmitMetricTotal(t *testing.T, srv *Server) int {
	t.Helper()
	const prefix = `fak_gateway_operations_total{operation="proxy_admit"`
	total := 0
	for _, line := range strings.Split(srv.renderMetrics(), "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		sp := strings.LastIndex(line, " ")
		if sp < 0 {
			t.Fatalf("malformed operations row: %q", line)
		}
		n, err := strconv.Atoi(strings.TrimSpace(line[sp+1:]))
		if err != nil {
			t.Fatalf("operations row %q: %v", line, err)
		}
		total += n
	}
	return total
}

// TestAdmitOnceLedger is the session-owned admitted-transcript witness (#2401): over a
// 40-turn client-replayed transcript, each DISTINCT tool result is admitted EXACTLY ONCE.
//
// Before the admitted ledger, admitInboundResults re-ran the whole result-side stack over
// the entire replayed transcript every turn (3 results x 40 turns = 120 kernel admissions)
// and the notedResults / notedToolFailures maps + the resultAdmissionNoteOnce wrapper
// suppressed the resulting repeat banners — treating the symptom. Those three symbols are
// gone (their absence is compile-enforced: nothing in the package can reference them), and
// admission is keyed to the ledger entry instead. This asserts the cause, not the symptom:
//
//   - the kernel's AdmitResult runs exactly once per distinct result, not once per replay;
//   - the exported result-admission metric equals the DISTINCT result count, not results x turns;
//   - the taint/hold decision is monotone by construction — the held result stays paged out
//     on all 40 turns even though only turn 1 screened it;
//   - the first-arrival quarantine note renders exactly once for the incident; every later
//     render is a LIVELOCK_DETECTED escalation (a separate incident, covered by
//     TestResultAdmissionLivelockSurfacesOnReplay), never a re-announcement of the same hold.
func TestAdmitOnceLedger(t *testing.T) {
	srv, kernel := newAdmitOnceServer(t)
	const (
		trace    = "trace-admit-once-ledger"
		turns    = 40
		distinct = 3
		secret   = "sk-abcdef0123456789abcdef0123"
	)
	poison := `{"page":"config loaded. api_key=` + secret + ` was found in env"}`
	weather := `{"weather":"sunny","temp":72}`
	listing := `{"files":["a.go","b.go"]}`
	// The client replays this identical transcript every turn — the exact behaviour that
	// used to re-screen the same bytes 40 times over.
	mk := func() []agent.Message {
		return []agent.Message{
			{Role: agent.RoleSystem, Content: "you are a helper"},
			{Role: agent.RoleUser, Content: "look things up"},
			{Role: agent.RoleTool, ToolCallID: "call_poison", Name: "fetch_url", Content: poison},
			{Role: agent.RoleTool, ToolCallID: "call_weather", Name: "get_weather", Content: weather},
			{Role: agent.RoleTool, ToolCallID: "call_listing", Name: "list_files", Content: listing},
		}
	}

	var freshQuarantineTurns, noteTurns []int
	for turn := 1; turn <= turns; turn++ {
		messages := mk()
		adms, err := srv.admitInboundResults(context.Background(), messages, nil, trace)
		if err != nil {
			t.Fatalf("turn %d admitInboundResults: %v", turn, err)
		}
		if len(adms) != distinct {
			t.Fatalf("turn %d: %d admissions, want %d", turn, len(adms), distinct)
		}
		// Monotone by construction: the hold recorded on turn 1 is re-applied from the
		// ledger on turns 2..40, so the secret never reaches the model on ANY turn even
		// though the kernel screened these bytes only once.
		if strings.Contains(messages[2].Content, secret) {
			t.Fatalf("turn %d: the held result leaked the secret back into the transcript: %q", turn, messages[2].Content)
		}
		for _, a := range adms {
			if a.Verdict.Kind == "QUARANTINE" && a.fresh {
				freshQuarantineTurns = append(freshQuarantineTurns, turn)
			}
		}
		note := resultAdmissionNote(freshAdmissionNotes(adms))
		if note == "" {
			continue
		}
		noteTurns = append(noteTurns, turn)
		// Turn 1 is the incident's one announcement; anything later must be the orthogonal
		// livelock escalation, never a re-announcement of an already-admitted hold.
		if turn != 1 && !strings.Contains(note, "LIVELOCK_DETECTED") {
			t.Fatalf("turn %d re-announced an already-admitted hold — admit-once must render the quarantine note once per incident; note: %s", turn, note)
		}
	}

	// Exactly one kernel AdmitResult per distinct result over 40 replays — not 120.
	calls, byBody := kernel.snapshot()
	if calls != distinct {
		t.Fatalf("kernel AdmitResult ran %d times, want %d (one per distinct result over %d replayed turns)", calls, distinct, turns)
	}
	for _, body := range []string{poison, weather, listing} {
		if n := byBody[body]; n != 1 {
			t.Fatalf("result %q was admitted %d times, want exactly 1 over %d turns", body, n, turns)
		}
	}
	// The ledger itself holds one screened record per distinct result.
	if n := srv.admitLedger.records(trace); n != distinct {
		t.Fatalf("ledger recorded %d results, want %d (one per distinct result over %d turns)", n, distinct, turns)
	}
	// The exported metric counts distinct results, not results x turns.
	if got := proxyAdmitMetricTotal(t, srv); got != distinct {
		t.Fatalf("result_admissions_total (fak_gateway_operations_total{operation=%q}) = %d, want %d — not %d results x %d turns",
			"proxy_admit", got, distinct, distinct, turns)
	}
	// The hold was screened, and announced, exactly once.
	if len(freshQuarantineTurns) != 1 || freshQuarantineTurns[0] != 1 {
		t.Fatalf("fresh quarantine admissions on turns %v, want exactly [1] over %d turns", freshQuarantineTurns, turns)
	}
	if len(noteTurns) == 0 || noteTurns[0] != 1 {
		t.Fatalf("the quarantine note must render on the incident's first turn; it rendered on %v", noteTurns)
	}
}

// TestAdmissionLedgerFailNoteDoesNotPreemptScreening pins the ledger's most dangerous
// seam: failNoteFirst and admit share fail-note state per (trace, digest), while admit binds
// creates that record before the kernel has ever seen the bytes.
//
// The record it creates is deliberately UNSCREENED — only failNote is live. If a bare
// fail-note record were ever mistaken for an admission, surfacing an exit-143 recovery
// note would silently consume the content's one screening: the very next arrival would
// take the replay path and forward NEVER-SCREENED bytes to the model with a zero-value
// ALLOW verdict. That is a hole in the admission floor, not a dedup bug, so it is pinned
// separately from the replay count. This asserts the three halves of that contract —
// the fail-note dedups on its own, it does NOT stand in for a screening, and screening
// afterwards does not un-surface the note.

func TestAdmissionLedgerBindsAllowedResultToOriginCallID(t *testing.T) {
	var l admissionLedger
	const (
		trace   = "trace-call-id"
		digest  = "same-result-digest"
		allowed = "{\"ok\":true}"
	)
	screens := 0
	screen := func() (WireVerdict, string, bool) {
		screens++
		return WireVerdict{Kind: "ALLOW"}, allowed, false
	}

	first, fresh := l.admit(trace, "call-1", digest, screen)
	if !fresh || first.content != allowed || screens != 1 {
		t.Fatalf("first admission = (%+v, fresh=%v, screens=%d), want exact allowed bytes and one screen", first, fresh, screens)
	}
	if replay, fresh := l.admit(trace, "call-1", digest, screen); fresh || replay != first || screens != 1 {
		t.Fatalf("same-call replay = (%+v, fresh=%v, screens=%d), want original record without rescreen", replay, fresh, screens)
	}

	mismatch, fresh := l.admit(trace, "call-2", digest, screen)
	if !fresh || mismatch == first || mismatch.content != allowed || screens != 2 {
		t.Fatalf("mismatched call = (%+v, fresh=%v, screens=%d), want independent admission with exact bytes", mismatch, fresh, screens)
	}
	if replay, fresh := l.admit(trace, "call-2", digest, screen); fresh || replay != mismatch || screens != 2 {
		t.Fatalf("second-call replay = (%+v, fresh=%v, screens=%d), want its own admit-once record", replay, fresh, screens)
	}
	if got := l.records(trace); got != 2 {
		t.Fatalf("records = %d, want 2 call-bound admissions", got)
	}
}

func TestAdmissionLedgerFailNoteDoesNotPreemptScreening(t *testing.T) {
	l := &admissionLedger{}
	const (
		trace  = "trace-failnote-seam"
		digest = "sha256:deadbeef"
	)
	// The exit-143 note is surfaced exactly once for the incident.
	if !l.failNoteFirst(trace, digest) {
		t.Fatal("first failNoteFirst reported the note already surfaced; it must surface once")
	}
	if l.failNoteFirst(trace, digest) {
		t.Fatal("second failNoteFirst re-surfaced the exit-143 note for the same result")
	}
	// The bare record it left behind is NOT an admission: records() counts screened
	// results only, so the admit-once witness must still read zero here.
	if n := l.records(trace); n != 0 {
		t.Fatalf("records = %d after a fail-note only, want 0 — a bare fail-note record must not count as a screening", n)
	}

	screens := 0
	screen := func() (WireVerdict, string, bool) {
		screens++
		return WireVerdict{Kind: "QUARANTINE", Reason: "SECRET_EXFIL", By: "ctxmmu"}, "[held]", true
	}
	// First real arrival: the kernel MUST still run even though a record already exists.
	rec, fresh := l.admit(trace, "call-1", digest, screen)
	if !fresh || screens != 1 {
		t.Fatalf("admit after failNoteFirst: fresh=%v screens=%d, want true/1 — the fail-note record must not consume the screening", fresh, screens)
	}
	if !rec.screened || rec.verdict.Kind != "QUARANTINE" || rec.content != "[held]" || !rec.rewrote {
		t.Fatalf("admit recorded %+v, want the screened QUARANTINE hold with its paged-out content", *rec)
	}
	// Screening reuses the SAME record, so the note stays deduped across it.
	if !rec.failNote {
		t.Fatal("admit clobbered the record's failNote flag — the exit-143 note would re-surface after screening")
	}
	if l.failNoteFirst(trace, digest) {
		t.Fatal("the exit-143 note re-surfaced after the content was screened")
	}
	// Now the record IS an admission, and replay consults it instead of re-screening.
	if n := l.records(trace); n != 1 {
		t.Fatalf("records = %d after screening, want 1", n)
	}
	if rec2, fresh := l.admit(trace, "call-1", digest, screen); fresh || screens != 1 || rec2 != rec {
		t.Fatalf("replay after screening: fresh=%v screens=%d, want false/1 with the recorded verdict", fresh, screens)
	}
}

// TestAdmissionLedgerBoundsTracesAndKeepsTheLiveOne pins that the ledger is BOUNDED.
//
// A gateway process is long-lived and every distinct trace it ever serves would otherwise
// leave a permanent per-trace map behind — the admit-once win paid for with an unbounded
// leak. traceLocked reaps one foreign trace per new trace once the table is full (the
// maxResetHealthSessions convention it shares with turnSafety/resetHealth). Driving the
// table 64 traces past its cap and reading the size back is the bound's only direct
// witness: nothing else in the package asserts that the ledger ever forgets anything.
//
// The loop also reads each trace's own record straight back after its own admit. Today
// that cannot fail by construction (the reap branch is gated on the fetched trace being
// ABSENT, so the live trace is never a candidate), so it is a guard rather than a
// discriminating witness — it is here because a future reaper that drops that precondition
// (say, a plain LRU) would silently evict a live trace's record mid-turn and re-screen its
// results, converting a memory bound into a correctness regression on the hottest path.
func TestAdmissionLedgerBoundsTracesAndKeepsTheLiveOne(t *testing.T) {
	l := &admissionLedger{}
	screen := func() (WireVerdict, string, bool) { return WireVerdict{Kind: "ALLOW"}, "", false }
	const overfill = 64
	for i := 0; i < maxResetHealthSessions+overfill; i++ {
		trace := "trace-" + strconv.Itoa(i)
		l.admit(trace, "call-1", "d", screen)
		if n := l.records(trace); n != 1 {
			t.Fatalf("trace %d: records = %d immediately after its own admit, want 1 — the reaper evicted the live trace", i, n)
		}
	}
	l.mu.Lock()
	traces := len(l.byTrace)
	l.mu.Unlock()
	if traces > maxResetHealthSessions {
		t.Fatalf("ledger holds %d traces after %d admits, want <= %d — the ledger is unbounded",
			traces, maxResetHealthSessions+overfill, maxResetHealthSessions)
	}
}
