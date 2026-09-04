package microagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"

	// The microagent-minimal registration set (#2009) — the defconfig designed
	// for an in-process microagent host. It wires the Ref resolver, the
	// adjudication floor, and the mock/inkernel engines gateway.New requires,
	// WITHOUT the full ~30-subsystem defconfig.
	_ "github.com/anthony-chaudhary/fak/internal/registrations/microagent"
)

// chatPlanner is the ONE shared gateway seam handed to the host: a deliberately
// minimal agent.Planner that drives the served /v1/chat/completions route of
// the in-process fak serve gateway. Every hosted agent's model turn goes
// through this one value — no per-agent client, credential, or process.
type chatPlanner struct {
	url    string
	client *http.Client
	calls  atomic.Int64
}

func (p *chatPlanner) Model() string { return "mock" }

func (p *chatPlanner) Complete(ctx context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	type wireMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	req := struct {
		Model    string    `json:"model"`
		Messages []wireMsg `json:"messages"`
	}{Model: "mock"}
	for _, m := range msgs {
		req.Messages = append(req.Messages, wireMsg{Role: m.Role, Content: m.Content})
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gateway status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, errors.New("gateway returned no choices")
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: out.Choices[0].Message.Content}}, nil
}

// turnAgent finishes after `turns` model turns, each taken through the SHARED
// gateway the host hands it — it holds no seam of its own.
type turnAgent struct {
	id    string
	turns int
	took  int
}

func (a *turnAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	a.took++
	msg := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("agent %s turn %d", a.id, a.took)}}
	if _, err := gw.Complete(ctx, msg, nil); err != nil {
		return false, err
	}
	return a.took >= a.turns, nil
}

// countingSink is the host's ONE audit sink.
type countingSink struct {
	mu     sync.Mutex
	kinds  map[microagent.EventKind]int
	agents map[string]bool
}

func (s *countingSink) Record(ev microagent.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.kinds == nil {
		s.kinds = map[microagent.EventKind]int{}
		s.agents = map[string]bool{}
	}
	s.kinds[ev.Kind]++
	s.agents[ev.Agent] = true
}

func (s *countingSink) kind(k microagent.EventKind) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.kinds[k]
}

func (s *countingSink) agentCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.agents)
}

// TestHostSmoke100AgentsOneGatewayOneAuditSink is the #2002 acceptance witness:
// >=100 microagents run to completion in ONE process against the Mock engine,
// every model turn served by EXACTLY ONE in-process fak serve gateway, every
// lifecycle event landing in EXACTLY ONE audit sink, one session-table entry
// per agent (contrast today: one guard process + one external CLI + one
// hash-chained audit JSONL per agent).
func TestHostSmoke100AgentsOneGatewayOneAuditSink(t *testing.T) {
	// The ONE gateway for the whole host: the real internal/gateway server over
	// the Mock engine (EngineID "mock"; with no upstream configured its chat
	// route is served by the deterministic offline mock planner).
	srv, err := gateway.New(gateway.Config{
		EngineID: "mock",
		Model:    "mock",
		VDSO:     true,
		Logf:     func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()
	var served atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		served.Add(1)
		srv.Handler().ServeHTTP(w, r)
	}))
	defer ts.Close()

	gw := &chatPlanner{url: ts.URL, client: ts.Client()}
	tbl := session.NewTable()
	sink := &countingSink{}
	h, err := microagent.NewHost(gw, microagent.Config{Workers: 16, Queue: 256, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	const agents, turns = 120, 3
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("ma-%03d", i)
		if err := h.Spawn(id, &turnAgent{id: id, turns: turns}); err != nil {
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
		if !r.Done || r.Err != nil {
			t.Fatalf("agent %s: done=%v err=%v steps=%d", r.ID, r.Done, r.Err, r.Steps)
		}
		if r.Steps != turns {
			t.Errorf("agent %s took %d steps, want %d", r.ID, r.Steps, turns)
		}
	}

	// Per-agent state (#2002 scope 2): ONE session.Table entry per agent — a map
	// entry, not a process — all retired Stopped/"done".
	if got := tbl.Len(); got != agents {
		t.Errorf("session table has %d entries, want %d", got, agents)
	}
	for _, st := range tbl.Snapshot() {
		if st.Run != session.Stopped || st.Reason != "done" {
			t.Errorf("session %s: run=%v reason=%q, want Stopped/done", st.TraceID, st.Run, st.Reason)
		}
	}

	// EXACTLY ONE gateway carried every model turn.
	wantTurns := int64(agents * turns)
	if got := gw.calls.Load(); got != wantTurns {
		t.Errorf("shared planner carried %d turns, want %d", got, wantTurns)
	}
	if got := served.Load(); got != wantTurns {
		t.Errorf("the one gateway served %d requests, want %d", got, wantTurns)
	}

	// EXACTLY ONE audit sink saw the whole fleet.
	if got := sink.kind(microagent.EventSpawn); got != agents {
		t.Errorf("audit sink saw %d spawns, want %d", got, agents)
	}
	if got := sink.kind(microagent.EventDone); got != agents {
		t.Errorf("audit sink saw %d dones, want %d", got, agents)
	}
	if got := sink.agentCount(); got != agents {
		t.Errorf("audit sink saw %d distinct agents, want %d", got, agents)
	}
}

// stubPlanner is an offline gateway seam for lifecycle tests that need no
// served gateway.
type stubPlanner struct{}

func (stubPlanner) Model() string { return "stub" }

func (stubPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

// blockAgent parks in Step until released or cancelled; it signals started on
// its first Step so tests can order deterministically.
type blockAgent struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (a *blockAgent) Step(ctx context.Context, _ microagent.Gateway) (bool, error) {
	a.once.Do(func() { close(a.started) })
	select {
	case <-a.release:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func newBlockAgent() *blockAgent {
	return &blockAgent{started: make(chan struct{}), release: make(chan struct{})}
}

// TestBoundedQueueRefusesLoudly pins #2002 scope 3: with the one worker
// occupied and the queue full, Spawn refuses with ErrQueueFull instead of
// blocking, and the rejected id stays spawnable later.
func TestBoundedQueueRefusesLoudly(t *testing.T) {
	sink := &countingSink{}
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 1, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	a1 := newBlockAgent()
	if err := h.Spawn("a1", a1); err != nil {
		t.Fatalf("Spawn(a1): %v", err)
	}
	<-a1.started // the one worker is now inside a1.Step
	a2 := newBlockAgent()
	if err := h.Spawn("a2", a2); err != nil {
		t.Fatalf("Spawn(a2): %v", err)
	}
	if err := h.Spawn("a3", newBlockAgent()); !errors.Is(err, microagent.ErrQueueFull) {
		t.Fatalf("Spawn(a3) = %v, want ErrQueueFull", err)
	}
	if got := sink.kind(microagent.EventReject); got != 1 {
		t.Errorf("audit sink saw %d rejects, want 1", got)
	}
	// The rejected id left no residue: it is spawnable once there is room.
	close(a1.release)
	<-a2.started
	close(a2.release)
	a3 := newBlockAgent()
	close(a3.release) // completes immediately
	if err := h.Spawn("a3", a3); err != nil {
		t.Fatalf("re-Spawn(a3) after room freed: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rs := h.Reap(); len(rs) != 3 {
		t.Fatalf("reaped %d, want 3", len(rs))
	}
}

// TestCancelRetiresQueuedAndRunning covers the cancel/reap lifecycle for both a
// running and a still-queued agent.
func TestCancelRetiresQueuedAndRunning(t *testing.T) {
	sink := &countingSink{}
	tbl := session.NewTable()
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	a1 := newBlockAgent()
	if err := h.Spawn("run", a1); err != nil {
		t.Fatalf("Spawn(run): %v", err)
	}
	<-a1.started
	if err := h.Spawn("queued", newBlockAgent()); err != nil {
		t.Fatalf("Spawn(queued): %v", err)
	}
	if !h.Cancel("queued") || !h.Cancel("run") {
		t.Fatal("Cancel should report true for live agents")
	}
	if h.Cancel("absent") {
		t.Fatal("Cancel(absent) should report false")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	rs := h.Reap()
	if len(rs) != 2 {
		t.Fatalf("reaped %d, want 2", len(rs))
	}
	for _, r := range rs {
		if r.Done || r.Err == nil {
			t.Errorf("agent %s: done=%v err=%v, want cancelled", r.ID, r.Done, r.Err)
		}
	}
	if got := sink.kind(microagent.EventCancel); got != 2 {
		t.Errorf("audit sink saw %d cancels, want 2", got)
	}
	for _, id := range []string{"run", "queued"} {
		if st := tbl.Get(id); st.Run != session.Stopped || st.Reason != "cancelled" {
			t.Errorf("session %s: run=%v reason=%q, want Stopped/cancelled", id, st.Run, st.Reason)
		}
	}
}

// TestSpawnRefusalsAndClose pins the remaining refusal edges: duplicate ids
// (live AND retired), draining, and closed.
func TestSpawnRefusalsAndClose(t *testing.T) {
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	a1 := newBlockAgent()
	if err := h.Spawn("a1", a1); err != nil {
		t.Fatalf("Spawn(a1): %v", err)
	}
	<-a1.started
	if err := h.Spawn("a1", newBlockAgent()); !errors.Is(err, microagent.ErrDuplicateID) {
		t.Fatalf("Spawn(live dup) = %v, want ErrDuplicateID", err)
	}
	close(a1.release)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	// Retired id: the session entry is terminal — an id is one agent lifetime.
	if err := h.Spawn("a1", newBlockAgent()); !errors.Is(err, microagent.ErrDuplicateID) {
		t.Fatalf("Spawn(retired dup) = %v, want ErrDuplicateID", err)
	}
	if err := h.Spawn("a2", newBlockAgent()); !errors.Is(err, microagent.ErrDraining) {
		t.Fatalf("Spawn while draining = %v, want ErrDraining", err)
	}
	h.Close()
	h.Close() // idempotent
	if err := h.Spawn("a3", newBlockAgent()); !errors.Is(err, microagent.ErrClosed) {
		t.Fatalf("Spawn after Close = %v, want ErrClosed", err)
	}
	if _, nerr := microagent.NewHost(nil, microagent.Config{}); !errors.Is(nerr, microagent.ErrNilGateway) {
		t.Fatalf("NewHost(nil) = %v, want ErrNilGateway", nerr)
	}
	if err := h.Spawn("a4", nil); !errors.Is(err, microagent.ErrNilAgent) {
		t.Fatalf("Spawn(nil agent) = %v, want ErrNilAgent", err)
	}
}

// errStep is the sentinel a failing agent returns to drive the EventError
// retirement branch: a non-cancel error, distinct from any context error.
var errStep = errors.New("microagent test: step failed")

// errAgent fails on its first Step with a non-cancel error (no ctx involved).
type errAgent struct{}

func (errAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	return false, errStep
}

// TestErrorRetiresWithEventError pins the last #2002 lifecycle edge (scope 3):
// a Step that returns a non-cancel error retires the agent as EventError — NOT
// EventCancel — even though retire() cancels the job ctx first. The one Reap
// result carries that error (errors.Is), the ONE audit sink sees exactly one
// error event and zero cancels, and the per-agent session entry lands Stopped
// with an "error: ..." reason. This is the branch the smoke/queue/cancel tests
// never take, so without it EventError has zero coverage.
func TestErrorRetiresWithEventError(t *testing.T) {
	sink := &countingSink{}
	tbl := session.NewTable()
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 2, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	if err := h.Spawn("boom", errAgent{}); err != nil {
		t.Fatalf("Spawn(boom): %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	rs := h.Reap()
	if len(rs) != 1 {
		t.Fatalf("reaped %d results, want 1", len(rs))
	}
	if r := rs[0]; r.Done || !errors.Is(r.Err, errStep) || r.Steps != 1 {
		t.Fatalf("result: done=%v err=%v steps=%d, want done=false err=errStep steps=1", r.Done, r.Err, r.Steps)
	}
	if got := sink.kind(microagent.EventError); got != 1 {
		t.Errorf("audit sink saw %d error events, want 1", got)
	}
	if got := sink.kind(microagent.EventCancel); got != 0 {
		t.Errorf("audit sink saw %d cancels, want 0 (a Step error is not a cancel)", got)
	}
	wantReason := "error: " + errStep.Error()
	if st := tbl.Get("boom"); st.Run != session.Stopped || st.Reason != wantReason {
		t.Errorf("session boom: run=%v reason=%q, want Stopped/%q", st.Run, st.Reason, wantReason)
	}
}

type turnBudgetAgent struct {
	steps int
	done  int
}

func (a *turnBudgetAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	a.steps++
	return a.steps >= a.done, nil
}

func TestHostEnforcesHardTurnBudget(t *testing.T) {
	tests := []struct {
		name      string
		maxTurns  int
		doneAt    int
		wantDone  bool
		wantSteps int
		wantErr   error
	}{
		{name: "one-turn-completion", maxTurns: 1, doneAt: 1, wantDone: true, wantSteps: 1},
		{name: "three-turn-completion", maxTurns: 3, doneAt: 3, wantDone: true, wantSteps: 3},
		{name: "over-budget-refusal", maxTurns: 3, doneAt: 4, wantSteps: 3, wantErr: microagent.ErrTurnBudget},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, MaxTurns: tt.maxTurns})
			if err != nil {
				t.Fatalf("NewHost: %v", err)
			}
			defer h.Close()
			a := &turnBudgetAgent{done: tt.doneAt}
			if err := h.Spawn("bounded", a); err != nil {
				t.Fatalf("Spawn: %v", err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			if err := h.Drain(ctx); err != nil {
				t.Fatalf("Drain: %v", err)
			}
			rs := h.Reap()
			if len(rs) != 1 {
				t.Fatalf("reaped %d results, want 1", len(rs))
			}
			r := rs[0]
			if r.Done != tt.wantDone || r.Steps != tt.wantSteps || !errors.Is(r.Err, tt.wantErr) {
				t.Fatalf("result: done=%v steps=%d err=%v, want done=%v steps=%d err=%v", r.Done, r.Steps, r.Err, tt.wantDone, tt.wantSteps, tt.wantErr)
			}
			if a.steps != tt.wantSteps {
				t.Fatalf("agent Step called %d times, want hard ceiling %d", a.steps, tt.wantSteps)
			}
		})
	}
}

// TestHostRuns100EnrolledAgentsUnder100MBRSS proves that 100 enrolled microagents
// execute concurrently within < 100 MB total heap memory (#11182).
func TestHostRuns100EnrolledAgentsUnder100MBRSS(t *testing.T) {
	// Pre-GC to measure clean baseline
	runtime.GC()
	var mBefore runtime.MemStats
	runtime.ReadMemStats(&mBefore)

	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 16, Queue: 256})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	const numAgents = 100
	const turns = 5
	for i := 0; i < numAgents; i++ {
		id := fmt.Sprintf("bench-agent-%03d", i)
		if err := h.Spawn(id, &turnAgent{id: id, turns: turns}); err != nil {
			t.Fatalf("Spawn: %v", err)
		}
	}

	var mPeak runtime.MemStats
	runtime.ReadMemStats(&mPeak)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	results := h.Reap()
	if len(results) != numAgents {
		t.Fatalf("reaped %d results, want %d", len(results), numAgents)
	}

	runtime.ReadMemStats(&mPeak)
	heapAllocMB := float64(mPeak.Alloc) / (1024 * 1024)
	t.Logf("100 enrolled microagents completed %d total steps; heap alloc: %.2f MB", numAgents*turns, heapAllocMB)

	// Memory budget ceiling: < 100 MB
	if heapAllocMB >= 100.0 {
		t.Fatalf("microagent host exceeded 100 MB memory bound: alloc=%.2f MB", heapAllocMB)
	}
}
