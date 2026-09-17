package microagent

import (
	"context"
	"errors"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// waveStubPlanner is the minimal in-package Gateway the wave tests step through;
// the microagent_test package has its own stubPlanner but that one is not visible
// from this internal test package.
type waveStubPlanner struct{}

func (waveStubPlanner) Model() string { return "wave-stub" }

func (waveStubPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

// countingWaveAgent is a tiny Microagent the wave tests drive. It completes in
// exactly one Step, so a test can assert N agents retired without modeling any
// real loop, and it counts concurrent Steps so the test can witness that the
// shared host actually ran agents in parallel up to its worker budget.
type countingWaveAgent struct {
	active *atomic.Int64
	peak   *atomic.Int64
}

func (a *countingWaveAgent) Step(ctx context.Context, _ Gateway) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	n := a.active.Add(1)
	defer a.active.Add(-1)
	for old := a.peak.Load(); n > old && !a.peak.CompareAndSwap(old, n); old = a.peak.Load() {
	}
	return true, nil
}

// blockingWaveAgent blocks in its first Step until release is closed (or ctx is
// done), so a worker that has picked this agent up cannot retire it and free a
// queue slot. It makes admission counts and resident-concurrency observations
// deterministic instead of racing a fast one-shot agent.
type blockingWaveAgent struct {
	release chan struct{}
	started *atomic.Int64
}

func (a *blockingWaveAgent) Step(ctx context.Context, _ Gateway) (bool, error) {
	a.started.Add(1)
	select {
	case <-a.release:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func newWaveEnrollments(n int, active, peak *atomic.Int64) []Enrollment {
	out := make([]Enrollment, n)
	for i := 0; i < n; i++ {
		out[i] = Enrollment{ID: "w-" + strconv.Itoa(i), Agent: &countingWaveAgent{active: active, peak: peak}}
	}
	return out
}

// TestBudgetForWaveIsADeclaredCapNotTheBacklog pins the pure sizing rule: the
// resident worker budget is the SMALLER of the caller's declared ceiling and the
// batch size, floored at 1. It is a property of the declared capacity, never of
// the backlog length -- a 1000-issue wave on a 16-slot box must yield 16, not
// 1000.
func TestBudgetForWaveIsADeclaredCapNotTheBacklog(t *testing.T) {
	cases := []struct{ declared, batch, want int }{
		{16, 1000, 16}, // declared cap binds
		{64, 10, 10},   // batch is smaller
		{0, 1000, DefaultWaveBudget},
		{16, 0, 1},
		{-5, 7, 7},
		{1, 1, 1},
	}
	for _, c := range cases {
		if got := BudgetForWave(c.declared, c.batch); got != c.want {
			t.Fatalf("BudgetForWave(%d,%d)=%d, want %d", c.declared, c.batch, got, c.want)
		}
	}
}

// TestWaveHostEnrollsThousandAgentsIntoOneHost is the core witness for the
// concurrency spine: 1000 agents are admitted into ONE shared host and all retire
// done, with the host stepping many at once -- no per-agent host construction.
//
// It constructs exactly one WaveHost (the whole point: host constructions == 1),
// enrolls 1000, drains once, and asserts every agent retired. Fan-out is witnessed
// by an agent that blocks in its Step: the host can only retire the first agent
// once its worker is released, so by the time the batch is drained the host must
// have had min(workers, agents) agents resident at once -- a real parallel fan-out,
// not a relabeled serial loop.
func TestWaveHostEnrollsThousandAgentsIntoOneHost(t *testing.T) {
	const agents = 1000
	const workers = 64

	w, err := NewWaveHost(waveStubPlanner{}, Config{}, WaveConfig{Workers: workers, Queue: agents})
	if err != nil {
		t.Fatalf("NewWaveHost: %v", err)
	}
	defer w.Close()

	release := make(chan struct{})
	started := &atomic.Int64{}
	enrollments := make([]Enrollment, agents)
	for i := 0; i < agents; i++ {
		enrollments[i] = Enrollment{ID: "w-" + strconv.Itoa(i), Agent: &blockingWaveAgent{release: release, started: started}}
	}
	admitted := w.Enroll(enrollments)
	if got := Admitted(admitted); got != agents {
		t.Fatalf("admitted=%d, want %d (first refusal: %v)", got, agents, FirstRefusal(admitted))
	}
	if refusal := FirstRefusal(admitted); refusal != nil {
		t.Fatalf("unexpected refusal: %v", refusal)
	}

	// All workers must be occupied before any retires: the first `workers` agents
	// are pinned in Step, so resident concurrency is exactly the worker budget.
	deadline := time.Now().Add(10 * time.Second)
	for started.Load() < workers && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := started.Load(); got < workers {
		t.Fatalf("resident agents=%d, want >=%d (the shared host must step agents in parallel up to its worker budget)", got, workers)
	}
	close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	results, err := w.DrainAll(ctx)
	if err != nil {
		t.Fatalf("DrainAll: %v", err)
	}
	if len(results) != agents {
		t.Fatalf("reaped %d results, want %d", len(results), agents)
	}
	done := 0
	for _, r := range results {
		if r.Done && r.Err == nil {
			done++
		}
	}
	if done != agents {
		t.Fatalf("done=%d, want %d", done, agents)
	}
}

// TestWaveHostPartialAdmissionIsObservable pins the fail-loud property: when the
// bounded queue cannot hold the whole batch, the refusals are reported per agent
// with their typed error, and the admitted agents still run to completion. A
// caller must never see a silently dropped tail.
func TestWaveHostPartialAdmissionIsObservable(t *testing.T) {
	// Workers=1, Queue=2: only 3 agents can be admitted at once; the 4th refuses.
	w, err := NewWaveHost(waveStubPlanner{}, Config{}, WaveConfig{Workers: 1, Queue: 2})
	if err != nil {
		t.Fatalf("NewWaveHost: %v", err)
	}
	defer w.Close()

	// The agents block in Step so the single worker cannot retire one and free a
	// queue slot mid-enrollment; admission is then exactly Workers + Queue.
	release := make(chan struct{})
	defer close(release)
	started := &atomic.Int64{}
	enrollments := make([]Enrollment, 4)
	for i := range enrollments {
		enrollments[i] = Enrollment{ID: "p-" + strconv.Itoa(i), Agent: &blockingWaveAgent{release: release, started: started}}
	}
	admitted := w.Enroll(enrollments)
	if got := Admitted(admitted); got != 3 {
		t.Fatalf("admitted=%d, want 3 (queue=2 + 1 in-flight)", got)
	}
	refusal := FirstRefusal(admitted)
	if !errors.Is(refusal, ErrQueueFull) {
		t.Fatalf("first refusal=%v, want ErrQueueFull", refusal)
	}
	if admitted[3].Admitted || !errors.Is(admitted[3].Err, ErrQueueFull) {
		t.Fatalf("4th enrollment result=%+v, want refused with ErrQueueFull", admitted[3])
	}
}

// TestRunWaveOneCallFanOut exercises the single-call harness entry point and
// confirms it constructs one host, admits the batch, and returns both the
// admission results and the per-agent outcomes.
func TestRunWaveOneCallFanOut(t *testing.T) {
	active, peak := &atomic.Int64{}, &atomic.Int64{}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	admitted, results, err := RunWave(ctx, waveStubPlanner{}, Config{}, WaveConfig{Workers: 16, Queue: 256}, newWaveEnrollments(256, active, peak))
	if err != nil {
		t.Fatalf("RunWave: %v", err)
	}
	if Admitted(admitted) != 256 || len(results) != 256 {
		t.Fatalf("admitted=%d results=%d, want 256/256", Admitted(admitted), len(results))
	}
	for _, r := range results {
		if !r.Done || r.Err != nil {
			t.Fatalf("agent %s not done: %+v", r.ID, r)
		}
	}
}

// stuckWaveAgent models the exact wedge #13080 was filed for: its Step IGNORES
// ctx entirely and blocks on a release channel until the test lets it go. It is
// the adversarial input the bounded close exists for -- a drain timeout alone
// cannot stop it, and an unbounded Close (workers.Wait) would hang forever behind
// this goroutine.
type stuckWaveAgent struct {
	release chan struct{}
	started *atomic.Int64
}

func (a *stuckWaveAgent) Step(context.Context, Gateway) (bool, error) {
	a.started.Add(1)
	<-a.release // deliberately not select-on-ctx: this is the ctx-ignoring agent
	return true, nil
}

// TestCloseContextIsBoundedAgainstACtxIgnoringAgent is the #13080 acceptance
// witness for the microagent seam: a host holding an agent whose Step ignores ctx
// still returns from CloseContext within the declared budget, reporting the
// timeout instead of blocking on workers.Wait().
func TestCloseContextIsBoundedAgainstACtxIgnoringAgent(t *testing.T) {
	h, err := NewHost(waveStubPlanner{}, Config{Workers: 1, Queue: 1})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	release := make(chan struct{})
	defer close(release) // let the wedged goroutine exit once the test is done
	started := &atomic.Int64{}
	if err := h.Spawn("stuck", &stuckWaveAgent{release: release, started: started}); err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	// Wait until the worker is actually pinned inside Step, so the close below is
	// racing the wedge and not an empty host.
	deadline := time.Now().Add(5 * time.Second)
	for started.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if started.Load() == 0 {
		t.Fatal("stuck agent never entered Step")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	began := time.Now()
	err = h.CloseContext(ctx)
	elapsed := time.Since(began)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("CloseContext err = %v, want context.DeadlineExceeded (bounded, not a hang)", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("CloseContext took %v, want bounded near the 200ms budget", elapsed)
	}
	if h.Live() != 1 {
		t.Fatalf("Live()=%d, want 1 (the wedged agent is abandoned, not retired)", h.Live())
	}
	// Idempotent: a second bounded close returns immediately without re-blocking.
	done := make(chan error, 1)
	go func() { done <- h.CloseContext(context.Background()) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("second CloseContext err = %v, want nil", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second CloseContext blocked; it must be idempotent")
	}
}

// TestHostCloseUnchangedIsUnbounded pins that the BASE Close keeps its historical
// semantics: it is the unbounded posture, so a cooperative fleet drains and the
// zero-agent case returns immediately (every other caller stays byte-identical).
func TestHostCloseUnchangedIsUnbounded(t *testing.T) {
	h, err := NewHost(waveStubPlanner{}, Config{Workers: 2, Queue: 4})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	for i := 0; i < 4; i++ {
		if err := h.Spawn("ok-"+strconv.Itoa(i), &countingWaveAgent{active: &atomic.Int64{}, peak: &atomic.Int64{}}); err != nil {
			t.Fatalf("Spawn: %v", err)
		}
	}
	done := make(chan struct{})
	go func() { h.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("base Close did not return for a cooperative fleet")
	}
	if err := h.Spawn("after", &countingWaveAgent{active: &atomic.Int64{}, peak: &atomic.Int64{}}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Spawn after Close = %v, want ErrClosed", err)
	}
}

// TestWaveHostCloseContextIsBounded is the WaveHost-level twin: the wave entry
// point the dispatch shell uses must itself be hang-proof, since a wave can hold
// an untrusted batch of N agents.
func TestWaveHostCloseContextIsBounded(t *testing.T) {
	w, err := NewWaveHost(waveStubPlanner{}, Config{}, WaveConfig{Workers: 1, Queue: 1})
	if err != nil {
		t.Fatalf("NewWaveHost: %v", err)
	}
	release := make(chan struct{})
	defer close(release)
	started := &atomic.Int64{}
	got := w.Enroll([]Enrollment{{ID: "stuck-wave", Agent: &stuckWaveAgent{release: release, started: started}}})
	if Admitted(got) != 1 {
		t.Fatalf("admitted=%d, want 1 (refusal %v)", Admitted(got), FirstRefusal(got))
	}
	deadline := time.Now().Add(5 * time.Second)
	for started.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if started.Load() == 0 {
		t.Fatal("stuck wave agent never entered Step")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if err := w.CloseContext(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("WaveHost.CloseContext err = %v, want context.DeadlineExceeded", err)
	}
}
