package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// TestAdmissionTokenBudget is the issue-#35 AC#1 token-budget witness: a request enters
// the running set only while the running set has token-budget headroom; one that would
// overflow the budget waits until a running request completes and frees it.
func TestAdmissionTokenBudget(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{TokenBudget: 100, MaxWaiting: 16, AgingRounds: 1})

	if v := c.Offer(SeqRequest{TraceID: "a", Tokens: 60}); v != VerdictAdmitted {
		t.Fatalf("first 60-token request: verdict = %s, want admitted", v)
	}
	// 60 + 60 > 100 — no headroom, so B waits rather than overflowing the budget.
	if v := c.Offer(SeqRequest{TraceID: "b", Tokens: 60}); v != VerdictQueued {
		t.Fatalf("second 60-token request: verdict = %s, want queued", v)
	}
	if got := c.Schedule(); len(got) != 0 {
		t.Fatalf("Schedule with the budget full admitted %v, want nothing", got)
	}
	if st := c.Stats(); st.Running != 1 || st.TokensInUse != 60 || st.Waiting != 1 {
		t.Fatalf("before completion: running=%d tokens=%d waiting=%d, want 1/60/1", st.Running, st.TokensInUse, st.Waiting)
	}
	// A frees its 60 tokens; the next Schedule round admits the waiter that now fits.
	if !c.Complete("a") {
		t.Fatal("Complete(a) reported not-running")
	}
	got := c.Schedule()
	if len(got) != 1 || got[0].TraceID != "b" {
		t.Fatalf("after Complete(a) Schedule admitted %v, want [b]", got)
	}
	if st := c.Stats(); st.Running != 1 || st.TokensInUse != 60 || st.Waiting != 0 {
		t.Fatalf("after promotion: running=%d tokens=%d waiting=%d, want 1/60/0", st.Running, st.TokensInUse, st.Waiting)
	}
}

// TestAdmissionMaxNumSeqs is the AC#1 max-num-seqs witness: the running set is capped at
// max-num-seqs independently of the token budget — a third request waits behind a full
// 2-slot running set even with unlimited token headroom.
func TestAdmissionMaxNumSeqs(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 2, MaxWaiting: 16, AgingRounds: 1})

	if v := c.Offer(SeqRequest{TraceID: "a"}); v != VerdictAdmitted {
		t.Fatalf("a: verdict = %s, want admitted", v)
	}
	if v := c.Offer(SeqRequest{TraceID: "b"}); v != VerdictAdmitted {
		t.Fatalf("b: verdict = %s, want admitted", v)
	}
	if v := c.Offer(SeqRequest{TraceID: "c"}); v != VerdictQueued {
		t.Fatalf("c with the 2-seq cap full: verdict = %s, want queued", v)
	}
	if got := c.Schedule(); len(got) != 0 {
		t.Fatalf("Schedule at the seq cap admitted %v, want nothing", got)
	}
	if !c.Complete("a") {
		t.Fatal("Complete(a) reported not-running")
	}
	got := c.Schedule()
	if len(got) != 1 || got[0].TraceID != "c" {
		t.Fatalf("after a freed a slot, Schedule admitted %v, want [c]", got)
	}
}

// TestAdmissionPriorityDequeue is the AC#1 priority witness: with two requests waiting,
// the lower Priority value (higher priority) is admitted first when a slot frees.
func TestAdmissionPriorityDequeue(t *testing.T) {
	// AgingRounds large so a single round of aging cannot flip the raw priority order.
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 16, AgingRounds: 1_000_000})

	c.Offer(SeqRequest{TraceID: "blocker", Priority: 0}) // fills the single slot
	if v := c.Offer(SeqRequest{TraceID: "low", Priority: 5}); v != VerdictQueued {
		t.Fatalf("low: verdict = %s, want queued", v)
	}
	if v := c.Offer(SeqRequest{TraceID: "high", Priority: 1}); v != VerdictQueued {
		t.Fatalf("high: verdict = %s, want queued", v)
	}
	if !c.Complete("blocker") {
		t.Fatal("Complete(blocker) reported not-running")
	}
	got := c.Schedule()
	if len(got) != 1 || got[0].TraceID != "high" {
		t.Fatalf("priority dequeue admitted %v, want [high] (lower Priority value first)", got)
	}
}

// TestAdmissionNoStarvation is the AC#1 no-starvation witness, asserted in BOTH directions.
// With aging ON, a low-priority waiter is admitted within a BOUNDED number of rounds even
// under a continuous flood of higher-priority arrivals; with aging OFF, the same flood
// starves it indefinitely — proving the guard is load-bearing.
func TestAdmissionNoStarvation(t *testing.T) {
	const lowPriority = 10
	const agingRounds = 2
	// Effective priority of the waiter reaches the flood's value (0) after lowPriority*
	// agingRounds rounds, at which point the older enqueue tiebreak admits it. Allow slack.
	bound := lowPriority*agingRounds + 4

	// flood runs the scenario and returns the round at which "low" was admitted, or -1 if
	// it was never admitted within maxRounds.
	flood := func(aging int, maxRounds int) int {
		c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1024, AgingRounds: aging})
		c.Offer(SeqRequest{TraceID: "blocker", Priority: 0}) // occupy the single slot
		c.Offer(SeqRequest{TraceID: "low", Priority: lowPriority})
		running := "blocker"
		for r := 0; r < maxRounds; r++ {
			// A fresh top-priority arrival every round — the starvation pressure.
			c.Offer(SeqRequest{TraceID: floodID(r), Priority: 0})
			c.Complete(running) // the slot frees
			admitted := c.Schedule()
			if len(admitted) != 1 {
				t.Fatalf("round %d (aging=%d): Schedule admitted %d, want exactly 1", r, aging, len(admitted))
			}
			running = admitted[0].TraceID
			if running == "low" {
				return r
			}
		}
		return -1
	}

	// Aging ON: admitted within the bound.
	gotRound := flood(agingRounds, bound)
	if gotRound < 0 {
		t.Fatalf("with aging on, low was NOT admitted within %d rounds — starved", bound)
	}

	// Aging OFF: the flood starves it — never admitted even over many more rounds.
	if r := flood(0, bound*5); r >= 0 {
		t.Fatalf("with aging off, low was admitted at round %d; expected starvation under the flood", r)
	}
}

// TestAdmissionTrustDenyRejects is the AC#3 witness: a request carrying a denying trust
// verdict is rejected outright — never admitted, never queued — even with free headroom,
// and the rejection maps to HTTP 403.
func TestAdmissionTrustDenyRejects(t *testing.T) {
	c := NewAdmissionController(DefaultAdmissionPolicy()) // ample headroom

	v := c.Offer(SeqRequest{TraceID: "t", Tokens: 1, Trust: AdmissionTrust{Deny: true, Reason: "TENANT_OVER_SLA"}})
	if v != VerdictDenied {
		t.Fatalf("denying trust verdict: verdict = %s, want denied", v)
	}
	if v.HTTPStatus() != 403 {
		t.Fatalf("denied HTTPStatus = %d, want 403", v.HTTPStatus())
	}
	if st := c.Stats(); st.Running != 0 || st.Waiting != 0 || st.Denied != 1 {
		t.Fatalf("after deny: running=%d waiting=%d denied=%d, want 0/0/1", st.Running, st.Waiting, st.Denied)
	}
	// Schedule must never resurrect a denied request.
	if got := c.Schedule(); len(got) != 0 {
		t.Fatalf("Schedule admitted %v after a deny, want nothing", got)
	}
}

// TestAdmissionShed429 is the AC#2 backpressure witness: driven past the bound (running set
// saturated AND the waiting queue at its bound), the gate sheds the next request rather
// than queueing it unboundedly, and the shed maps to HTTP 429.
func TestAdmissionShed429(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})

	if v := c.Offer(SeqRequest{TraceID: "a"}); v != VerdictAdmitted {
		t.Fatalf("a: verdict = %s, want admitted", v)
	}
	if v := c.Offer(SeqRequest{TraceID: "b"}); v != VerdictQueued { // fills the 1-deep waiting bound
		t.Fatalf("b: verdict = %s, want queued", v)
	}
	// Running saturated (1/1) and waiting at bound (1/1): the node sheds.
	v := c.Offer(SeqRequest{TraceID: "c"})
	if v != VerdictShed {
		t.Fatalf("c past the bound: verdict = %s, want shed", v)
	}
	if v.HTTPStatus() != 429 {
		t.Fatalf("shed HTTPStatus = %d, want 429", v.HTTPStatus())
	}
	if st := c.Stats(); st.Shed != 1 || st.Running != 1 || st.Waiting != 1 {
		t.Fatalf("after shed: shed=%d running=%d waiting=%d, want 1/1/1", st.Shed, st.Running, st.Waiting)
	}
}

// TestAdmissionMetricsFragment is the AC#4 witness: the running/waiting/admitted/queued/
// shed counts render into the shared L2 serving-metrics schema with the documented names
// and values.
func TestAdmissionMetricsFragment(t *testing.T) {
	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})
	c.Offer(SeqRequest{TraceID: "a", Tokens: 7}) // admitted -> running 1, tokens 7
	c.Offer(SeqRequest{TraceID: "b"})            // queued   -> waiting 1
	c.Offer(SeqRequest{TraceID: "c"})            // shed     -> shed_total 1

	var b strings.Builder
	c.WriteMetrics(&b)
	out := b.String()

	for _, want := range []string{
		"fak_sched_running 1",
		"fak_sched_waiting 1",
		"fak_sched_tokens_in_use 7",
		"fak_sched_max_num_seqs 1",
		"fak_sched_admitted_total 1",
		"fak_sched_queued_total 1",
		"fak_sched_shed_total 1",
		"fak_sched_denied_total 0",
		"# TYPE fak_sched_running gauge",
		"# TYPE fak_sched_admitted_total counter",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics fragment missing %q in:\n%s", want, out)
		}
	}
}

// TestAdmissionControllerRendersIntoLiveMetrics is the issue-#35 AC#4 LIVE-surface witness:
// a Server emits no fak_sched_* family until a controller is wired (no phantom zero series);
// once a host attaches one with SetAdmissionController, scraping the real /metrics render
// surfaces the admission gate's running/waiting/admitted counts in the shared L2 serving-
// metrics schema; detaching with nil takes the family back off the surface. This proves the
// schema is exported per-worker on the live surface, not only in the WriteMetrics unit above.
func TestAdmissionControllerRendersIntoLiveMetrics(t *testing.T) {
	srv := newTestServer(t)

	// No controller attached -> the family is absent from the live surface (inert by default).
	if pre := srv.renderMetrics(); strings.Contains(pre, schedMetricPrefix) {
		t.Fatalf("fak_sched_* present before SetAdmissionController:\n%s", pre)
	}

	c := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})
	c.Offer(SeqRequest{TraceID: "a", Tokens: 7}) // admitted -> running 1, tokens 7
	c.Offer(SeqRequest{TraceID: "b"})            // queued   -> waiting 1
	c.Offer(SeqRequest{TraceID: "c"})            // shed     -> shed_total 1
	srv.SetAdmissionController(c)

	out := srv.renderMetrics()
	for _, want := range []string{
		"fak_sched_running 1",
		"fak_sched_waiting 1",
		"fak_sched_tokens_in_use 7",
		"fak_sched_admitted_total 1",
		"fak_sched_queued_total 1",
		"fak_sched_shed_total 1",
		"fak_sched_denied_total 0",
		"# TYPE fak_sched_running gauge",
		"# TYPE fak_sched_admitted_total counter",
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}

	// Detaching takes the family back off the surface — host-injected, inert by default.
	srv.SetAdmissionController(nil)
	if post := srv.renderMetrics(); strings.Contains(post, schedMetricPrefix) {
		t.Fatalf("fak_sched_* still present after detaching the controller:\n%s", post)
	}
}

// TestNativeGatewayNewAttachesAdmissionController is the issue-#35 live-worker witness:
// a native in-kernel gateway gets the default admission controller at construction time,
// so a real fak-native worker exposes fak_sched_* load without a separate CLI-only attach.
// Proxy/mock gateways stay inert unless their host explicitly calls SetAdmissionController.
func TestNativeGatewayNewAttachesAdmissionController(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	noop := func(string, ...any) {}

	native, err := New(Config{
		EngineID:      "test",
		Model:         "test-model",
		InKernelModel: &model.Model{},
		Tokenizer:     &tokenizer.Tokenizer{},
		Logf:          noop,
	})
	if err != nil {
		t.Fatalf("New(native): %v", err)
	}
	t.Cleanup(native.Close)
	if native.admissionCtl == nil {
		t.Fatal("native admission controller is nil, want default controller attached")
	}
	if out := native.renderMetrics(); !strings.Contains(out, "fak_sched_max_num_seqs 256\n") {
		t.Fatalf("native /metrics missing default fak_sched_* family:\n%s", out)
	}

	proxy, err := New(Config{
		EngineID:      "test",
		Model:         "test-model",
		BaseURL:       "http://127.0.0.1:1",
		InKernelModel: &model.Model{},
		Tokenizer:     &tokenizer.Tokenizer{},
		Logf:          noop,
	})
	if err != nil {
		t.Fatalf("New(proxy): %v", err)
	}
	t.Cleanup(proxy.Close)
	if proxy.admissionCtl != nil {
		t.Fatal("proxy admission controller attached by default, want native-only attachment")
	}
}

// TestServedAdmissionSheds429BeforePlanner is the live issue-#35 backpressure witness:
// with the native serving gate wired into the gateway, one request holds the only running
// slot, one request fills the waiting queue, and the next request gets a real HTTP 429
// before it reaches the planner instead of joining an unbounded queue.
func TestServedAdmissionSheds429BeforePlanner(t *testing.T) {
	srv := newTestServer(t)
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})
	srv.SetAdmissionController(ctl)

	planner := newBlockingAdmissionPlanner()
	defer planner.Release()
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	first := make(chan int, 1)
	go func() { first <- postAdmissionChat(t, ts.URL, "first") }()
	select {
	case <-planner.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("first request never reached the planner")
	}

	second := make(chan int, 1)
	go func() { second <- postAdmissionChat(t, ts.URL, "second") }()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if st := ctl.Stats(); st.Running == 1 && st.Waiting == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("second request did not fill waiting queue; stats=%+v", ctl.Stats())
		}
		time.Sleep(10 * time.Millisecond)
	}

	if status := postAdmissionChat(t, ts.URL, "third"); status != http.StatusTooManyRequests {
		t.Fatalf("third status = %d, want 429", status)
	}
	if calls := planner.Calls(); calls != 1 {
		t.Fatalf("planner calls after shed = %d, want only the first request reached planner", calls)
	}

	planner.Release()
	if status := <-first; status != http.StatusOK {
		t.Fatalf("first status = %d, want 200", status)
	}
	if status := <-second; status != http.StatusOK {
		t.Fatalf("second status = %d, want 200 after first releases", status)
	}
	if st := ctl.Stats(); st.Running != 0 || st.Waiting != 0 || st.Shed != 1 || st.Admitted != 2 {
		t.Fatalf("final admission stats = %+v, want running=0 waiting=0 shed=1 admitted=2", st)
	}
}

// floodID names the r-th flood arrival without time/randomness, so the scenario is
// byte-reproducible across machines.
func floodID(r int) string {
	return "flood-" + strconv.Itoa(r)
}

type blockingAdmissionPlanner struct {
	entered     chan struct{}
	release     chan struct{}
	enterOnce   sync.Once
	releaseOnce sync.Once
	mu          sync.Mutex
	calls       int
}

func newBlockingAdmissionPlanner() *blockingAdmissionPlanner {
	return &blockingAdmissionPlanner{entered: make(chan struct{}), release: make(chan struct{})}
}

func (p *blockingAdmissionPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	p.calls++
	p.mu.Unlock()
	p.enterOnce.Do(func() { close(p.entered) })
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		Model:        "admission-test",
	}, nil
}

func (*blockingAdmissionPlanner) Model() string { return "admission-test" }

func (p *blockingAdmissionPlanner) Release() {
	p.releaseOnce.Do(func() { close(p.release) })
}

func (p *blockingAdmissionPlanner) Calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func postAdmissionChat(t *testing.T, base, trace string) int {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":1}`))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(traceHeader, trace)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST chat: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}
