package gateway

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type replicaRouterTestPlanner struct {
	name               string
	streaming          bool
	streamingSupported bool

	mu          sync.Mutex
	completeN   int
	streamN     int
	gotMessages [][]agent.Message
	gotTools    [][]agent.ToolDef
	gotSamples  []agent.SampleParams
}

func (p *replicaRouterTestPlanner) Model() string { return p.name }

func (p *replicaRouterTestPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(false, messages, tools, opts...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.name}, Model: p.name}, nil
}

func (p *replicaRouterTestPlanner) StreamingSupported() bool {
	return p.streaming && p.streamingSupported
}

func (p *replicaRouterTestPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if !p.StreamingSupported() {
		return nil, agent.ErrStreamingUnsupported
	}
	p.record(true, messages, tools, opts...)
	if sink != nil {
		if err := sink(p.name); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.name}, Model: p.name}, nil
}

func (p *replicaRouterTestPlanner) record(stream bool, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if stream {
		p.streamN++
	} else {
		p.completeN++
	}
	p.gotMessages = append(p.gotMessages, append([]agent.Message(nil), messages...))
	p.gotTools = append(p.gotTools, append([]agent.ToolDef(nil), tools...))
	p.gotSamples = append(p.gotSamples, sp)
}

func (p *replicaRouterTestPlanner) counts() (complete, stream int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.completeN, p.streamN
}

func (p *replicaRouterTestPlanner) samples() []agent.SampleParams {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]agent.SampleParams(nil), p.gotSamples...)
}

func TestReplicaRouterValidatesStaticRegistry(t *testing.T) {
	replica := &replicaRouterTestPlanner{name: "r1"}
	tests := []struct {
		name     string
		model    string
		replicas []PlannerReplica
		wantIs   error
	}{
		{name: "empty model", replicas: []PlannerReplica{{Name: "r1", Planner: replica}}},
		{name: "empty replicas", model: "fleet", wantIs: ErrReplicaRouterEmpty},
		{name: "empty replica name", model: "fleet", replicas: []PlannerReplica{{Planner: replica}}},
		{name: "nil planner", model: "fleet", replicas: []PlannerReplica{{Name: "r1"}}},
		{name: "duplicate name", model: "fleet", replicas: []PlannerReplica{{Name: "r1", Planner: replica}, {Name: "r1", Planner: replica}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewReplicaRouter(tt.model, tt.replicas)
			if err == nil {
				t.Fatalf("NewReplicaRouter() succeeded, want validation error")
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Fatalf("NewReplicaRouter() error = %v, want %v", err, tt.wantIs)
			}
		})
	}
}

func TestReplicaRouterRoundRobinsCompleteAndForwardsInputs(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "r1"}
	b := &replicaRouterTestPlanner{name: "r2"}
	c := &replicaRouterTestPlanner{name: "r3"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "a", Planner: a},
		{Name: "b", Planner: b},
		{Name: "c", Planner: c},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	if got := router.Model(); got != "fleet" {
		t.Fatalf("Model() = %q, want fleet", got)
	}
	wantRegistry := []ReplicaInfo{{Name: "a", Model: "r1"}, {Name: "b", Model: "r2"}, {Name: "c", Model: "r3"}}
	if got := router.Replicas(); !reflect.DeepEqual(got, wantRegistry) {
		t.Fatalf("Replicas() = %+v, want %+v", got, wantRegistry)
	}

	messages := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "search"}}}
	var got []string
	for i := 0; i < 5; i++ {
		comp, err := router.Complete(context.Background(), messages, tools, agent.WithMaxTokens(17), agent.WithModel("client-model"))
		if err != nil {
			t.Fatalf("Complete(%d): %v", i, err)
		}
		got = append(got, comp.Message.Content)
	}
	want := []string{"r1", "r2", "r3", "r1", "r2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round-robin = %v, want %v", got, want)
	}
	for _, planner := range []*replicaRouterTestPlanner{a, b, c} {
		for _, sp := range planner.samples() {
			if sp.MaxTokens == nil || *sp.MaxTokens != 17 {
				t.Fatalf("%s MaxTokens = %v, want 17", planner.name, sp.MaxTokens)
			}
			if sp.Model != "client-model" {
				t.Fatalf("%s Model sample = %q, want client-model", planner.name, sp.Model)
			}
		}
	}
}

func TestReplicaRouterCompleteIsConcurrentSafe(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "r1"}
	b := &replicaRouterTestPlanner{name: "r2"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{{Name: "a", Planner: a}, {Name: "b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	const calls = 100
	var wg sync.WaitGroup
	for i := 0; i < calls; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := router.Complete(context.Background(), nil, nil); err != nil {
				t.Errorf("Complete: %v", err)
			}
		}()
	}
	wg.Wait()
	aComplete, _ := a.counts()
	bComplete, _ := b.counts()
	if aComplete != calls/2 || bComplete != calls/2 {
		t.Fatalf("complete counts = r1:%d r2:%d, want %d each", aComplete, bComplete, calls/2)
	}
}

// dispatchCounts runs n round-robin Completes and tallies which replica served each
// (the test planner echoes its own model name as the completion content).
func dispatchCounts(t *testing.T, router *ReplicaRouter, n int) map[string]int {
	t.Helper()
	got := make(map[string]int)
	for i := 0; i < n; i++ {
		comp, err := router.Complete(context.Background(), nil, nil)
		if err != nil {
			t.Fatalf("Complete(%d): %v", i, err)
		}
		got[comp.Message.Content]++
	}
	return got
}

func TestReplicaRouterRoutesOnlyToHealthyWorkersWithHysteresis(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "ra"}
	b := &replicaRouterTestPlanner{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}

	var mu sync.Mutex
	healthy := map[string]bool{"w-a": true, "w-b": true}
	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2, // a single missed beat must not flap a worker out (hysteresis)
		Probe: func(_ context.Context, s WorkerSpec) bool {
			mu.Lock()
			defer mu.Unlock()
			return healthy[s.ID]
		},
	})
	for _, id := range []string{"w-a", "w-b"} {
		if err := mem.Add(WorkerSpec{ID: id, Endpoint: id}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	router.WithMembership(mem)
	ctx := context.Background()

	// Freshly registered workers are UNKNOWN, so nothing is admissible and the
	// router returns the typed verdict rather than route to an unprobed upstream.
	if _, err := router.Complete(ctx, nil, nil); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("unprobed fleet: Complete err = %v, want ErrNoHealthyWorker", err)
	}

	// One probe tick admits both (HealthyAfter=1); the router round-robins them.
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("healthy fleet round-robin = %v, want ra:2 rb:2", got)
	}

	// w-b begins failing. ONE failed tick must NOT evict it (UnhealthyAfter=2).
	mu.Lock()
	healthy["w-b"] = false
	mu.Unlock()
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 4); got["rb"] == 0 {
		t.Fatalf("single transient failure flapped w-b out of rotation: %v", got)
	}

	// A second consecutive failure crosses w-b to unhealthy: the router drops it
	// within the bounded health interval and every request now lands on w-a.
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 4); got["rb"] != 0 || got["ra"] != 4 {
		t.Fatalf("unhealthy worker still in rotation: %v, want ra:4 rb:0", got)
	}
}

func TestReplicaRouterDrainStopsNewWorkAndTypedVerdict(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "ra"}
	b := &replicaRouterTestPlanner{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 1,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range []string{"w-a", "w-b"} {
		if err := mem.Add(WorkerSpec{ID: id, Endpoint: id}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	router.WithMembership(mem)
	ctx := context.Background()
	mem.ProbeOnce(ctx) // both healthy

	// Drain w-a: the router must route NO new work to it while w-b keeps serving.
	if err := mem.Drain("w-a"); err != nil {
		t.Fatalf("Drain(w-a): %v", err)
	}
	if got := dispatchCounts(t, router, 4); got["ra"] != 0 || got["rb"] != 4 {
		t.Fatalf("drained worker still received new work: %v, want ra:0 rb:4", got)
	}

	// Drain the survivor too: with no admissible worker left, a pick is the typed
	// verdict, never a silent drop.
	if err := mem.Drain("w-b"); err != nil {
		t.Fatalf("Drain(w-b): %v", err)
	}
	if _, err := router.Complete(ctx, nil, nil); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("fully drained fleet: Complete err = %v, want ErrNoHealthyWorker", err)
	}
}

// TestReplicaRouterWithoutMembershipStaysBlindRoundRobin pins the opt-in contract:
// a router with no membership attached keeps the policy-free rotation unchanged.
func TestReplicaRouterWithoutMembershipStaysBlindRoundRobin(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "ra"}
	b := &replicaRouterTestPlanner{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{{Name: "w-a", Planner: a}, {Name: "w-b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	if got := dispatchCounts(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("blind round-robin = %v, want ra:2 rb:2", got)
	}
}

func TestReplicaRouterStreamsOnlyWhenEveryReplicaSupportsStreaming(t *testing.T) {
	streaming := &replicaRouterTestPlanner{name: "stream", streaming: true, streamingSupported: true}
	buffered := &replicaRouterTestPlanner{name: "buffered"}
	mixed, err := NewReplicaRouter("fleet", []PlannerReplica{{Name: "stream", Planner: streaming}, {Name: "buffered", Planner: buffered}})
	if err != nil {
		t.Fatalf("NewReplicaRouter mixed: %v", err)
	}
	if mixed.StreamingSupported() {
		t.Fatalf("mixed router advertised streaming support")
	}
	if _, err := mixed.CompleteStream(context.Background(), nil, nil, nil); err != nil {
		t.Fatalf("first streaming replica should stream: %v", err)
	}
	if _, err := mixed.CompleteStream(context.Background(), nil, nil, nil); !errors.Is(err, agent.ErrStreamingUnsupported) {
		t.Fatalf("second non-streaming replica error = %v, want ErrStreamingUnsupported", err)
	}

	a := &replicaRouterTestPlanner{name: "a", streaming: true, streamingSupported: true}
	b := &replicaRouterTestPlanner{name: "b", streaming: true, streamingSupported: true}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{{Name: "a", Planner: a}, {Name: "b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaRouter streaming: %v", err)
	}
	if !router.StreamingSupported() {
		t.Fatalf("streaming router did not advertise streaming support")
	}
	var deltas []string
	for i := 0; i < 2; i++ {
		_, err := router.CompleteStream(context.Background(), func(delta string) error {
			deltas = append(deltas, delta)
			return nil
		}, nil, nil)
		if err != nil {
			t.Fatalf("CompleteStream(%d): %v", i, err)
		}
	}
	if want := []string{"a", "b"}; !reflect.DeepEqual(deltas, want) {
		t.Fatalf("stream deltas = %v, want %v", deltas, want)
	}
}
