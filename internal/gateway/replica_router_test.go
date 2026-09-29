package gateway

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type replicaDispatchTestPlanner struct {
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

func (p *replicaDispatchTestPlanner) Model() string { return p.name }

func (p *replicaDispatchTestPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(false, messages, tools, opts...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.name}, Model: p.name}, nil
}

func (p *replicaDispatchTestPlanner) StreamingSupported() bool {
	return p.streaming && p.streamingSupported
}

func (p *replicaDispatchTestPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
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

func (p *replicaDispatchTestPlanner) record(stream bool, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) {
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

func (p *replicaDispatchTestPlanner) counts() (complete, stream int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.completeN, p.streamN
}

func (p *replicaDispatchTestPlanner) samples() []agent.SampleParams {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]agent.SampleParams(nil), p.gotSamples...)
}

func TestReplicaDispatchValidatesStaticRegistry(t *testing.T) {
	replica := &replicaDispatchTestPlanner{name: "r1"}
	tests := []struct {
		name     string
		model    string
		replicas []PlannerReplica
		wantIs   error
	}{
		{name: "empty model", replicas: []PlannerReplica{{Name: "r1", Planner: replica}}},
		{name: "empty replicas", model: "fleet", wantIs: ErrReplicaDispatchEmpty},
		{name: "empty replica name", model: "fleet", replicas: []PlannerReplica{{Planner: replica}}},
		{name: "nil planner", model: "fleet", replicas: []PlannerReplica{{Name: "r1"}}},
		{name: "duplicate name", model: "fleet", replicas: []PlannerReplica{{Name: "r1", Planner: replica}, {Name: "r1", Planner: replica}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewReplicaDispatch(tt.model, tt.replicas)
			if err == nil {
				t.Fatalf("NewReplicaDispatch() succeeded, want validation error")
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Fatalf("NewReplicaDispatch() error = %v, want %v", err, tt.wantIs)
			}
		})
	}
}

func TestReplicaDispatchKeyedPlacementAndForwardsInputs(t *testing.T) {
	a := &replicaDispatchTestPlanner{name: "r1"}
	b := &replicaDispatchTestPlanner{name: "r2"}
	c := &replicaDispatchTestPlanner{name: "r3"}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "a", Planner: a},
		{Name: "b", Planner: b},
		{Name: "c", Planner: c},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
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
	// A keyed rendezvous is deterministic: the SAME request always lands on the
	// same replica. Five identical requests all hit one winner (no rotation).
	var got []string
	for i := 0; i < 5; i++ {
		comp, err := router.Complete(context.Background(), messages, tools, agent.WithMaxTokens(17), agent.WithModel("client-model"))
		if err != nil {
			t.Fatalf("Complete(%d): %v", i, err)
		}
		got = append(got, comp.Message.Content)
	}
	for i := 1; i < len(got); i++ {
		if got[i] != got[0] {
			t.Fatalf("identical requests were placed on different replicas: %v (placement is not a pure function of the request)", got)
		}
	}
	// Distinct request identities spread across the replica set: over enough
	// prefixes every replica is chosen at least once (no counter involved).
	seen := map[string]bool{}
	for i := 0; i < 64; i++ {
		key := fmt.Sprintf("distinct-prefix-%d", i)
		repl, err := router.pickKeyed([]string{key})
		if err != nil {
			t.Fatalf("pickKeyed(%s): %v", key, err)
		}
		seen[repl.Name] = true
	}
	if len(seen) != 3 {
		t.Fatalf("keyed rendezvous spread over %d/3 replicas: %v", len(seen), seen)
	}
	for _, planner := range []*replicaDispatchTestPlanner{a, b, c} {
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

func TestReplicaDispatchCompleteIsConcurrentSafe(t *testing.T) {
	a := &replicaDispatchTestPlanner{name: "r1"}
	b := &replicaDispatchTestPlanner{name: "r2"}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{{Name: "a", Planner: a}, {Name: "b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
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
	// Keyed placement is deterministic for a fixed request identity, so every
	// identical concurrent call lands on the SAME replica (no cursor to race).
	if aComplete+bComplete != calls {
		t.Fatalf("complete counts = r1:%d r2:%d, want %d total", aComplete, bComplete, calls)
	}
	if aComplete != calls && bComplete != calls {
		t.Fatalf("identical concurrent requests were split across replicas: r1:%d r2:%d", aComplete, bComplete)
	}
}

// dispatchCounts runs n Completes with DISTINCT request identities (a distinct
// leading message yields a distinct rendezvous key) and tallies which replica
// served each. Because placement is a pure function of request identity, distinct
// requests spread across the eligible replicas with no counter.
func dispatchCounts(t *testing.T, router *ReplicaDispatch, n int) map[string]int {
	t.Helper()
	got := make(map[string]int)
	for i := 0; i < n; i++ {
		messages := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("dispatch-%d", i)}}
		comp, err := router.Complete(context.Background(), messages, nil)
		if err != nil {
			t.Fatalf("Complete(%d): %v", i, err)
		}
		got[comp.Message.Content]++
	}
	return got
}

func TestReplicaDispatchRoutesOnlyToHealthyWorkersWithHysteresis(t *testing.T) {
	a := &replicaDispatchTestPlanner{name: "ra"}
	b := &replicaDispatchTestPlanner{name: "rb"}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
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

	// One probe tick admits both (HealthyAfter=1); distinct request identities
	// spread across them with no counter.
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 16); got["ra"] == 0 || got["rb"] == 0 || got["ra"]+got["rb"] != 16 {
		t.Fatalf("healthy fleet keyed spread = %v, want both replicas serving 16 requests", got)
	}

	// w-b begins failing. ONE failed tick must NOT evict it (UnhealthyAfter=2).
	mu.Lock()
	healthy["w-b"] = false
	mu.Unlock()
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 16); got["rb"] == 0 {
		t.Fatalf("single transient failure flapped w-b out of rotation: %v", got)
	}

	// A second consecutive failure crosses w-b to unhealthy: the router drops it
	// within the bounded health interval and every request now lands on w-a.
	mem.ProbeOnce(ctx)
	if got := dispatchCounts(t, router, 16); got["rb"] != 0 || got["ra"] != 16 {
		t.Fatalf("unhealthy worker still in rotation: %v, want ra:16 rb:0", got)
	}
}

func TestReplicaDispatchDrainStopsNewWorkAndTypedVerdict(t *testing.T) {
	a := &replicaDispatchTestPlanner{name: "ra"}
	b := &replicaDispatchTestPlanner{name: "rb"}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
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

// TestReplicaDispatchWithoutMembershipStaysPolicyFree pins the opt-in contract: a
// router with no membership attached keeps the pure keyed placement over every
// replica (still spreading distinct requests, still deterministic per request).
func TestReplicaDispatchWithoutMembershipStaysPolicyFree(t *testing.T) {
	a := &replicaDispatchTestPlanner{name: "ra"}
	b := &replicaDispatchTestPlanner{name: "rb"}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{{Name: "w-a", Planner: a}, {Name: "w-b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	if got := dispatchCounts(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("policy-free keyed spread = %v, want ra:2 rb:2", got)
	}
}

func TestReplicaDispatchStreamsOnlyWhenEveryReplicaSupportsStreaming(t *testing.T) {
	streaming := &replicaDispatchTestPlanner{name: "stream", streaming: true, streamingSupported: true}
	buffered := &replicaDispatchTestPlanner{name: "buffered"}
	mixed, err := NewReplicaDispatch("fleet", []PlannerReplica{{Name: "stream", Planner: streaming}, {Name: "buffered", Planner: buffered}})
	if err != nil {
		t.Fatalf("NewReplicaDispatch mixed: %v", err)
	}
	if mixed.StreamingSupported() {
		t.Fatalf("mixed router advertised streaming support")
	}
	// Route one request at the streaming replica and one at the buffered replica by
	// keyed identity; the buffered winner must surface ErrStreamingUnsupported.
	streamMsg := messageKeyingTo(t, mixed, "stream")
	if _, err := mixed.CompleteStream(context.Background(), nil, streamMsg, nil); err != nil {
		t.Fatalf("first streaming replica should stream: %v", err)
	}
	bufferedMsg := messageKeyingTo(t, mixed, "buffered")
	if _, err := mixed.CompleteStream(context.Background(), nil, bufferedMsg, nil); !errors.Is(err, agent.ErrStreamingUnsupported) {
		t.Fatalf("buffered replica error = %v, want ErrStreamingUnsupported", err)
	}

	a := &replicaDispatchTestPlanner{name: "a", streaming: true, streamingSupported: true}
	b := &replicaDispatchTestPlanner{name: "b", streaming: true, streamingSupported: true}
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{{Name: "a", Planner: a}, {Name: "b", Planner: b}})
	if err != nil {
		t.Fatalf("NewReplicaDispatch streaming: %v", err)
	}
	if !router.StreamingSupported() {
		t.Fatalf("streaming router did not advertise streaming support")
	}
	// Distinct request identities spread across both streaming replicas.
	seen := map[string]bool{}
	for i := 0; i < 8; i++ {
		comp, err := router.CompleteStream(context.Background(), func(delta string) error {
			seen[delta] = true
			return nil
		}, []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("stream-%d", i)}}, nil)
		if err != nil {
			t.Fatalf("CompleteStream(%d): %v", i, err)
		}
		if comp == nil {
			t.Fatalf("CompleteStream(%d): nil completion", i)
		}
	}
	if !seen["a"] || !seen["b"] {
		t.Fatalf("streaming replicas served = %v, want both a and b over distinct requests", seen)
	}
}

// messageKeyingTo finds a synthetic request whose shared-prefix rendezvous selects
// the replica named want, so a test can steer a keyed (counter-free) dispatch to a
// specific replica without reaching into its internals.
func messageKeyingTo(t *testing.T, router *ReplicaDispatch, want string) []agent.Message {
	t.Helper()
	for i := 0; i < 4096; i++ {
		messages := []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("steer-%s-%d", want, i)}}
		repl, err := router.pickKeyed(prefixSegments(messages))
		if err != nil {
			t.Fatalf("pickKeyed: %v", err)
		}
		if repl.Name == want {
			return messages
		}
	}
	t.Fatalf("no synthetic request in 4096 selects replica %q", want)
	return nil
}
