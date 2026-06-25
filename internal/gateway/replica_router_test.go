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
