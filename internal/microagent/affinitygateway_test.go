package microagent

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type namedGateway struct {
	name    string
	entered chan struct{}
	release chan struct{}
	mu      sync.Mutex
	calls   int
}

func (g *namedGateway) Model() string { return g.name }

func (g *namedGateway) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	if g.entered != nil {
		g.entered <- struct{}{}
	}
	if g.release != nil {
		select {
		case <-g.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &agent.Completion{Model: g.name, Message: agent.Message{Content: g.name}}, nil
}

func newAffinityTestGateway(t *testing.T, cfg AffinityGatewayConfig, gateways ...*namedGateway) *SessionAffinityGateway {
	t.Helper()
	seats := make([]GatewaySeat, 0, len(gateways))
	for i, gw := range gateways {
		seats = append(seats, GatewaySeat{ID: string(rune('a' + i)), Gateway: gw, Scheduler: NewScheduler(1)})
	}
	router, err := NewSessionAffinityGateway(seats, session.NewTable(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func completeAffinity(t *testing.T, gw Gateway, trace, key string) *agent.Completion {
	t.Helper()
	ctx := WithTrace(context.Background(), trace)
	if key != "" {
		ctx = WithAffinity(ctx, key)
	}
	got, err := gw.Complete(ctx, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestSessionAffinityGatewayInvokesSameRealSeatForStableKey(t *testing.T) {
	a, b := &namedGateway{name: "provider-a"}, &namedGateway{name: "provider-b"}
	var selected []AffinitySelection
	router := newAffinityTestGateway(t, AffinityGatewayConfig{Observe: func(s AffinitySelection) { selected = append(selected, s) }}, a, b)
	first := completeAffinity(t, router, "trace-1", "shared-prefix")
	second := completeAffinity(t, router, "trace-2", "shared-prefix")
	if first.Model != second.Model || len(selected) != 2 || selected[0].SeatID != selected[1].SeatID {
		t.Fatalf("first=%+v second=%+v selected=%+v", first, second, selected)
	}
}

func TestSessionAffinityGatewayFallsBackToDistinctGatewayWhenPreferredBusy(t *testing.T) {
	a := &namedGateway{name: "provider-a", entered: make(chan struct{}, 1), release: make(chan struct{})}
	b := &namedGateway{name: "provider-b", entered: make(chan struct{}, 1), release: make(chan struct{})}
	router := newAffinityTestGateway(t, AffinityGatewayConfig{}, a, b)
	ctx := WithAffinity(WithTrace(context.Background(), "one"), "shared-prefix")
	firstDone := make(chan *agent.Completion, 1)
	go func() { got, _ := router.Complete(ctx, nil, nil); firstDone <- got }()
	select {
	case <-a.entered:
	case <-b.entered:
	}
	secondDone := make(chan *agent.Completion, 1)
	go func() {
		got, _ := router.Complete(WithAffinity(WithTrace(context.Background(), "two"), "shared-prefix"), nil, nil)
		secondDone <- got
	}()
	select {
	case <-a.entered:
	case <-b.entered:
	}
	close(a.release)
	close(b.release)
	first, second := <-firstDone, <-secondDone
	if first.Model == second.Model {
		t.Fatalf("busy affinity did not reach distinct fallback gateway: %s", first.Model)
	}
}

func TestSessionAffinityGatewayOffRotatesFirstSeat(t *testing.T) {
	a, b := &namedGateway{name: "provider-a"}, &namedGateway{name: "provider-b"}
	router := newAffinityTestGateway(t, AffinityGatewayConfig{DisableAffinity: true}, a, b)
	first := completeAffinity(t, router, "trace", "same")
	second := completeAffinity(t, router, "trace", "same")
	if first.Model == second.Model {
		t.Fatalf("affinity-off did not rotate seats: %s", first.Model)
	}
}

func TestSessionAffinityGatewayUsesTraceByDefault(t *testing.T) {
	a, b := &namedGateway{name: "provider-a"}, &namedGateway{name: "provider-b"}
	var selected []AffinitySelection
	router := newAffinityTestGateway(t, AffinityGatewayConfig{Observe: func(s AffinitySelection) { selected = append(selected, s) }}, a, b)
	completeAffinity(t, router, "stable-trace", "")
	completeAffinity(t, router, "stable-trace", "")
	if len(selected) != 2 || selected[0].AffinityKey != "stable-trace" || selected[0].SeatID != selected[1].SeatID {
		t.Fatalf("selected=%+v", selected)
	}
}

func TestSessionAffinityGatewayRejectsInvalidSeat(t *testing.T) {
	if _, err := NewSessionAffinityGateway([]GatewaySeat{{ID: "missing"}}, session.NewTable(), AffinityGatewayConfig{}); err == nil {
		t.Fatal("invalid gateway seat accepted")
	}
}
