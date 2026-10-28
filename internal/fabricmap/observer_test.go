package fabricmap

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestTierHopObserverSeesOnlyValidatedHops(t *testing.T) {
	var seen atomic.Int64
	unreg := RegisterHopObserver(func(rc HopReceipt) { seen.Add(int64(rc.Bytes)) })
	defer unreg()

	g := Graph{
		Endpoints: []Endpoint{{ID: "L1"}, {ID: "L2"}},
		Links:     []Link{{ID: "l", From: "L1", To: "L2", Transport: "t"}},
	}
	authorized := AuthorizedRoute{Route: Route{From: "L1", To: "L2", Links: g.Links}}
	exec := ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
		return HopReceipt{
			HopIndex: r.HopIndex, LinkID: r.Link.ID, From: r.Link.From, To: r.Link.To,
			Transport: r.Link.Transport, Bytes: n, StartedAt: time.Now(), CompletedAt: time.Now(),
			Integrity: Integrity{Algorithm: "sha256", Digest: "d"}, State: HopComplete,
		}, nil
	})
	if _, err := Execute(context.Background(), authorized, 777, exec); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got := seen.Load(); got != 777 {
		t.Fatalf("observer saw %d bytes, want 777", got)
	}

	// Unregister stops delivery.
	unreg()
	seen.Store(0)
	if _, err := Execute(context.Background(), authorized, 1, exec); err != nil {
		t.Fatalf("Execute after unregister: %v", err)
	}
	if got := seen.Load(); got != 0 {
		t.Fatalf("unregistered observer still delivered %d bytes", got)
	}
}

func TestTierHopObserverPanicCannotBreakExecution(t *testing.T) {
	panicFlag := new(atomic.Bool)
	panicFlag.Store(true)
	unreg := RegisterHopObserver(func(HopReceipt) {
		if panicFlag.Load() {
			panic("telemetry bug")
		}
	})
	defer unreg()

	g := Graph{
		Endpoints: []Endpoint{{ID: "L1"}, {ID: "L2"}},
		Links:     []Link{{ID: "l", From: "L1", To: "L2", Transport: "t"}},
	}
	authorized := AuthorizedRoute{Route: Route{From: "L1", To: "L2", Links: g.Links}}
	exec := ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
		return HopReceipt{
			HopIndex: r.HopIndex, LinkID: r.Link.ID, From: r.Link.From, To: r.Link.To,
			Transport: r.Link.Transport, Bytes: n, StartedAt: time.Now(), CompletedAt: time.Now(),
			Integrity: Integrity{Algorithm: "sha256", Digest: "d"}, State: HopComplete,
		}, nil
	})
	receipt, err := Execute(context.Background(), authorized, 10, exec)
	if err != nil {
		t.Fatalf("observer panic leaked into execution: %v", err)
	}
	if receipt.State != ExecutionComplete || len(receipt.Hops) != 1 {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestTierHopObserverNilRegistrationIsNoop(t *testing.T) {
	unreg := RegisterHopObserver(nil)
	if unreg == nil {
		t.Fatal("nil registration must still return an unregister function")
	}
	unreg()
}
