package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestLeaseWriteRenewAndReleaseRejectStaleSameHolderGeneration(t *testing.T) {
	useLeaseWriteTestRepo(t)
	previous := leasePublish
	leasePublish = func(context.Context, *leaseref.Store) {}
	t.Cleanup(func() {
		<-leasePublishes.idleC()
		leasePublish = previous
	})

	ctx := context.Background()
	acquire := func() gateway.LeaseWriteResult {
		res, err := serveLeaseWrite(ctx, "acquire", gateway.LeaseWriteRequest{ID: "endpoint-lane", Holder: "worker", TTLSeconds: 300})
		if err != nil || !res.OK {
			t.Fatalf("acquire: result=%+v err=%v", res, err)
		}
		return res
	}
	first := acquire()
	released, err := serveLeaseWrite(ctx, "release", gateway.LeaseWriteRequest{ID: first.ID, Holder: first.Holder, Generation: first.Generation})
	if err != nil || !released.OK {
		t.Fatalf("release first epoch: result=%+v err=%v", released, err)
	}
	second := acquire()
	if second.Generation <= first.Generation {
		t.Fatalf("reacquired generation=%d, want > %d", second.Generation, first.Generation)
	}

	for _, op := range []string{"renew", "release"} {
		res, err := serveLeaseWrite(ctx, op, gateway.LeaseWriteRequest{ID: first.ID, Holder: first.Holder, Generation: first.Generation, TTLSeconds: 600})
		if err != nil {
			t.Fatalf("stale %s: %v", op, err)
		}
		if res.OK || res.Reason != leaseref.ReasonStaleLease || res.Generation != first.Generation || res.CurrentGeneration != second.Generation {
			t.Fatalf("stale %s result=%+v, want STALE_LEASE presented=%d current=%d", op, res, first.Generation, second.Generation)
		}
	}

	store := leaseref.NewInDir(leasePlaneDir())
	got, ok, err := store.Get(ctx, first.ID)
	if err != nil || !ok || got.Generation != second.Generation {
		t.Fatalf("stale endpoint write changed replacement: got=%+v ok=%v err=%v", got, ok, err)
	}
}
