package hwutil_test

// Fake-fabric contract test in the device_spine style: a fake transport
// executor moves bytes through the REAL fabricmap.Execute path, the hop
// observer forwards validated receipts into a Recorder exactly like the
// fabricmapdemo bridge does, and the emitted tier rows are asserted to the
// byte. No hardware, no nvidia-smi, no WMI — only the closed wire contract.

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/pkg/hwutil"
)

// fakeHopExecutor witnesses each directed link with a valid receipt whose
// timing spans the requested movement window.
func fakeHopExecutor(hopDur time.Duration) fabricmap.Executor {
	return fabricmap.ExecutorFunc(func(_ context.Context, r fabricmap.AuthorizationRequest, n uint64) (fabricmap.HopReceipt, error) {
		start := time.Now()
		return fabricmap.HopReceipt{
			HopIndex:    r.HopIndex,
			LinkID:      r.Link.ID,
			From:        r.Link.From,
			To:          r.Link.To,
			Transport:   r.Link.Transport,
			Bytes:       n,
			StartedAt:   start,
			CompletedAt: start.Add(hopDur),
			Integrity:   fabricmap.Integrity{Algorithm: "sha256", Digest: "fake-digest"},
			State:       fabricmap.HopComplete,
		}, nil
	})
}

// bridge mirrors cmd/fabricmapdemo's registerHwutilHopObserver mapping so the
// test pins the same endpoint-kind -> tier vocabulary the demo ships; it
// returns the observer's unregister function.
func bridge(rec *hwutil.Recorder, g fabricmap.Graph) func() {
	tierByID := make(map[string]hwutil.TierKind, len(g.Endpoints))
	for _, ep := range g.Endpoints {
		switch ep.Kind {
		case "gpu-memory":
			tierByID[ep.ID] = hwutil.TierVRAM
		case "host-memory":
			tierByID[ep.ID] = hwutil.TierRAM
		case "ssd":
			tierByID[ep.ID] = hwutil.TierSSD
		}
	}
	return fabricmap.RegisterHopObserver(func(rc fabricmap.HopReceipt) {
		from, okFrom := tierByID[rc.From]
		to, okTo := tierByID[rc.To]
		if !okFrom || !okTo {
			return
		}
		rec.RecordTransfer(from, to, int64(rc.Bytes), rc.CompletedAt.Sub(rc.StartedAt), hwutil.TransferUnknown)
	})
}

func TestTierFakeFabricMovesBytesBetweenTiers(t *testing.T) {
	g := fabricmap.Graph{
		Endpoints: []fabricmap.Endpoint{
			{ID: "L1", Kind: "gpu-memory"},
			{ID: "L2", Kind: "host-memory"},
			{ID: "L3", Kind: "ssd"},
		},
		Links: []fabricmap.Link{
			{ID: "ssd-to-host", From: "L3", To: "L2", Transport: "nvme-copy"},
			{ID: "host-to-gpu", From: "L2", To: "L1", Transport: "pcie-copy"},
		},
	}
	route := fabricmap.Route{From: "L3", To: "L1", Links: g.Links}
	authorized := fabricmap.AuthorizedRoute{Route: route, Access: fabricmap.Access{Principal: "tier-test", Operation: fabricmap.OperationCopy}}

	rec := hwutil.NewRecorder(16)
	unreg := bridge(rec, g)
	defer unreg()

	const bytes = 4 << 30 // 4 GiB staged load
	receipt, err := fabricmap.Execute(context.Background(), authorized, bytes, fakeHopExecutor(50*time.Millisecond))
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if receipt.State != fabricmap.ExecutionComplete || len(receipt.Hops) != 2 {
		t.Fatalf("receipt = %+v", receipt)
	}

	tiers, transfers := rec.Drain()
	if len(tiers) != 0 {
		t.Fatalf("fabric hops record movements, not residency bands: %+v", tiers)
	}
	if len(transfers) != 2 {
		t.Fatalf("transfers = %+v, want one row per witnessed hop", transfers)
	}
	first, second := transfers[0], transfers[1]
	if first.FromTier != "ssd" || first.ToTier != "ram" || first.Bytes != bytes || first.Reason != hwutil.TransferUnknown {
		t.Fatalf("hop 1 row = %+v", first)
	}
	if second.FromTier != "ram" || second.ToTier != "vram" || second.Bytes != bytes || second.Reason != hwutil.TransferUnknown {
		t.Fatalf("hop 2 row = %+v", second)
	}
	// Measured rate must reflect the fake hop window (~50ms -> ~80 GiB/s for 4GiB),
	// pinned loosely to stay deterministic across scheduling jitter.
	if first.RateGiBS < 40 || first.RateGiBS > 200 {
		t.Fatalf("hop 1 rate %.2f GiB/s outside plausible band for 4GiB/50ms", first.RateGiBS)
	}
	if second.RateGiBS < 40 || second.RateGiBS > 200 {
		t.Fatalf("hop 2 rate %.2f GiB/s outside plausible band for 4GiB/50ms", second.RateGiBS)
	}
	// The rows validate against the frozen snapshot shape.
	snap := hwutil.Snapshot{Schema: hwutil.Schema, Host: "h", Arch: "amd64", Os: "linux", At: time.Now().UTC(), Transfers: transfers}
	if err := snap.Validate(); err != nil {
		t.Fatalf("emitted transfer rows violate the frozen schema: %v", err)
	}
}

func TestTierFailedFabricHopRecordsNothing(t *testing.T) {
	g := fabricmap.Graph{
		Endpoints: []fabricmap.Endpoint{{ID: "L2", Kind: "host-memory"}, {ID: "L3", Kind: "ssd"}},
		Links:     []fabricmap.Link{{ID: "ssd-to-host", From: "L3", To: "L2", Transport: "nvme-copy"}},
	}
	authorized := fabricmap.AuthorizedRoute{
		Route:  fabricmap.Route{From: "L3", To: "L2", Links: g.Links},
		Access: fabricmap.Access{Principal: "tier-test", Operation: fabricmap.OperationRead},
	}
	rec := hwutil.NewRecorder(16)
	unreg := bridge(rec, g)
	defer unreg()

	failExec := fabricmap.ExecutorFunc(func(context.Context, fabricmap.AuthorizationRequest, uint64) (fabricmap.HopReceipt, error) {
		return fabricmap.HopReceipt{}, context.DeadlineExceeded
	})
	if _, err := fabricmap.Execute(context.Background(), authorized, 1024, failExec); err == nil {
		t.Fatal("expected the failing hop to refuse execution")
	}
	if tiers, transfers := rec.Drain(); tiers != nil || transfers != nil {
		t.Fatalf("failed movement must not emit rows: (%v, %v)", tiers, transfers)
	}
}
