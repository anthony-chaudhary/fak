package modelengine

import (
	"context"
	"errors"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
func TestSelectPeerOrderFewestLanesTieLowestIndex(t *testing.T) {
	cases := []struct {
		name   string
		counts []int
		want   []int
	}{
		{name: "fewest first", counts: []int{2, 0, 1}, want: []int{1, 2, 0}},
		{name: "all tied falls to index", counts: []int{0, 0, 0}, want: []int{0, 1, 2}},
		{name: "all unknown falls to index", counts: []int{-1, -1, -1}, want: []int{0, 1, 2}},
		{name: "known beats unknown", counts: []int{-1, 3, -1}, want: []int{1, 0, 2}},
		{name: "single peer", counts: []int{7}, want: []int{0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := selectPeerOrder(tc.counts)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("selectPeerOrder(%v) = %v, want %v", tc.counts, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestParseActiveLanesSumsDecodeRoleGauges(t *testing.T) {
	exposition := `# HELP fak_native_pd_worker_requests_total Native prefill/decode worker requests by role.
# TYPE fak_native_pd_active_lanes gauge
fak_native_pd_active_lanes{role="decode",worker="decode-0"} 2
fak_native_pd_active_lanes{role="decode",worker="decode-1"} 3
fak_native_pd_worker_requests_total{role="prefill",worker="prefill-0"} 9
this line is not a metric
fak_native_pd_active_lanes{role="decode",worker="decode-2"} notanumber
`
	if got := ParseActiveLanes(exposition); got != 5 {
		t.Fatalf("ParseActiveLanes = %d, want 5 (decode-role gauges only)", got)
	}
	if got := ParseActiveLanes(""); got != 0 {
		t.Fatalf("ParseActiveLanes(empty) = %d, want 0", got)
	}
}

// pdRouterTestServer boots one decode cluster + wire server on loopback and
// reports its bound address.
func pdRouterTestServer(t *testing.T, m *model.Model) (addr string, cleanup func()) {
	t.Helper()
	cluster := NewNativePDCluster(m, 1)
	server := NewPDWireServer(cluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	return server.Addr().String(), func() {
		_ = server.Close()
		cluster.Close()
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDPeerRouterRetriesOnceOnDifferentPeer(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()

	// Peer 0 reports the fewest lanes but its wire plane is down; peer 1 is
	// healthy. The router must pick peer 0, fail, and retry ONCE on peer 1.
	liveAddr, cleanupLive := pdRouterTestServer(t, m)
	defer cleanupLive()
	deadAddr := deadLoopbackAddr(t)

	fetcher := func(_ context.Context, controlAddr string) (int, error) {
		switch controlAddr {
		case "ctrl-0":
			return 0, nil // fewest lanes, but the wire plane is dead
		case "ctrl-1":
			return 5, nil
		}
		return 0, errors.New("unknown control address")
	}
	router := NewPDPeerRouter(prefillCluster, []PDDecodePeer{
		{WireAddr: deadAddr, ControlAddr: "ctrl-0"},
		{WireAddr: liveAddr, ControlAddr: "ctrl-1"},
	}, fetcher)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	res, err := router.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt})
	if err != nil {
		t.Fatalf("router Ship: %v", err)
	}
	if want := serialGreedyTokens(m, prompt); !reflect.DeepEqual(res.Tokens, want) {
		t.Fatalf("routed tokens changed:\n got %v\nwant %v", res.Tokens, want)
	}

	ledger := router.RouteLedger()
	if len(ledger) != 1 {
		t.Fatalf("route ledger has %d receipts, want 1", len(ledger))
	}
	rec := ledger[0]
	if rec.Chosen != 0 || rec.RetryTo != 1 || rec.ServedBy != 1 {
		t.Fatalf("receipt = %+v, want chosen 0 retried to 1 served by 1", rec)
	}
	if !reflect.DeepEqual(rec.LaneCounts, []int{0, 5}) {
		t.Fatalf("receipt lane counts = %v, want [0 5] recorded per step", rec.LaneCounts)
	}
	if metrics := router.Metrics(); !strings.Contains(metrics, "fak_pd_router_retries_total 1") || !strings.Contains(metrics, "fak_pd_router_routes_total 1") {
		t.Fatalf("router metrics missing counters:\n%s", metrics)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDPeerRouterSecondFailureIsTypedRefusal(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()

	deadAddr := deadLoopbackAddr(t)
	fetcher := func(_ context.Context, _ string) (int, error) { return 0, nil }
	router := NewPDPeerRouter(prefillCluster, []PDDecodePeer{
		{WireAddr: deadAddr, ControlAddr: "ctrl-0"},
		{WireAddr: deadAddr, ControlAddr: "ctrl-1"},
	}, fetcher)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	_, err = router.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt})
	if !errors.Is(err, ErrPDHandoffFailed) {
		t.Fatalf("router Ship error = %v, want errors.Is(_, ErrPDHandoffFailed)", err)
	}
	var failure *PDHandoffFailure
	if !errors.As(err, &failure) || len(failure.Attempts) != 2 {
		t.Fatalf("failure = %#v, want *PDHandoffFailure with two attempts", err)
	}
	if failure.Attempts[0].PeerIndex != 0 || failure.Attempts[1].PeerIndex != 1 {
		t.Fatalf("attempts = %+v, want peers 0 then 1", failure.Attempts)
	}
	ledger := router.RouteLedger()
	if len(ledger) != 1 || ledger[0].ServedBy != -1 {
		t.Fatalf("failed handoff receipt = %+v, want ServedBy -1", ledger)
	}
	if metrics := router.Metrics(); !strings.Contains(metrics, "fak_pd_router_failures_total 1") {
		t.Fatalf("router metrics missing failure counter:\n%s", metrics)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDPeerRouterScrapeFailureMarksCountUnknown(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	liveAddr, cleanupLive := pdRouterTestServer(t, m)
	defer cleanupLive()

	// Peer 0's control plane is unreachable: its count stays unknown (-1) and
	// sorts after the readable peer 1, which therefore serves despite carrying
	// more lanes.
	fetcher := func(_ context.Context, controlAddr string) (int, error) {
		if controlAddr == "ctrl-0" {
			return 0, errors.New("scrape refused")
		}
		return 4, nil
	}
	router := NewPDPeerRouter(prefillCluster, []PDDecodePeer{
		{WireAddr: liveAddr, ControlAddr: "ctrl-0"},
		{WireAddr: liveAddr, ControlAddr: "ctrl-1"},
	}, fetcher)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	if _, err := router.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}); err != nil {
		t.Fatalf("router Ship: %v", err)
	}
	ledger := router.RouteLedger()
	if len(ledger) != 1 {
		t.Fatalf("route ledger has %d receipts, want 1", len(ledger))
	}
	rec := ledger[0]
	if !reflect.DeepEqual(rec.LaneCounts, []int{-1, 4}) {
		t.Fatalf("receipt lane counts = %v, want [-1 4] (scrape failure is unknown, not zero)", rec.LaneCounts)
	}
	if rec.Chosen != 1 || rec.ServedBy != 1 {
		t.Fatalf("receipt = %+v, want the readable peer 1 chosen and serving", rec)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDPeerRouterStreamedHandoffRecordsReceipt(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	liveAddr, cleanupLive := pdRouterTestServer(t, m)
	defer cleanupLive()

	router := NewPDPeerRouter(prefillCluster, []PDDecodePeer{{WireAddr: liveAddr}}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	var streamed []int
	res, err := router.ShipStream(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}, func(tok int) error {
		streamed = append(streamed, tok)
		return nil
	})
	if err != nil {
		t.Fatalf("router ShipStream: %v", err)
	}
	if want := serialGreedyTokens(m, prompt); !reflect.DeepEqual(streamed, want) || !reflect.DeepEqual(res.Tokens, want) {
		t.Fatalf("streamed %v / result %v, want %v", streamed, res.Tokens, want)
	}
	ledger := router.RouteLedger()
	if len(ledger) != 1 || ledger[0].ServedBy != 0 {
		t.Fatalf("streamed handoff receipt = %+v, want one receipt served by peer 0", ledger)
	}
}

// deadLoopbackAddr returns a 127.0.0.1 port that nothing listens on, so dials
// fail fast with connection refused.
func deadLoopbackAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}
