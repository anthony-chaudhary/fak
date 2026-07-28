package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// leasewrite_endpoint_test.go covers the two things #5422 asked for: that the accepted-write
// plane-0 publish is HANDED OFF rather than performed inside the gateway's leaseWriteMu
// critical section, and that this file's init() really installs the write hook. The second is
// a regression test for a shipped-but-dead surface: the handler, the route, the gateway-side
// test and the OpenAPI contract all landed in fec8da6f7 with ZERO callers of
// gateway.SetLeaseWriteFunc outside the gateway package, so POST /v1/leases/{verb} 404'd in
// every real binary until f02811ab5 supplied the init(). Deleting that init() is a silent
// un-wiring no other test in the tree notices.

// leaseWriteTestBudget bounds every "did this return without waiting on the publish?" wait.
// It is generous on purpose: the point is to distinguish RETURNED from BLOCKED-FOREVER, not
// to measure latency, and the test must FAIL rather than hang when the publish is inline.
const leaseWriteTestBudget = 10 * time.Second

// TestLeaseWriteEndpointInitInstallsGatewayWriteHook witnesses the wiring end to end, through
// the real gateway mux: POST /v1/leases/acquire with an EMPTY id. The handler checks the
// injected function FIRST (nil => 404 "not configured for this deployment") and only then
// validates the body (empty id => 400). So a 400 is reachable if and only if
// gateway.SetLeaseWriteFunc was actually called — and it never touches the leaseref store,
// so the witness needs no git, no temp repo and no network.
func TestLeaseWriteEndpointInitInstallsGatewayWriteHook(t *testing.T) {
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/leases/acquire", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("POST /v1/leases/acquire: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	got := strings.TrimSpace(string(body))

	if resp.StatusCode == http.StatusNotFound {
		t.Fatalf("POST /v1/leases/acquire = 404 %q — the multi-node WRITE plane is UNWIRED: "+
			"cmd/fak's init() no longer calls gateway.SetLeaseWriteFunc, so every documented "+
			"POST /v1/leases/{acquire,renew,release} refuses as unconfigured (the f02811ab5 regression)", got)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("POST /v1/leases/acquire with an empty id = %d %q, want 400 (the wired handler's own body validation)", resp.StatusCode, got)
	}
	if !strings.Contains(got, "non-empty id") {
		t.Fatalf("400 body = %q, want the wired handler's empty-id refusal", got)
	}
}

// TestLeaseWritePublishRunsOutsideTheArbiterCall is the #5422 throughput witness. The gateway
// holds leaseWriteMu across the WHOLE serveLeaseWrite call, so "the publish is not inside the
// single-arbiter critical section" is exactly "serveLeaseWrite returns before the publish
// finishes". The test blocks the publish boundary and asserts the accepted write returns
// anyway — and that a SECOND lease write is admitted while the first publish is still out on
// the network. With the publish inline both waits expire and this test fails.
func TestLeaseWritePublishRunsOutsideTheArbiterCall(t *testing.T) {
	useLeaseWriteTestRepo(t)

	entered := make(chan struct{}, 8)
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }

	var mu sync.Mutex
	calls, live, maxLive := 0, 0, 0

	prev := leasePublish
	leasePublish = func(ctx context.Context, s *leaseref.Store) {
		mu.Lock()
		calls, live = calls+1, live+1
		if live > maxLive {
			maxLive = live
		}
		mu.Unlock()
		entered <- struct{}{}
		<-release
		mu.Lock()
		live--
		mu.Unlock()
	}
	t.Cleanup(func() {
		unblock()
		<-leasePublishes.idleC()
		leasePublish = prev
	})

	// The first accepted acquire. Its verdict must come back while the publish it caused is
	// still parked inside the (blocked) boundary.
	first := make(chan gateway.LeaseWriteResult, 1)
	go func() {
		res, err := serveLeaseWrite(context.Background(), "acquire", gateway.LeaseWriteRequest{
			ID: "lanea", Holder: "A", TTLSeconds: 300, TreeGlobs: []string{"internal/x/**"},
		})
		if err != nil {
			t.Errorf("serveLeaseWrite(acquire lanea): %v", err)
		}
		first <- res
	}()

	select {
	case res := <-first:
		if !res.OK {
			t.Fatalf("acquire on an empty store was refused: %+v", res)
		}
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("serveLeaseWrite did not return within %s while the plane-0 publish was blocked — "+
			"the publish is still INLINE on the accepted-write path, i.e. still inside the gateway's "+
			"leaseWriteMu critical section (#5422)", leaseWriteTestBudget)
	}

	// The publish really was requested (the hand-off publishes, it does not drop).
	select {
	case <-entered:
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("no plane-0 publish was requested for the accepted acquire within %s", leaseWriteTestBudget)
	}

	// A SECOND lease write, admitted while the first publish is still blocked. This is the
	// property the issue is about: concurrent writes no longer queue behind a network round
	// trip held under the single-arbiter mutex.
	second := make(chan gateway.LeaseWriteResult, 1)
	go func() {
		res, err := serveLeaseWrite(context.Background(), "acquire", gateway.LeaseWriteRequest{
			ID: "laneb", Holder: "B", TTLSeconds: 300,
		})
		if err != nil {
			t.Errorf("serveLeaseWrite(acquire laneb): %v", err)
		}
		second <- res
	}()
	select {
	case res := <-second:
		if !res.OK {
			t.Fatalf("second acquire (a distinct id) was refused: %+v", res)
		}
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("a second lease write did not complete within %s while an earlier publish was in flight — "+
			"the arbiter is still serialized behind the plane-0 push (#5422)", leaseWriteTestBudget)
	}

	unblock()
	select {
	case <-leasePublishes.idleC():
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("the publish queue did not drain within %s after the boundary was released", leaseWriteTestBudget)
	}

	mu.Lock()
	gotCalls, gotMax := calls, maxLive
	mu.Unlock()
	if gotMax != 1 {
		t.Fatalf("max concurrent plane-0 publishes = %d, want 1 — pushes of the same refspec must stay SERIALIZED "+
			"or origin can end up holding an older namespace snapshot than one already published", gotMax)
	}
	if gotCalls != 2 {
		t.Fatalf("plane-0 publishes = %d, want 2 (the in-flight one, then the latched follow-up for the second "+
			"accepted write) — a write accepted during a push must still be published", gotCalls)
	}
}

// TestLeasePublishQueueCoalescesWithoutDroppingAWrite states the ordering guarantee the
// hand-off keeps, with no git and no store: at most ONE publish in flight, and a request that
// lands during a push is LATCHED into exactly one follow-up rather than dropped (the in-flight
// push may have read the ref store before that write committed, so it cannot be assumed to
// cover it) or fanned out into one push per write.
func TestLeasePublishQueueCoalescesWithoutDroppingAWrite(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{}, 8)
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }

	var mu sync.Mutex
	calls, live, maxLive := 0, 0, 0

	prev := leasePublish
	leasePublish = func(ctx context.Context, s *leaseref.Store) {
		mu.Lock()
		calls, live = calls+1, live+1
		if live > maxLive {
			maxLive = live
		}
		mu.Unlock()
		entered <- struct{}{}
		<-release
		mu.Lock()
		live--
		mu.Unlock()
	}
	var q leasePublishQueue
	t.Cleanup(func() {
		unblock()
		<-q.idleC()
		leasePublish = prev
	})

	store := leaseref.NewInDir(t.TempDir())
	q.request(store)
	select {
	case <-entered:
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("the first request did not start a publish within %s", leaseWriteTestBudget)
	}
	// Three more accepted writes while that push is out on the network. They must coalesce
	// into exactly one follow-up push, not three.
	q.request(store)
	q.request(store)
	q.request(store)

	unblock()
	select {
	case <-q.idleC():
	case <-time.After(leaseWriteTestBudget):
		t.Fatalf("the queue did not drain within %s", leaseWriteTestBudget)
	}

	mu.Lock()
	gotCalls, gotMax := calls, maxLive
	mu.Unlock()
	if gotMax != 1 {
		t.Fatalf("max concurrent publishes = %d, want 1 (serialized)", gotMax)
	}
	if gotCalls != 2 {
		t.Fatalf("publishes = %d, want 2 (the in-flight push plus ONE coalesced follow-up covering all three "+
			"writes that landed during it)", gotCalls)
	}
	// A drained queue answers idle immediately, so a later caller never blocks on no work.
	select {
	case <-q.idleC():
	default:
		t.Fatal("idleC on a drained queue did not answer immediately")
	}
}

// useLeaseWriteTestRepo points leasePlaneDir() at a throwaway git repo so serveLeaseWrite's
// fenced CAS is real (hash-object + update-ref under refs/fak/locks/*). No commit is ever
// made, so no identity config is needed, and no remote exists — the publish boundary is
// replaced by the caller, so nothing here ever reaches the network.
func useLeaseWriteTestRepo(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	c := exec.Command("git", "init", "-q")
	c.Dir = dir
	if out, err := c.CombinedOutput(); err != nil {
		t.Fatalf("git init %s: %v\n%s", dir, err, out)
	}
	t.Setenv("FAK_LEASEPLANE_DIR", dir)
}
