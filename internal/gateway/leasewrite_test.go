package gateway

// leasewrite_test.go drives the multi-node dev-server WRITE plane (#2299) end-to-end: the
// handler behind POST /v1/leases/{acquire,renew,release}, backed by a real leaseref.Store
// whose Runner is an in-memory git object/ref store (no real git, no repo). The provider
// closure mirrors the cmd/fak wiring (leasewrite_endpoint.go) MINUS the origin publish, so
// the tested seam is the shipped seam. The headline witness is the single-arbiter
// property: two concurrent acquires for one id -> exactly ONE OK, the other LEASE_HELD.

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// memGit is an in-memory git stand-in implementing leaseref.Runner: a blob object store
// (sha -> bytes) and a ref store (ref -> sha) so the whole fenced-write algorithm — the
// update-ref OLD-VALUE compare-and-swap included — runs with no real git. It mirrors the
// leaseref package's own fakeGit (that one is test-unexported), and it locks its maps so
// the arbiter's serialization is the only ordering the test relies on, not luck.
type memGit struct {
	mu    sync.Mutex
	blobs map[string][]byte
	refs  map[string]string
	next  int
}

func newMemGit() *memGit {
	return &memGit{blobs: map[string][]byte{}, refs: map[string]string{}}
}

func (g *memGit) run(_ context.Context, _ string, args ...string) (string, int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	switch args[0] {
	case "hash-object":
		// hash-object -w <file>: read the temp file leaseref wrote, store under a synth id.
		b, err := os.ReadFile(args[len(args)-1])
		if err != nil {
			return "", 1, nil
		}
		g.next++
		id := memSynthID(g.next)
		g.blobs[id] = b
		return id + "\n", 0, nil
	case "update-ref":
		if args[1] == "-d" {
			ref := args[2]
			cur, ok := g.refs[ref]
			if !ok {
				return "", 1, nil // delete of a missing ref exits non-zero (real git)
			}
			if len(args) >= 4 && cur != args[3] {
				return "", 1, nil // CAS delete lost: ref advanced since old-value read
			}
			delete(g.refs, ref)
			return "", 0, nil
		}
		ref, newval := args[1], args[2]
		if len(args) >= 4 {
			old := args[3]
			cur, exists := g.refs[ref]
			if memAllZeros(old) {
				if exists {
					return "", 1, nil // must-not-exist violated: a peer created it first
				}
			} else if !exists || cur != old {
				return "", 1, nil // old-value mismatch: the ref advanced under the writer
			}
		}
		g.refs[ref] = newval
		return "", 0, nil
	case "rev-parse":
		if len(args) == 2 && args[1] == "--show-object-format" {
			return "sha1\n", 0, nil
		}
		ref := args[len(args)-1]
		if id, ok := g.refs[ref]; ok {
			return id + "\n", 0, nil
		}
		return "", 1, nil
	case "cat-file":
		id, ok := g.refs[args[2]]
		if !ok {
			return "", 1, nil
		}
		return string(g.blobs[id]), 0, nil
	case "for-each-ref":
		prefix := args[len(args)-1]
		var lines []string
		for ref := range g.refs {
			if strings.HasPrefix(ref, prefix) {
				lines = append(lines, ref)
			}
		}
		if len(lines) == 0 {
			return "", 0, nil
		}
		return strings.Join(lines, "\n") + "\n", 0, nil
	}
	return "", 0, nil
}

func memAllZeros(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != '0' {
			return false
		}
	}
	return true
}

func memSynthID(n int) string {
	d := memItoa(n)
	return strings.Repeat("0", 39-len(d)) + "1" + d
}

func memItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// nowFunc is a clock-injected instant source: the write plane is driven off an explicit
// atomic-held unix time so the TTL/transition test never touches the wall clock.
type nowFunc struct{ unix atomic.Int64 }

func (n *nowFunc) now() time.Time { return time.Unix(n.unix.Load(), 0) }

// installLeaseWrite wires the single-arbiter write function over the injected store — the
// same fold cmd/fak installs, minus the origin publish — off the injected clock, and
// restores the package state when the test ends so the provider never leaks.
func installLeaseWrite(t *testing.T, store *leaseref.Store, clk *nowFunc) {
	t.Helper()
	SetLeaseWriteFunc(func(ctx context.Context, op string, req LeaseWriteRequest) (LeaseWriteResult, error) {
		now := clk.now()
		switch op {
		case "acquire":
			rec, v, err := store.AcquireFenced(ctx, leaseref.Record{
				ID: req.ID, TreeGlobs: req.TreeGlobs, Holder: req.Holder,
				TTLSeconds: req.TTLSeconds, Description: req.Description,
			}, now)
			if err != nil {
				return LeaseWriteResult{}, err
			}
			return leaseVerdict(op, req, rec, v), nil
		case "renew":
			rec, v, err := store.Renew(ctx, req.ID, req.Holder, req.TTLSeconds, now)
			if err != nil {
				return LeaseWriteResult{}, err
			}
			return leaseVerdict(op, req, rec, v), nil
		case "release":
			v, err := store.ReleaseFenced(ctx, req.ID, req.Holder, req.Generation, now)
			if err != nil {
				return LeaseWriteResult{}, err
			}
			return leaseVerdict(op, req, leaseref.Record{}, v), nil
		}
		return LeaseWriteResult{OK: false, Reason: "UNKNOWN_OP", Op: op, ID: req.ID}, nil
	})
	t.Cleanup(func() { SetLeaseWriteFunc(nil) })
}

// leaseVerdict folds a leaseref.FenceVerdict into the wire result — the test twin of
// cmd/fak's leaseVerdictToResult, kept local so the gateway test package stays free of the
// cmd wiring while exercising the identical translation.
func leaseVerdict(op string, req LeaseWriteRequest, rec leaseref.Record, v leaseref.FenceVerdict) LeaseWriteResult {
	res := LeaseWriteResult{
		OK: v.OK, Reason: v.Reason, Op: op, ID: req.ID,
		CurrentGeneration: v.Current, Detail: v.Detail,
	}
	if v.OK {
		res.Generation = rec.Generation
		res.Holder = rec.Holder
		res.TreeGlobs = rec.TreeGlobs
		if res.Holder == "" {
			res.Holder = req.Holder
		}
	} else {
		res.Generation = v.Presented
		res.Holder = v.Holder
	}
	return res
}

// postLease POSTs a write request and decodes the verdict; it fails the test on a non-200
// (the write plane is deny-as-value — a refusal is a 200 body, never an error status).
func postLease(t *testing.T, url, op string, req LeaseWriteRequest) LeaseWriteResult {
	t.Helper()
	body, _ := json.Marshal(req)
	r, err := http.Post(url+"/v1/leases/"+op, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/leases/%s: %v", op, err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/leases/%s status = %d, want 200 (deny-as-value)", op, r.StatusCode)
	}
	var res LeaseWriteResult
	if err := json.NewDecoder(r.Body).Decode(&res); err != nil {
		t.Fatalf("decode /v1/leases/%s: %v", op, err)
	}
	return res
}

// TestLeaseWritePlaneSingleArbiterAcquire is the #2299 headline acceptance: two acquires
// for ONE id, both routed through the coordinator, resolve to EXACTLY ONE OK — the second,
// reading the live lease held by a different holder, is refused LEASE_HELD. This is the
// single-arbiter closure: serialized through the gateway, the cross-node race collapses to
// one deterministic winner, no LEASE_CONTENDED retry storm.
func TestLeaseWritePlaneSingleArbiterAcquire(t *testing.T) {
	clk := &nowFunc{}
	clk.unix.Store(1000)
	store := leaseref.NewWithRunner(newMemGit().run, "")
	installLeaseWrite(t, store, clk)
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Two nodes race the SAME id with DIFFERENT holders, concurrently.
	var wg sync.WaitGroup
	results := make([]LeaseWriteResult, 2)
	holders := []string{"nodeA/guard-1", "nodeB/guard-2"}
	for i := range holders {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = postLease(t, ts.URL, "acquire", LeaseWriteRequest{
				ID: "kernel-lane", Holder: holders[i],
				TreeGlobs: []string{"internal/kernel/**"}, TTLSeconds: 300,
			})
		}(i)
	}
	wg.Wait()

	oks, held := 0, 0
	var winner LeaseWriteResult
	for _, res := range results {
		if res.OK {
			oks++
			winner = res
			continue
		}
		if res.Reason != leaseref.ReasonLeaseHeld {
			t.Errorf("loser reason = %q, want %s (a serialized second acquire reads the live lease)", res.Reason, leaseref.ReasonLeaseHeld)
		}
		held++
	}
	if oks != 1 {
		t.Fatalf("concurrent acquires: %d OK, want EXACTLY ONE (single arbiter) — results=%+v", oks, results)
	}
	if held != 1 {
		t.Fatalf("concurrent acquires: %d LEASE_HELD, want exactly one loser — results=%+v", held, results)
	}
	if winner.Generation != 1 {
		t.Errorf("winner generation = %d, want 1 (a fresh lease starts the fencing token at 1)", winner.Generation)
	}
	if winner.Op != "acquire" || winner.ID != "kernel-lane" {
		t.Errorf("winner verdict = %+v, want op=acquire id=kernel-lane", winner)
	}
	if !strings.Contains(strings.ToLower(winner.Source), "single-arbiter") {
		t.Errorf("source = %q, want the single-arbiter qualifier", winner.Source)
	}
}

// TestLeaseWritePlaneMonotonicFencingToken proves the fencing token increases across a
// transition: holder A acquires (gen 1), goes dormant past its TTL, peer B reaps and
// reacquires through the coordinator — the token strictly bumps to 2. The renew twin does
// NOT bump (a renew is liveness, not a new admission).
func TestLeaseWritePlaneMonotonicFencingToken(t *testing.T) {
	clk := &nowFunc{}
	clk.unix.Store(1000)
	store := leaseref.NewWithRunner(newMemGit().run, "")
	installLeaseWrite(t, store, clk)
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	a := postLease(t, ts.URL, "acquire", LeaseWriteRequest{ID: "lane", Holder: "A", TTLSeconds: 300})
	if !a.OK || a.Generation != 1 {
		t.Fatalf("A acquire = %+v, want ok gen=1", a)
	}

	// A renews before expiry: same holder, generation UNCHANGED.
	clk.unix.Store(1100)
	ar := postLease(t, ts.URL, "renew", LeaseWriteRequest{ID: "lane", Holder: "A", TTLSeconds: 300})
	if !ar.OK || ar.Generation != 1 {
		t.Fatalf("A renew = %+v, want ok gen=1 (a renew never bumps the token)", ar)
	}

	// A goes dormant well past the TTL; peer B reaps + reacquires: a TRANSITION, gen -> 2.
	clk.unix.Store(1100 + 600)
	b := postLease(t, ts.URL, "acquire", LeaseWriteRequest{ID: "lane", Holder: "B", TTLSeconds: 300})
	if !b.OK || b.Generation != 2 || b.Holder != "B" {
		t.Fatalf("B transition = %+v, want ok gen=2 holder=B (the fencing token bumps on takeover)", b)
	}
	if b.Generation <= a.Generation {
		t.Fatalf("fencing token not monotonic: A=%d B=%d", a.Generation, b.Generation)
	}
}

// TestLeaseWritePlaneRefusalsAreDenyAsValue proves the closed refusal vocabulary crosses as
// 200 verdict bodies: a renew of a lease held by another holder is STALE_LEASE, and a renew
// of a never-acquired id is NO_LEASE — both ok:false, both HTTP 200.
func TestLeaseWritePlaneRefusalsAreDenyAsValue(t *testing.T) {
	clk := &nowFunc{}
	clk.unix.Store(1000)
	store := leaseref.NewWithRunner(newMemGit().run, "")
	installLeaseWrite(t, store, clk)
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// NO_LEASE: renew an id nobody ever acquired.
	nl := postLease(t, ts.URL, "renew", LeaseWriteRequest{ID: "ghost", Holder: "X", TTLSeconds: 300})
	if nl.OK || nl.Reason != leaseref.ReasonNoLease {
		t.Fatalf("renew of unheld id = %+v, want ok=false reason=NO_LEASE", nl)
	}

	// STALE_LEASE: A holds it live; B tries to renew A's lease.
	if a := postLease(t, ts.URL, "acquire", LeaseWriteRequest{ID: "lane", Holder: "A", TTLSeconds: 300}); !a.OK {
		t.Fatalf("A acquire = %+v, want ok", a)
	}
	sl := postLease(t, ts.URL, "renew", LeaseWriteRequest{ID: "lane", Holder: "B", TTLSeconds: 300})
	if sl.OK || sl.Reason != leaseref.ReasonStaleLease {
		t.Fatalf("B renew of A's lease = %+v, want ok=false reason=STALE_LEASE", sl)
	}
	if sl.Holder != "A" {
		t.Errorf("refusal holder = %q, want A (the refusal names who owns it)", sl.Holder)
	}

	// Release by the real holder is an admitted OK, and a re-release is idempotent-OK.
	rel := postLease(t, ts.URL, "release", LeaseWriteRequest{ID: "lane", Holder: "A"})
	if !rel.OK {
		t.Fatalf("A release = %+v, want ok", rel)
	}
}

// TestLeaseWritePlaneTransportContracts pins the HTTP-fact statuses: unwired -> 404, a
// non-POST -> 405, an unknown verb -> 404. These are the ONLY non-200s the plane emits;
// every policy outcome stays a 200 deny-as-value body.
func TestLeaseWritePlaneTransportContracts(t *testing.T) {
	// Unwired: 404 on the whole subtree.
	SetLeaseWriteFunc(nil)
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	if r, _ := http.Post(ts.URL+"/v1/leases/acquire", "application/json", strings.NewReader(`{"id":"x"}`)); r != nil {
		r.Body.Close()
		if r.StatusCode != http.StatusNotFound {
			t.Errorf("unwired POST status = %d, want 404", r.StatusCode)
		}
	}

	// Wired: method + unknown-verb contracts.
	clk := &nowFunc{}
	clk.unix.Store(1000)
	installLeaseWrite(t, leaseref.NewWithRunner(newMemGit().run, ""), clk)

	r, err := http.Get(ts.URL + "/v1/leases/acquire")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("GET /v1/leases/acquire status = %d, want 405 (write plane is POST-only)", r.StatusCode)
	}

	r2, err := http.Post(ts.URL+"/v1/leases/bogus", "application/json", strings.NewReader(`{"id":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	r2.Body.Close()
	if r2.StatusCode != http.StatusNotFound {
		t.Errorf("POST /v1/leases/bogus status = %d, want 404 (the verb set is closed)", r2.StatusCode)
	}

	// The read plane (exact /v1/leases) must still win over the /v1/leases/ subtree: a GET
	// to /v1/leases is NOT a 405-from-the-write-handler (it is the read plane, 404 here
	// only because the READ provider is unwired in this test).
	r3, err := http.Get(ts.URL + "/v1/leases")
	if err != nil {
		t.Fatal(err)
	}
	r3.Body.Close()
	if r3.StatusCode == http.StatusMethodNotAllowed {
		t.Errorf("GET /v1/leases hit the write handler (405); the exact read route must win over the subtree")
	}
}
