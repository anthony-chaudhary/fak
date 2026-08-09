package main

// Witness for #5568: `fak session ls --fleet` must GIVE UP on a remote that never
// answers, print a degradation line that NAMES the budget it spent, and still list the
// C2 session refs this clone already has.
//
// The stall is simulated deterministically through internal/leaseref's injected-Runner
// seam (leaseref.NewWithRunner) — the same shape
// TestAmbientLeaseRefSyncGivesUpOnAStalledGitRunner uses for #5564. NO REMOTE IS
// CONTACTED: only the transport verb blocks, and it blocks on <-ctx.Done(), so it
// returns if and only if the call site really put a deadline on the fetch.

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestSessionLsFleetGivesUpOnAStalledFetch fails in three distinguishable ways on a
// regression, all of them loud. If the fetch goes back to an unexpirable context the
// runner's <-ctx.Done() never fires and the select below fails in bounded time with the
// diagnosis, instead of wedging the package suite until the go-test timeout. If the
// deadline arrives but the degradation line does not name it, the operator would blame a
// remote this file cut off, and the message assertions say so. And if the LOCAL ref read
// is handed the fetch's spent context, the recorded read-context state names it — that is
// the timing-free witness that "showing already-fetched C2 refs only" is a promise the
// command can actually keep.
func TestSessionLsFleetGivesUpOnAStalledFetch(t *testing.T) {
	restore := sessionFleetFetchBudget
	sessionFleetFetchBudget = 200 * time.Millisecond
	t.Cleanup(func() { sessionFleetFetchBudget = restore })

	now := time.Unix(4_000_000, 0)
	// One already-fetched peer descriptor: the C2 ref this clone HAS, which the stalled
	// fetch must not cost it.
	blob, err := json.Marshal(leaseref.SessionDescriptor{
		ID: "peer-1", Host: "node-2", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600,
	})
	if err != nil {
		t.Fatalf("marshal descriptor: %v", err)
	}

	type readCall struct {
		hasDeadline    bool
		expiredOnEntry bool
	}
	var transport []string
	var reads []readCall

	// Only the transport verbs hang. The local ref reads answer normally, so what is
	// witnessed is a wedged NETWORK git — not a store that cannot read its own refs.
	stalled := func(ctx context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "fetch", "push":
			transport = append(transport, verb)
			<-ctx.Done() // returns ONLY when the context does — never, if it cannot expire
			return "", -1, ctx.Err()
		case "for-each-ref":
			_, has := ctx.Deadline()
			reads = append(reads, readCall{hasDeadline: has, expiredOnEntry: ctx.Err() != nil})
			return "refs/fak/locks/session-peer-1\n", 0, nil
		case "cat-file":
			return string(blob), 0, nil
		default:
			// rev-parse --git-common-dir and friends: an empty answer keeps the
			// injected-runner store on its pure-argv path.
			return "", 0, nil
		}
	}
	store := leaseref.NewWithRunner(stalled, t.TempDir())

	var stderr bytes.Buffer
	done := make(chan []leaseref.SessionDescriptor, 1)
	go func() { done <- fleetSessionDescriptors(&stderr, store, "origin", now) }()
	var descs []leaseref.SessionDescriptor
	select {
	case descs = <-done:
	case <-time.After(30 * sessionFleetFetchBudget):
		t.Fatal("fak session ls --fleet never returned against a stalled git: the fetch context cannot expire, so an unreachable-but-not-refusing remote hangs the command (#5568)")
	}

	// The give-up is SURFACED, and it names the budget rather than leaving a bare git
	// exit code that reads as the remote's fault.
	out := stderr.String()
	if !strings.Contains(out, "showing already-fetched C2 refs only") {
		t.Errorf("stderr did not degrade to the already-fetched view: %q", out)
	}
	if !strings.Contains(out, "fetch budget") || !strings.Contains(out, sessionFleetFetchBudget.String()) {
		t.Errorf("degradation line %q does not NAME the %s timeout — the operator would blame the remote for a deadline this command set", out, sessionFleetFetchBudget)
	}

	// ...and the C2 refs this clone already has are still listed. A fleet listing that
	// gave up on the fetch AND dropped the local peer rows would have degraded twice.
	if len(descs) != 1 || descs[0].ID != "peer-1" {
		t.Fatalf("fleet descriptors = %+v, want the already-fetched peer-1 row", descs)
	}

	// Fetch-only: --fleet reads the fleet, it never publishes this node's leases.
	if len(transport) != 1 || transport[0] != "fetch" {
		t.Fatalf("transport calls = %v, want exactly one fetch", transport)
	}

	// The local read got its OWN live deadline, not a share of the spent fetch one.
	if len(reads) == 0 {
		t.Fatal("the local C2 ref read never ran after the stalled fetch")
	}
	if !reads[0].hasDeadline {
		t.Error("the local C2 ref read carried NO deadline: a wedged local git still hangs the command")
	}
	if reads[0].expiredOnEntry {
		t.Error("the local C2 ref read inherited the fetch's SPENT context: the command promises \"already-fetched C2 refs only\" and then cannot read a single one (#5568)")
	}
}

// TestSessionLsFleetHealthyFetchStaysQuiet pins the other side of the bound: when the
// remote answers, the budget is invisible — no degradation line, and the peer rows come
// back. A fix that reported a give-up on every run would pass the test above and still be
// wrong.
func TestSessionLsFleetHealthyFetchStaysQuiet(t *testing.T) {
	now := time.Unix(5_000_000, 0)
	blob, err := json.Marshal(leaseref.SessionDescriptor{
		ID: "peer-2", Host: "node-3", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600,
	})
	if err != nil {
		t.Fatalf("marshal descriptor: %v", err)
	}
	healthy := func(_ context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "for-each-ref":
			return "refs/fak/locks/session-peer-2\n", 0, nil
		case "cat-file":
			return string(blob), 0, nil
		default:
			return "", 0, nil
		}
	}

	var stderr bytes.Buffer
	descs := fleetSessionDescriptors(&stderr, leaseref.NewWithRunner(healthy, t.TempDir()), "origin", now)
	if got := stderr.String(); got != "" {
		t.Errorf("a healthy fetch must print nothing, got %q", got)
	}
	if len(descs) != 1 || descs[0].ID != "peer-2" {
		t.Fatalf("fleet descriptors = %+v, want peer-2", descs)
	}
}
