package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// fakeLedgerReader is a hermetic relay.LedgerReader for the verified-path test — it
// returns a fixed row set for the anchor it knows and an error for any other, mirroring
// the injected-reader discipline internal/relay/progress_test.go uses.
type fakeLedgerReader struct {
	anchor string
	steps  []relay.ProgressStep
}

func (f fakeLedgerReader) ReadProgress(ledgerRef string) ([]relay.ProgressStep, error) {
	if ledgerRef == f.anchor {
		return f.steps, nil
	}
	return nil, errUnknownAnchor
}

var errUnknownAnchor = &ledgerErr{"unknown anchor"}

type ledgerErr struct{ s string }

func (e *ledgerErr) Error() string { return e.s }

// TestA2AVerifiedProgressFailsClosed pins the three unbound/unreadable edges: each must
// yield verdict "unknown" with no steps, so the A2A read never asserts unverifiable
// progress to a foreign peer.
func TestA2AVerifiedProgressFailsClosed(t *testing.T) {
	cases := []struct {
		name string
		task *a2aTask
		lr   relay.LedgerReader
	}{
		{"nil task", nil, nil},
		{"no anchor, nil reader", &a2aTask{TaskID: "t1", State: "running"}, nil},
		{"anchor set, no reader wired", &a2aTask{TaskID: "t2", State: "running", LedgerRef: "run:abc"}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := a2aVerifiedProgress(c.task, c.lr)
			if got.Verdict != relay.ProgressUnknown {
				t.Fatalf("want verdict %q, got %q", relay.ProgressUnknown, got.Verdict)
			}
			if len(got.Steps) != 0 {
				t.Fatalf("fail-closed progress must carry no steps, got %d", len(got.Steps))
			}
		})
	}
}

// TestA2AVerifiedProgressReadsLedger proves the reuse half: with a bound anchor and a
// wired reader, the projection returns the ledger's OWN rows (verdict verified), never a
// number the task asserted.
func TestA2AVerifiedProgressReadsLedger(t *testing.T) {
	steps := []relay.ProgressStep{
		{Ref: "abc123", Note: "commit"},
		{Ref: "#1879", Note: "issue"},
	}
	task := &a2aTask{TaskID: "t3", State: "running", LedgerRef: "run:xyz"}
	got := a2aVerifiedProgress(task, fakeLedgerReader{anchor: "run:xyz", steps: steps})
	if got.Verdict != relay.ProgressVerified {
		t.Fatalf("want verdict %q, got %q (%s)", relay.ProgressVerified, got.Verdict, got.Reason)
	}
	if len(got.Steps) != len(steps) {
		t.Fatalf("want %d steps read from the ledger, got %d", len(steps), len(got.Steps))
	}
	if got.Steps[0].Ref != "abc123" {
		t.Fatalf("steps must be the ledger's own rows, got %+v", got.Steps)
	}
}

// TestA2AGetTaskCarriesVerifiedProgress is the edge witness: a peer's GET on an in-flight
// task receives a `progress` object shaped like relay.VerifiedProgress (a re-verifiable
// cursor), and that object carries NO self-report key — the no-`claimed`-field invariant
// held across the A2A HTTP edge, not just in-kernel. relay.TestVerifiedProgressHasNoClaimedField
// pins the deep structural guarantee; this asserts the shape a foreign agent actually reads.
func TestA2AGetTaskCarriesVerifiedProgress(t *testing.T) {
	srv, err := New(Config{EngineID: "mock", Model: "test-model"})
	if err != nil {
		t.Fatalf("failed to create gateway: %v", err)
	}
	defer srv.Close()

	const taskID = "test-inflight-progress-1"
	store := getA2AStore()
	store.mu.Lock()
	store.tasks[taskID] = &a2aTask{
		TaskID:    taskID,
		State:     "running", // in-flight: the state a peer must NOT trust as a claim
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	store.mu.Unlock()
	defer func() {
		store.mu.Lock()
		delete(store.tasks, taskID)
		store.mu.Unlock()
	}()

	req := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID, nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	var body map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	progRaw, ok := body["progress"]
	if !ok {
		t.Fatalf("GetTask response must carry a `progress` cursor for in-flight state")
	}
	prog, ok := progRaw.(map[string]interface{})
	if !ok {
		t.Fatalf("`progress` must be an object, got %T", progRaw)
	}
	// Fail-closed: no run anchor bound -> the peer is told "unknown", never "verified".
	if prog["verdict"] != string(relay.ProgressUnknown) {
		t.Fatalf("unanchored in-flight task must report verdict %q, got %v", relay.ProgressUnknown, prog["verdict"])
	}
	// The load-bearing invariant at the edge: no self-report key may appear as progress.
	for _, banned := range []string{"claimed", "success", "done", "percent", "progress_pct", "completed"} {
		if _, bad := prog[banned]; bad {
			t.Fatalf("progress cursor leaked a self-report key %q — the no-claimed-field invariant is broken at the A2A edge", banned)
		}
	}
}
