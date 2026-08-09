package gateway

// session_checkpoint_test.go — the wire half of #2425's witness: POST
// /v1/fak/session/{id}/checkpoint mints the two-axis record, echoes BOTH hashes, and a
// re-check names the axis that moved. The binding itself is proven hermetically in
// internal/sessionledger (TestCheckpointWitnessBinds); this proves the route.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// postCheckpoint drives the checkpoint verb against the given ledger and returns the
// recorder, with teleportLedger (the control-plane verbs' ledger seam) pointed at it.
func postCheckpoint(t *testing.T, l *sessionledger.Ledger, trace string, body CheckpointRequest) *httptest.ResponseRecorder {
	t.Helper()
	prev := teleportLedger
	teleportLedger = func() (*sessionledger.Ledger, error) { return l, nil }
	defer func() { teleportLedger = prev }()

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	srv := &Server{logf: func(string, ...any) {}}
	rr := httptest.NewRecorder()
	srv.handleFakSession(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/session/"+trace+"/checkpoint", bytes.NewReader(raw)))
	return rr
}

func decodeCheckpointResp(t *testing.T, rr *httptest.ResponseRecorder) CheckpointResponse {
	t.Helper()
	var got CheckpointResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response %s: %v", rr.Body.String(), err)
	}
	return got
}

// TestSessionCheckpointRouteBindsBothAxes is the route's acceptance: mint, then verify
// clean, then verify against a mutated tree — which must be refused NAMING the tree axis
// while the transcript half is untouched.
func TestSessionCheckpointRouteBindsBothAxes(t *testing.T) {
	l, trace := seedHostA(t)
	head := l.Head(trace)
	witness := CheckpointRequest{
		HeadSHA: "9a8b7c6d5e4f30211122334455667788990aabbc",
		Dirty: []sessionledger.DirtyEntry{
			{Path: "internal/gateway/http.go", Status: " M", SHA256: "11aa"},
		},
	}

	rr := postCheckpoint(t, l, trace, witness)
	if rr.Code != http.StatusOK {
		t.Fatalf("mint status = %d, want 200 (body %s)", rr.Code, rr.Body.String())
	}
	got := decodeCheckpointResp(t, rr)
	if got.TraceID != trace {
		t.Fatalf("trace_id = %q, want %q", got.TraceID, trace)
	}
	// Both hashes, in one record: the transcript head it was minted over and the tree
	// witness it binds to.
	if got.Checkpoint.LedgerHead != head {
		t.Fatalf("ledger_head = %q, want the pre-checkpoint head %q", got.Checkpoint.LedgerHead, head)
	}
	if got.Checkpoint.Tree.HeadSHA != witness.HeadSHA {
		t.Fatalf("tree head_sha = %q, want %q", got.Checkpoint.Tree.HeadSHA, witness.HeadSHA)
	}
	if got.Checkpoint.Hash == "" || got.Checkpoint.Hash == got.Checkpoint.LedgerHead {
		t.Fatalf("checkpoint id %q must be the record hash binding both axes", got.Checkpoint.Hash)
	}
	if got.Checkpoint.Hash != l.Head(trace) {
		t.Fatalf("the checkpoint should be the trace's new head: %s vs %s", got.Checkpoint.Hash, l.Head(trace))
	}
	if got.Verified != nil {
		t.Fatalf("a mint carries no verdict, got %+v", got.Verified)
	}

	// A peer re-checks the claim with the same witness: it holds.
	check := witness
	check.Verify = true
	rr = postCheckpoint(t, l, trace, check)
	if rr.Code != http.StatusOK {
		t.Fatalf("verify status = %d, want 200 (body %s)", rr.Code, rr.Body.String())
	}
	ok := decodeCheckpointResp(t, rr)
	if ok.Verified == nil || !*ok.Verified {
		t.Fatalf("clean verify should report verified=true: %+v", ok)
	}
	if ok.Checkpoint.Hash != got.Checkpoint.Hash {
		t.Fatalf("verify re-derived a different checkpoint: %s vs %s", ok.Checkpoint.Hash, got.Checkpoint.Hash)
	}

	// One tracked file's bytes move: the claim no longer holds, and the failure names
	// the TREE axis (409 — the request is fine, the world conflicts with the claim).
	drifted := CheckpointRequest{
		HeadSHA: witness.HeadSHA,
		Dirty:   []sessionledger.DirtyEntry{{Path: "internal/gateway/http.go", Status: " M", SHA256: "11aa-EDITED"}},
		Verify:  true,
	}
	rr = postCheckpoint(t, l, trace, drifted)
	if rr.Code != http.StatusConflict {
		t.Fatalf("drifted verify status = %d, want 409 (body %s)", rr.Code, rr.Body.String())
	}
	bad := decodeCheckpointResp(t, rr)
	if bad.Verified == nil || *bad.Verified {
		t.Fatalf("drifted verify should report verified=false: %+v", bad)
	}
	if bad.Axis != sessionledger.AxisTree {
		t.Fatalf("drifted verify named the %q axis, want %q (detail %s)", bad.Axis, sessionledger.AxisTree, bad.Detail)
	}
}

// TestSessionCheckpointRouteFailsClosed pins the refusals: no tree witness is a 400 (a
// checkpoint with only one axis is not a checkpoint), and a verify against a trace that
// never minted one is a 404 rather than an invented pass.
func TestSessionCheckpointRouteFailsClosed(t *testing.T) {
	l, trace := seedHostA(t)

	rr := postCheckpoint(t, l, trace, CheckpointRequest{})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("a body with no head_sha should be 400, got %d (%s)", rr.Code, rr.Body.String())
	}
	if head := l.Head(trace); head == "" {
		t.Fatal("the refused mint must not have disturbed the trace")
	}

	rr = postCheckpoint(t, l, "sess:never-checkpointed", CheckpointRequest{HeadSHA: "abc123", Verify: true})
	if rr.Code != http.StatusNotFound {
		t.Fatalf("verifying a trace with no checkpoint should be 404, got %d (%s)", rr.Code, rr.Body.String())
	}
}
