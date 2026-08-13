package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"testing"
)

func placement(provider, account, model, compute string, caps ...string) SessionPlacement {
	return SessionPlacement{Provider: provider, AccountRef: account, Model: model, Compute: compute, Capabilities: caps, ContextLimit: 128000, BudgetAvailable: 100, ComputeAvailable: true, CacheLineage: provider + ":cache"}
}

func TestMoveSessionPreservesLogicalIdentityChangesEpochAndFencesOldClient(t *testing.T) {
	state := SessionState{TraceID: "portable-1", Run: "IDLE", Rev: 9}
	s := &Server{observeSession: func(context.Context, string) SessionState { return state }, sessionFeed: newSessionFeed(32)}
	s.PublishSessionRevision(state)
	s.RecordSessionTerminalOutput(state.TraceID, []byte("before move\r\n"))
	source := placement("provider-a", "account-ref-a", "model-a", "compute-a", "tools", "vision")
	destination := placement("provider-b", "account-ref-b", "model-b", "compute-b", "tools")
	destination.SemanticDegradations = []string{"vision unavailable"}
	var restored, committed bool
	var transitions []SessionMoveTransition
	err := s.ConfigureSessionMove(state.TraceID, source, SessionMoveHooks{
		RequestSafePoint: func(context.Context, string) error { return nil },
		AdmitDestination: func(_ context.Context, _ string, cp SessionMoveCheckpoint, req SessionMoveRequest) error {
			if strings.Contains(string(mustJSON(t, cp)), "credential") {
				t.Fatal("credential leaked into checkpoint")
			}
			return nil
		},
		RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { restored = true; return nil },
		CommitDestination:  func(context.Context, SessionMoveCheckpoint) error { committed = true; return nil },
		RecordTransition: func(_ context.Context, tr SessionMoveTransition) error {
			transitions = append(transitions, tr)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	oldEpoch := rt.sessionLocked(state.TraceID).executionEpoch
	rt.attachments["old"] = sessionClientAttachment{SessionID: state.TraceID}
	rt.leases[state.TraceID] = "old"
	rt.mu.Unlock()
	resp, err := s.MoveSession(context.Background(), state.TraceID, SessionMoveRequest{ExecutionEpoch: oldEpoch, Destination: destination, RequiredCaps: []string{"tools"}, RequiredContext: 64000, RequiredBudget: 10})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Descriptor.SessionID != state.TraceID || resp.Descriptor.ExecutionEpoch == oldEpoch {
		t.Fatalf("identity/epoch not moved: %+v", resp.Descriptor)
	}
	want := []SessionMovePhase{SessionMoveSafePointRequested, SessionMoveCheckpointed, SessionMoveDestinationAdmitted, SessionMoveRestored, SessionMoveCutoverCommitted}
	if len(transitions) != len(want) {
		t.Fatalf("transitions=%+v", transitions)
	}
	for i, phase := range want {
		if transitions[i].Phase != phase {
			t.Fatalf("transition %d=%s", i, transitions[i].Phase)
		}
	}
	if !restored || !committed {
		t.Fatalf("restore=%v commit=%v", restored, committed)
	}
	t.Logf("MOVE PASS session=%s old_epoch=%s new_epoch=%s source=%s/%s/%s/%s destination=%s/%s/%s/%s phases=%v removed=%v degradation=%v", state.TraceID, oldEpoch, resp.Descriptor.ExecutionEpoch, source.Provider, source.AccountRef, source.Model, source.Compute, destination.Provider, destination.AccountRef, destination.Model, destination.Compute, want, resp.Delta.CapabilityRemoved, resp.Delta.SemanticDegradations)
	if len(resp.Delta.CapabilityRemoved) != 1 || resp.Delta.CapabilityRemoved[0] != "vision" || !resp.Delta.CacheLineageChanged || len(resp.Delta.SemanticDegradations) != 1 {
		t.Fatalf("delta=%+v", resp.Delta)
	}
	rt.mu.Lock()
	_, oldAttached := rt.attachments["old"]
	_, oldLease := rt.leases[state.TraceID]
	rt.mu.Unlock()
	if oldAttached || oldLease {
		t.Fatal("old epoch retained attachment authority")
	}
	_, err = s.MoveSession(context.Background(), state.TraceID, SessionMoveRequest{ExecutionEpoch: oldEpoch, Destination: source})
	var me *SessionMoveError
	if !errors.As(err, &me) || me.Code != "STALE_EPOCH" {
		t.Fatalf("old epoch err=%v", err)
	}
}

func TestMoveSessionUnsafeAndPreCommitFailureKeepSourceAuthoritative(t *testing.T) {
	state := SessionState{TraceID: "portable-rollback", Run: "RUNNING", Rev: 1}
	s := &Server{observeSession: func(context.Context, string) SessionState { return state }, sessionFeed: newSessionFeed(8)}
	source, destination := placement("a", "ar-a", "m-a", "c-a"), placement("b", "ar-b", "m-b", "c-b")
	if err := s.ConfigureSessionMove(state.TraceID, source, SessionMoveHooks{AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil }, RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return nil }}); err != nil {
		t.Fatal(err)
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	epoch := rt.sessionLocked(state.TraceID).executionEpoch
	rt.mu.Unlock()
	_, err := s.MoveSession(context.Background(), state.TraceID, SessionMoveRequest{ExecutionEpoch: epoch, Destination: destination})
	var me *SessionMoveError
	if !errors.As(err, &me) || me.Code != "MOVE_UNSAFE" {
		t.Fatalf("unsafe err=%v", err)
	}
	state.Run = "IDLE"
	rolled := false
	if err := s.ConfigureSessionMove(state.TraceID, source, SessionMoveHooks{RequestSafePoint: func(context.Context, string) error { return nil }, AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil }, RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return errors.New("destination crashed") }, RollbackDestination: func(context.Context, SessionMoveCheckpoint) error { rolled = true; return nil }}); err != nil {
		t.Fatal(err)
	}
	_, err = s.MoveSession(context.Background(), state.TraceID, SessionMoveRequest{ExecutionEpoch: epoch, Destination: destination})
	if !errors.As(err, &me) || me.Code != "DESTINATION_RESTORE_FAILED" || !rolled {
		t.Fatalf("rollback err=%v rolled=%v", err, rolled)
	}
	rt.mu.Lock()
	sess := rt.sessionLocked(state.TraceID)
	gotEpoch, gotProvider, moving := sess.executionEpoch, sess.placement.Provider, sess.moving
	rt.mu.Unlock()
	if gotEpoch != epoch || gotProvider != "a" || moving {
		t.Fatalf("source not authoritative epoch=%s provider=%s moving=%v", gotEpoch, gotProvider, moving)
	}
	t.Logf("ROLLBACK PASS session=%s authoritative_epoch=%s provider=%s refused=MOVE_UNSAFE restore_failure=DESTINATION_RESTORE_FAILED", state.TraceID, gotEpoch, gotProvider)
}

func TestSessionMoveHTTPReturnsTypedUnsafeAndSuccessfulCutover(t *testing.T) {
	state := SessionState{TraceID: "http-portable", Run: "RUNNING", Rev: 1}
	s := &Server{observeSession: func(context.Context, string) SessionState { return state }, sessionFeed: newSessionFeed(8)}
	source, dest := placement("pa", "aa", "ma", "ca"), placement("pb", "ab", "mb", "cb")
	if err := s.ConfigureSessionMove(state.TraceID, source, SessionMoveHooks{AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil }, RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return nil }}); err != nil {
		t.Fatal(err)
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	epoch := rt.sessionLocked(state.TraceID).executionEpoch
	rt.mu.Unlock()
	reqBody := SessionMoveRequest{ExecutionEpoch: epoch, Destination: dest}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/session/http-portable/move", bytes.NewReader(mustJSON(t, reqBody)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Principal-Kind", "human")
	s.handleFakSession(rr, req)
	if rr.Code != http.StatusConflict || !strings.Contains(rr.Body.String(), "MOVE_UNSAFE") {
		t.Fatalf("unsafe status=%d body=%s", rr.Code, rr.Body.String())
	}
	state.Run = "IDLE"
	if err := s.ConfigureSessionMove(state.TraceID, source, SessionMoveHooks{RequestSafePoint: func(context.Context, string) error { return nil }, AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil }, RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return nil }}); err != nil {
		t.Fatal(err)
	}
	rr = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/fak/session/http-portable/move", bytes.NewReader(mustJSON(t, reqBody)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Principal-Kind", "human")
	s.handleFakSession(rr, req)
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), "CUTOVER_COMMITTED") {
		t.Fatalf("move status=%d body=%s", rr.Code, rr.Body.String())
	}
}

func TestSessionMoveJournalHooksReplaysCommittedLineage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "move.journal")
	identity := sessionjournal.ResidencyIdentity{WorkspaceHead: "head", WorkspaceDirty: "clean", PolicyHash: "policy", ToolSchema: "tools", CredentialEpoch: "credential-ref", AdapterIdentity: "adapter"}
	if err := sessionjournal.AppendWorkEvent(path, sessionjournal.WorkEvent{SessionID: "portable-journal", Kind: sessionjournal.WorkSessionOpened, WriterEpoch: "writer", Residency: &identity}); err != nil {
		t.Fatal(err)
	}
	record := SessionMoveJournalHooks(path, "writer")
	for _, phase := range []SessionMovePhase{SessionMoveSafePointRequested, SessionMoveCheckpointed, SessionMoveDestinationAdmitted, SessionMoveRestored, SessionMoveCutoverCommitted} {
		if err := record(context.Background(), SessionMoveTransition{Phase: phase, SessionID: "portable-journal", SourceEpoch: "epoch-a", Destination: placement("provider-b", "account-ref-b", "model-b", "compute-b", "tools"), CheckpointHash: "sha256:checkpoint"}); err != nil {
			t.Fatal(err)
		}
	}
	replay, err := sessionjournal.ReplayWork(path)
	if err != nil {
		t.Fatal(err)
	}
	moves := replay.Sessions["portable-journal"].MoveTransitions
	if len(moves) != 5 || moves[4].Phase != "CUTOVER_COMMITTED" || moves[4].Destination.Compute != "compute-b" {
		t.Fatalf("moves=%+v", moves)
	}
}

func TestMoveSessionAdmissionRejectsCapabilityContextBudgetAndCompute(t *testing.T) {
	base := placement("p", "a", "m", "c", "tools")
	for name, mutate := range map[string]func(*SessionPlacement, *SessionMoveRequest){
		"capability": func(p *SessionPlacement, r *SessionMoveRequest) { r.RequiredCaps = []string{"vision"} },
		"context":    func(p *SessionPlacement, r *SessionMoveRequest) { r.RequiredContext = p.ContextLimit + 1 },
		"budget":     func(p *SessionPlacement, r *SessionMoveRequest) { r.RequiredBudget = p.BudgetAvailable + 1 },
		"compute":    func(p *SessionPlacement, r *SessionMoveRequest) { p.ComputeAvailable = false },
	} {
		t.Run(name, func(t *testing.T) {
			p := base
			r := SessionMoveRequest{Destination: p}
			mutate(&p, &r)
			r.Destination = p
			if err := validatePlacement(r.Destination, r.RequiredCaps, r.RequiredContext, r.RequiredBudget); err == nil {
				t.Fatal("admission unexpectedly passed")
			}
		})
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
