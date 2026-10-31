package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

type sessionMoveRollbackFixture struct {
	server      *Server
	runtime     *sessionClientRuntime
	id          string
	epoch       string
	source      SessionPlacement
	destination SessionPlacement
	attachment  sessionClientAttachment
	effect      SessionEffect
}

func newSessionMoveRollbackFixture(t *testing.T) *sessionMoveRollbackFixture {
	t.Helper()
	state := SessionState{TraceID: "move-rollback-fixture", Run: "IDLE", Rev: 1}
	s := &Server{observeSession: func(context.Context, string) SessionState { return state }, sessionFeed: newSessionFeed(8)}
	f := &sessionMoveRollbackFixture{
		server: s, runtime: s.clientRuntime(), id: state.TraceID,
		source:      placement("source", "source-ref", "source-model", "source-compute", "tools"),
		destination: placement("destination", "destination-ref", "destination-model", "destination-compute", "tools"),
		attachment:  sessionClientAttachment{SessionID: state.TraceID, WorkspaceID: "original-workspace", Kind: "human"},
		effect:      SessionEffect{ID: "effect-1", Command: "fixture-only", Verdict: SessionEffectKnownNotRun},
	}
	// Match the stored placement contract before any movement or failure occurs.
	f.source = normalizedPlacement(f.source)
	f.destination = normalizedPlacement(f.destination)
	f.configure(t, SessionMoveHooks{})
	f.runtime.mu.Lock()
	sess := f.runtime.sessionLocked(f.id)
	f.epoch = sess.executionEpoch
	sess.transcript = []byte("retained transcript\n")
	sess.effects[f.effect.ID] = f.effect
	f.runtime.attachments["original"] = f.attachment
	f.runtime.leases[f.id] = "original"
	f.runtime.mu.Unlock()
	t.Cleanup(func() { sessionClientRuntimes.Delete(s) })
	return f
}

func (f *sessionMoveRollbackFixture) configure(t *testing.T, hooks SessionMoveHooks) {
	t.Helper()
	if err := f.server.ConfigureSessionMove(f.id, f.source, hooks); err != nil {
		t.Fatal(err)
	}
}

func (f *sessionMoveRollbackFixture) request() SessionMoveRequest {
	return SessionMoveRequest{ExecutionEpoch: f.epoch, Destination: f.destination}
}

func (f *sessionMoveRollbackFixture) assertSource(t *testing.T, epoch string, source SessionPlacement) {
	t.Helper()
	f.runtime.mu.Lock()
	defer f.runtime.mu.Unlock()
	sess := f.runtime.sessions[f.id]
	if sess == nil || sess.executionEpoch != epoch || !reflect.DeepEqual(sess.placement, source) || sess.moving {
		t.Fatalf("source state changed: session=%+v epoch=%s placement=%+v", sess, epoch, source)
	}
	if f.runtime.attachments["original"] != f.attachment || f.runtime.leases[f.id] != "original" {
		t.Fatal("pre-cutover failure changed source attachment or lease")
	}
	if string(sess.transcript) != "retained transcript\n" || !reflect.DeepEqual(sess.effects, map[string]SessionEffect{f.effect.ID: f.effect}) || len(sess.lastMove) != 0 {
		t.Fatal("pre-cutover failure changed transcript, effects, or committed movement history")
	}
}

func TestMoveSessionCompensatesAdmittedFailures(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		journalPhase                   SessionMovePhase
		missingRestore, restoreFailure bool
		commitFailure, staleEpoch      bool
		wantCode                       string
		wantStatus, wantTransitions    int
		wantRestore, wantCommit        int
	}{
		{name: "admitted journal", journalPhase: SessionMoveDestinationAdmitted, wantCode: "MOVE_JOURNAL_FAILED", wantStatus: http.StatusServiceUnavailable, wantTransitions: 2},
		{name: "missing restore", missingRestore: true, wantCode: "DESTINATION_RESTORE_UNAVAILABLE", wantStatus: http.StatusNotImplemented, wantTransitions: 3},
		{name: "restore", restoreFailure: true, wantCode: "DESTINATION_RESTORE_FAILED", wantStatus: http.StatusBadGateway, wantTransitions: 3, wantRestore: 1},
		{name: "restored journal", journalPhase: SessionMoveRestored, wantCode: "MOVE_JOURNAL_FAILED", wantStatus: http.StatusServiceUnavailable, wantTransitions: 3, wantRestore: 1},
		{name: "destination commit", commitFailure: true, wantCode: "CUTOVER_FAILED", wantStatus: http.StatusBadGateway, wantTransitions: 4, wantRestore: 1, wantCommit: 1},
		{name: "stale after destination commit", staleEpoch: true, wantCode: "STALE_EPOCH", wantStatus: http.StatusConflict, wantTransitions: 4, wantRestore: 1, wantCommit: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newSessionMoveRollbackFixture(t)
			primary := errors.New("primary movement failure")
			admissions, restores, commits, rollbacks := 0, 0, 0, 0
			var admitted, compensated SessionMoveCheckpoint
			wantEpoch, wantPlacement := f.epoch, f.source
			hooks := SessionMoveHooks{
				AdmitDestination: func(_ context.Context, _ string, cp SessionMoveCheckpoint, _ SessionMoveRequest) error {
					admissions++
					admitted = cp
					return nil
				},
				RestoreDestination: func(context.Context, SessionMoveCheckpoint) error {
					restores++
					if tc.restoreFailure {
						return primary
					}
					return nil
				},
				CommitDestination: func(context.Context, SessionMoveCheckpoint) error {
					commits++
					if tc.staleEpoch {
						wantEpoch = "concurrent-newer-epoch"
						wantPlacement = placement("concurrent", "concurrent-ref", "concurrent-model", "concurrent-compute", "tools")
						f.runtime.mu.Lock()
						sess := f.runtime.sessions[f.id]
						sess.executionEpoch, sess.placement, sess.moving = wantEpoch, wantPlacement, false
						f.runtime.mu.Unlock()
					}
					if tc.commitFailure {
						return primary
					}
					return nil
				},
				RollbackDestination: func(_ context.Context, cp SessionMoveCheckpoint) error {
					rollbacks++
					compensated = cp
					return nil
				},
				RecordTransition: func(_ context.Context, tr SessionMoveTransition) error {
					if tr.Phase == tc.journalPhase {
						return primary
					}
					return nil
				},
			}
			if tc.missingRestore {
				hooks.RestoreDestination = nil
			}
			f.configure(t, hooks)
			resp, err := f.server.MoveSession(context.Background(), f.id, f.request())
			var me *SessionMoveError
			if !errors.As(err, &me) || me.Code != tc.wantCode || me.Status != tc.wantStatus || me.CauseCode != "" {
				t.Fatalf("failure=%+v, want %s/%d with confirmed compensation", err, tc.wantCode, tc.wantStatus)
			}
			if resp.Schema != "" || resp.Descriptor.SessionID != "" {
				t.Fatalf("failed movement reported a success response: %+v", resp)
			}
			if admissions != 1 || rollbacks != 1 || restores != tc.wantRestore || commits != tc.wantCommit {
				t.Fatalf("calls admission/restore/commit/rollback=%d/%d/%d/%d", admissions, restores, commits, rollbacks)
			}
			if !reflect.DeepEqual(admitted, compensated) || compensated.SessionID != f.id || compensated.SourceEpoch != f.epoch || compensated.Digest != checkpointDigest(compensated) {
				t.Fatalf("rollback did not receive the exact admitted checkpoint: %+v", compensated)
			}
			if len(me.Transitions) != tc.wantTransitions {
				t.Fatalf("transitions=%+v, want %d confirmed transitions", me.Transitions, tc.wantTransitions)
			}
			for _, tr := range me.Transitions {
				if tr.Phase == SessionMoveCutoverCommitted {
					t.Fatal("failed pre-cutover move reported committed cutover")
				}
			}
			if !tc.missingRestore && !tc.staleEpoch && !errors.Is(err, primary) {
				t.Fatal("original callback cause was lost")
			}
			f.assertSource(t, wantEpoch, wantPlacement)
		})
	}
}

func TestMoveSessionDoesNotCompensateBeforeAdmission(t *testing.T) {
	for _, stage := range []string{"safe-point journal", "safe-point hook", "checkpoint journal", "missing admission", "admission refusal"} {
		t.Run(stage, func(t *testing.T) {
			f := newSessionMoveRollbackFixture(t)
			primary := errors.New("pre-admission failure")
			admissions, restores, rollbacks := 0, 0, 0
			hooks := SessionMoveHooks{
				RequestSafePoint: func(context.Context, string) error {
					if stage == "safe-point hook" {
						return primary
					}
					return nil
				},
				AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error {
					admissions++
					return primary
				},
				RestoreDestination:  func(context.Context, SessionMoveCheckpoint) error { restores++; return nil },
				RollbackDestination: func(context.Context, SessionMoveCheckpoint) error { rollbacks++; return nil },
				RecordTransition: func(_ context.Context, tr SessionMoveTransition) error {
					if stage == "safe-point journal" && tr.Phase == SessionMoveSafePointRequested || stage == "checkpoint journal" && tr.Phase == SessionMoveCheckpointed {
						return primary
					}
					return nil
				},
			}
			if stage == "missing admission" {
				hooks.AdmitDestination = nil
			}
			f.configure(t, hooks)
			_, err := f.server.MoveSession(context.Background(), f.id, f.request())
			var me *SessionMoveError
			if !errors.As(err, &me) || me.CauseCode != "" || me.Code == "DESTINATION_ROLLBACK_RECOVERY_REQUIRED" {
				t.Fatalf("pre-admission failure=%v", err)
			}
			wantAdmissions := 0
			if stage == "admission refusal" {
				wantAdmissions = 1
			}
			if admissions != wantAdmissions || restores != 0 || rollbacks != 0 {
				t.Fatalf("pre-admission calls=%d/%d/%d", admissions, restores, rollbacks)
			}
			f.assertSource(t, f.epoch, f.source)
		})
	}
}

func TestMoveSessionRollbackFailureRequiresRecovery(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "failing hook"
		if missing {
			name = "missing hook"
		}
		t.Run(name, func(t *testing.T) {
			f := newSessionMoveRollbackFixture(t)
			primary, cleanup := errors.New("primary-internal-detail"), errors.New("cleanup-internal-detail")
			rollbacks := 0
			hooks := SessionMoveHooks{
				AdmitDestination:    func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil },
				RestoreDestination:  func(context.Context, SessionMoveCheckpoint) error { return primary },
				RollbackDestination: func(context.Context, SessionMoveCheckpoint) error { rollbacks++; return cleanup },
			}
			if missing {
				hooks.RollbackDestination = nil
			}
			f.configure(t, hooks)
			_, err := f.server.MoveSession(context.Background(), f.id, f.request())
			var me *SessionMoveError
			if !errors.As(err, &me) || me.Code != "DESTINATION_ROLLBACK_RECOVERY_REQUIRED" || me.CauseCode != "DESTINATION_RESTORE_FAILED" || me.Status != http.StatusServiceUnavailable {
				t.Fatalf("unconfirmed cleanup=%+v", err)
			}
			if !errors.Is(err, primary) || !missing && !errors.Is(err, cleanup) {
				t.Fatal("primary or rollback callback cause was discarded")
			}
			if strings.Contains(me.Message, primary.Error()) || strings.Contains(me.Message, cleanup.Error()) {
				t.Fatal("recovery message exposes internal callback details")
			}
			wantRollbacks := 1
			if missing {
				wantRollbacks = 0
			}
			if rollbacks != wantRollbacks {
				t.Fatalf("rollback calls=%d, want %d", rollbacks, wantRollbacks)
			}
			// Recovery-required is not a quarantine claim: retain the existing source
			// authority/moving-reset behavior and require the caller to reconcile.
			f.assertSource(t, f.epoch, f.source)
		})
	}
}

func TestMoveSessionRollbackUsesBoundedIndependentContext(t *testing.T) {
	type requestKey struct{}
	f := newSessionMoveRollbackFixture(t)
	ctx, cancel := context.WithCancel(context.WithValue(context.Background(), requestKey{}, "request-value"))
	defer cancel()
	primary := errors.New("restore interrupted")
	var cleanupCtx context.Context
	var cleanupErr error
	var cleanupValue any
	var remaining time.Duration
	var hasDeadline bool
	rollbacks := 0
	f.configure(t, SessionMoveHooks{
		AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil },
		RestoreDestination: func(context.Context, SessionMoveCheckpoint) error {
			cancel()
			return primary
		},
		RollbackDestination: func(ctx context.Context, _ SessionMoveCheckpoint) error {
			rollbacks++
			cleanupCtx, cleanupErr, cleanupValue = ctx, ctx.Err(), ctx.Value(requestKey{})
			var deadline time.Time
			deadline, hasDeadline = ctx.Deadline()
			remaining = time.Until(deadline)
			return nil
		},
	})
	_, err := f.server.MoveSession(ctx, f.id, f.request())
	var me *SessionMoveError
	if !errors.As(err, &me) || me.Code != "DESTINATION_RESTORE_FAILED" || !errors.Is(err, primary) {
		t.Fatalf("canceled request result=%v", err)
	}
	if ctx.Err() != context.Canceled || rollbacks != 1 || cleanupErr != nil || cleanupValue != "request-value" || !hasDeadline || remaining <= 0 || remaining > sessionMoveRollbackTimeout {
		t.Fatalf("cleanup context: request=%v calls=%d err=%v value=%v deadline=%v remaining=%v", ctx.Err(), rollbacks, cleanupErr, cleanupValue, hasDeadline, remaining)
	}
	select {
	case <-cleanupCtx.Done():
	default:
		t.Fatal("cleanup context was not canceled after the hook returned")
	}
	f.assertSource(t, f.epoch, f.source)
}

func TestMoveSessionDoesNotRollbackAuthoritativeCutover(t *testing.T) {
	for _, journalFailure := range []bool{false, true} {
		name := "success"
		if journalFailure {
			name = "committed journal recovery"
		}
		t.Run(name, func(t *testing.T) {
			f := newSessionMoveRollbackFixture(t)
			commits, rollbacks := 0, 0
			f.configure(t, SessionMoveHooks{
				AdmitDestination:    func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil },
				RestoreDestination:  func(context.Context, SessionMoveCheckpoint) error { return nil },
				CommitDestination:   func(context.Context, SessionMoveCheckpoint) error { commits++; return nil },
				RollbackDestination: func(context.Context, SessionMoveCheckpoint) error { rollbacks++; return nil },
				RecordTransition: func(_ context.Context, tr SessionMoveTransition) error {
					if journalFailure && tr.Phase == SessionMoveCutoverCommitted {
						return errors.New("post-cutover journal failure")
					}
					return nil
				},
			})
			resp, err := f.server.MoveSession(context.Background(), f.id, f.request())
			if journalFailure {
				var me *SessionMoveError
				if !errors.As(err, &me) || me.Code != "CUTOVER_JOURNAL_RECOVERY_REQUIRED" || me.CauseCode != "" {
					t.Fatalf("post-cutover error=%v", err)
				}
			} else if err != nil || resp.Descriptor.ExecutionEpoch == f.epoch {
				t.Fatalf("successful cutover response=%+v err=%v", resp, err)
			}
			if commits != 1 || rollbacks != 0 {
				t.Fatalf("post-cutover calls commit=%d rollback=%d", commits, rollbacks)
			}
			f.runtime.mu.Lock()
			defer f.runtime.mu.Unlock()
			sess := f.runtime.sessions[f.id]
			if sess.executionEpoch == f.epoch || !reflect.DeepEqual(sess.placement, f.destination) || sess.moving {
				t.Fatalf("committed destination authority changed: %+v", sess)
			}
			if _, attached := f.runtime.attachments["original"]; attached {
				t.Fatal("cutover retained old attachment authority")
			}
			if _, leased := f.runtime.leases[f.id]; leased {
				t.Fatal("cutover retained old input lease")
			}
		})
	}
}

func TestSessionMoveHTTPReportsRollbackRecoveryWithoutCallbackDetails(t *testing.T) {
	f := newSessionMoveRollbackFixture(t)
	rollbacks := 0
	f.configure(t, SessionMoveHooks{
		AdmitDestination:   func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil },
		RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return errors.New("primary-internal-detail") },
		RollbackDestination: func(context.Context, SessionMoveCheckpoint) error {
			rollbacks++
			return errors.New("cleanup-internal-detail")
		},
	})
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/session/"+f.id+"/move", bytes.NewReader(mustJSON(t, f.request())))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Principal-Kind", "human")
	f.server.handleFakSession(rr, req)
	var body struct {
		Error       string                  `json:"error"`
		CauseCode   string                  `json:"cause_code"`
		Message     string                  `json:"message"`
		Transitions []SessionMoveTransition `json:"transitions"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if rr.Code != http.StatusServiceUnavailable || body.Error != "DESTINATION_ROLLBACK_RECOVERY_REQUIRED" || body.CauseCode != "DESTINATION_RESTORE_FAILED" || rollbacks != 1 {
		t.Fatalf("HTTP recovery: status=%d body=%+v rollback=%d", rr.Code, body, rollbacks)
	}
	if strings.Contains(rr.Body.String(), "primary-internal-detail") || strings.Contains(rr.Body.String(), "cleanup-internal-detail") {
		t.Fatal("HTTP recovery exposed callback details")
	}
	if len(body.Transitions) != 3 || body.Transitions[2].Phase != SessionMoveDestinationAdmitted {
		t.Fatalf("HTTP recovery lost confirmed transitions: %+v", body.Transitions)
	}
	f.assertSource(t, f.epoch, f.source)
}

func TestSessionMoveRollbackPlacementCanonicalization(t *testing.T) {
	for _, input := range []SessionPlacement{
		{Provider: "source", AccountRef: "source-ref", Model: "source-model", Compute: "source-compute", Capabilities: []string{"tools"}},
		{Provider: "destination", AccountRef: "destination-ref", Model: "destination-model", Compute: "destination-compute", Capabilities: []string{"tools"}},
	} {
		want := input
		want.SemanticDegradations = []string{}
		got := normalizedPlacement(input)
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(normalizedPlacement(got), want) {
			t.Fatalf("placement is not canonical and idempotent: got=%#v want=%#v", got, want)
		}
		if input.SemanticDegradations != nil {
			t.Fatal("normalization mutated the original placement")
		}
		if reflect.DeepEqual(got, input) {
			t.Fatal("raw nil-list fixture unexpectedly equals canonical stored placement")
		}
	}
}

func TestSessionMoveRollbackPlacementCanonicalFields(t *testing.T) {
	input := SessionPlacement{
		Provider: " provider ", AccountRef: " account ", Model: " model ", Compute: " compute ",
		Capabilities: []string{" tools ", "chat", "tools", ""},
		ContextLimit: 123, BudgetAvailable: 456, ComputeAvailable: true, CacheLineage: "lineage",
		SemanticDegradations: []string{" second ", "first", "second"},
	}
	want := SessionPlacement{
		Provider: "provider", AccountRef: "account", Model: "model", Compute: "compute",
		Capabilities: []string{"chat", "tools"},
		ContextLimit: 123, BudgetAvailable: 456, ComputeAvailable: true, CacheLineage: "lineage",
		SemanticDegradations: []string{"first", "second"},
	}
	got := normalizedPlacement(input)
	if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(normalizedPlacement(got), want) {
		t.Fatalf("canonical placement changed fields: got=%#v want=%#v", got, want)
	}
	if input.Capabilities[0] != " tools " || input.SemanticDegradations[0] != " second " {
		t.Fatal("normalization mutated input lists")
	}
}
