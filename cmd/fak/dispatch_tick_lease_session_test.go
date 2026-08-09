package main

// dispatch_tick_lease_session_test.go pins #5566 at the dispatch tick's acquire site: the
// lane lease it publishes now carries the SESSION BINDING that leaseref's liveness
// classification consumes. Before, `acquireDispatchLaneLease` built its leaseref.Record
// with ID/TreeGlobs/Holder/TTLSeconds and no SessionID, so ClassifyLiveness took its very
// first branch — peer-unknown / EvidenceNoBinding, fail-closed to not-reclaimable — for
// EVERY lane the tick ever held. A lane whose owner died without releasing was therefore
// unreclaimable by construction, and only the TTL ever freed it. #5565 covered the owner
// that EXITED (releaseAbandonedLaneLease on the spawn-failure strands); this is the other
// half, the owner that DIED, and it is a read-side fix: the write publishes the binding,
// the reader decides.
//
// WHICH SEAM THESE TESTS GUARD. There are two id fields one screen apart in
// acquireDispatchLaneLease and they are NOT the same thing:
//
//   - regionadmit.Request.SelfID, at ADMISSION, is compared against a LEASE id inside
//     regionadmit.Decide. It stays deliberately EMPTY so a live lease on this lane refuses
//     the next tick even when FAK_LEASE_OWNER pins the holder string — otherwise the fence
//     reads the new tick as the same holder, silently renews, and double-spawns the lane.
//   - leaseref.Record.SessionID, on the record being WRITTEN, is never read by Decide or
//     by AcquireFenced. It is a field on the published blob that only the read side looks
//     at, which is why stamping it cannot loosen any admission.
//
// Only the second is touched. TestDispatchLaneLeaseSessionBindingKeepsSelfIDRefusal is the
// guard against a later "simplification" that folds the two into one field: it holds a
// lane under a BOUND, heartbeating session and then demands that the same identity be
// refused anyway.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

// dispatchSessionLeaseFixture is a region test repo with the acquiring identity pinned:
// a fixed holder plus the session id under test. CLAUDE_CODE_SESSION_ID is cleared in
// every case so the binding a test observes is the one the test set, never one the agent
// harness running the suite happened to export into the environment.
func dispatchSessionLeaseFixture(t *testing.T, sessionID string) string {
	t.Helper()
	dir := initRegionTestRepo(t)
	t.Setenv("FAK_LEASE_OWNER", "worker-on-gateway")
	t.Setenv("FAK_SESSION_ID", sessionID)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "")
	return dir
}

// dispatchSessionPublish writes the session descriptor the classification reads —
// the same ref `fak leaseref session-publish` writes, straight through the store.
func dispatchSessionPublish(t *testing.T, dir, id, state string, updated time.Time, ttlSecs int64) {
	t.Helper()
	if _, err := leaseref.NewInDir(dir).PublishSession(context.Background(), leaseref.SessionDescriptor{
		ID:        id,
		Host:      "test-host",
		PCBState:  state,
		UpdatedAt: updated.Unix(),
		TTLSecs:   ttlSecs,
	}); err != nil {
		t.Fatalf("publish session %s: %v", id, err)
	}
}

// dispatchClassifiedLease reads the lease back through the REAL classification an arbiter
// would run (`Store.ClassifyLive`), with an anonymous reader — a different tick, so the
// row must classify peer-* and never self.
func dispatchClassifiedLease(t *testing.T, dir, leaseID string, now time.Time) leaseref.ClassifiedLease {
	t.Helper()
	rows, err := leaseref.NewInDir(dir).ClassifyLive(context.Background(), "", now)
	if err != nil {
		t.Fatalf("classify live leases: %v", err)
	}
	for _, r := range rows {
		if r.ID == leaseID {
			return r
		}
	}
	t.Fatalf("lease %q is not in the live classified set %+v", leaseID, rows)
	return leaseref.ClassifiedLease{}
}

func dispatchAcquireGateway(t *testing.T, dir, id string) map[string]any {
	t.Helper()
	return acquireDispatchLaneLease(dir, id, "gateway", []string{"internal/gateway/**"}, 1800, "")
}

// TestDispatchLaneLeaseStoppedSessionIsPeerDeadReclaimable is checkable end state (1): a
// lane the tick acquired, whose owning session then published the terminal STOPPED state,
// classifies peer-dead and IS reclaimable. Pre-#5566 the identical sequence returned
// peer-unknown/no-session-binding, because the written record had nothing to look up.
func TestDispatchLaneLeaseStoppedSessionIsPeerDeadReclaimable(t *testing.T) {
	dir := dispatchSessionLeaseFixture(t, "dispatch-sess-dead")

	lease := dispatchAcquireGateway(t, dir, "resolve-gateway")
	if acquired, _ := lease["acquired"].(bool); !acquired {
		t.Fatalf("a free gateway lane must acquire, got %+v", lease)
	}

	// The binding is on the PUBLISHED record, not merely in the result map: read the ref
	// back, because the blob is what a peer on another machine actually classifies.
	rec, ok, err := leaseref.NewInDir(dir).Get(context.Background(), "resolve-gateway")
	if err != nil || !ok {
		t.Fatalf("read back the written lease: ok=%v err=%v", ok, err)
	}
	if rec.SessionID != "dispatch-sess-dead" {
		t.Fatalf("written lease session_id = %q, want %q — an unbound record can only ever classify peer-unknown", rec.SessionID, "dispatch-sess-dead")
	}

	now := time.Now()
	dispatchSessionPublish(t, dir, "dispatch-sess-dead", "STOPPED", now, 3600)

	got := dispatchClassifiedLease(t, dir, "resolve-gateway", now)
	if got.Liveness != leaseref.LivenessPeerDead {
		t.Fatalf("liveness = %q, want %q (evidence: %s)", got.Liveness, leaseref.LivenessPeerDead, got.Evidence)
	}
	if !got.Reclaimable {
		t.Fatalf("reclaimable = false for a positively dead owner; only the TTL would ever free this lane (evidence: %s)", got.Evidence)
	}
	if got.EvidenceKind != leaseref.EvidenceTerminalStopped {
		t.Fatalf("evidence_kind = %q, want %q", got.EvidenceKind, leaseref.EvidenceTerminalStopped)
	}
}

// TestDispatchLaneLeaseLiveSessionIsPeerLiveNotReclaimable is checkable end state (2), and
// it is the half that keeps end state (1) honest: binding the session must not make lanes
// stealable in general. A heartbeating owner classifies peer-live and is NEVER reclaimable
// — this is the lane-steal the whole classification exists to prevent.
func TestDispatchLaneLeaseLiveSessionIsPeerLiveNotReclaimable(t *testing.T) {
	dir := dispatchSessionLeaseFixture(t, "dispatch-sess-live")

	lease := dispatchAcquireGateway(t, dir, "resolve-gateway")
	if acquired, _ := lease["acquired"].(bool); !acquired {
		t.Fatalf("a free gateway lane must acquire, got %+v", lease)
	}

	now := time.Now()
	dispatchSessionPublish(t, dir, "dispatch-sess-live", "RUNNING", now, 3600)

	got := dispatchClassifiedLease(t, dir, "resolve-gateway", now)
	if got.Liveness != leaseref.LivenessPeerLive {
		t.Fatalf("liveness = %q, want %q (evidence: %s)", got.Liveness, leaseref.LivenessPeerLive, got.Evidence)
	}
	if got.Reclaimable {
		t.Fatalf("reclaimable = true while the owner is heartbeating — that is the lane steal, not a reclaim (evidence: %s)", got.Evidence)
	}
	if got.EvidenceKind != leaseref.EvidenceHeartbeating {
		t.Fatalf("evidence_kind = %q, want %q", got.EvidenceKind, leaseref.EvidenceHeartbeating)
	}
}

// TestDispatchLaneLeaseSessionBindingKeepsSelfIDRefusal is checkable end state (3) and the
// trap guard. The pre-existing decision — regionadmit.Request.SelfID stays EMPTY at
// admission — must survive the session binding untouched: a previous worker on this lane
// that is still running has to refuse the next tick even though FAK_LEASE_OWNER is pinned
// to the same holder AND the tick now names the same session.
//
// It fails if anyone folds the two id fields into one. Feeding the LEASE id into SelfID
// makes Decide skip the live lease and admit the re-acquire (which then renews the fence
// under the same holder and double-spawns the lane) — the same-id case below catches that.
// Feeding the SESSION id in is caught by the same case whenever the two ids coincide, and
// the differently-named case catches any fold that grants self-skip on holder identity.
func TestDispatchLaneLeaseSessionBindingKeepsSelfIDRefusal(t *testing.T) {
	dir := dispatchSessionLeaseFixture(t, "dispatch-sess-live")

	// Pre-condition: the binding really is active. Without this the refusals below would
	// pass vacuously against an unbound lease and would guard nothing.
	if got := dispatchLeaseSessionID(); got != "dispatch-sess-live" {
		t.Fatalf("fixture session binding = %q, want %q — the refusal must be witnessed WITH a binding in place", got, "dispatch-sess-live")
	}

	first := dispatchAcquireGateway(t, dir, "resolve-gateway")
	if acquired, _ := first["acquired"].(bool); !acquired {
		t.Fatalf("the first tick must acquire the free lane, got %+v", first)
	}

	// The previous worker is still running: it heartbeats, so the read side says peer-live.
	now := time.Now()
	dispatchSessionPublish(t, dir, "dispatch-sess-live", "RUNNING", now, 3600)
	if got := dispatchClassifiedLease(t, dir, "resolve-gateway", now); got.Liveness != leaseref.LivenessPeerLive {
		t.Fatalf("precondition: lane liveness = %q, want peer-live (evidence: %s)", got.Liveness, got.Evidence)
	}

	// Same lease id, same holder, same session — the exact shape the load-bearing comment
	// above `req` describes. It must still be REFUSED at admission.
	same := dispatchAcquireGateway(t, dir, "resolve-gateway")
	if refused, _ := same["refused"].(bool); !refused {
		t.Fatalf("a re-acquire under the same id/holder/session must refuse; admitting it renews the fence and double-spawns the lane, got %+v", same)
	}
	if same["reason"] != "COLLISION_RISK" || same["rung"] != regionadmit.RungSameLane {
		t.Fatalf("re-acquire refusal = reason %v rung %v, want COLLISION_RISK/%s", same["reason"], same["rung"], regionadmit.RungSameLane)
	}
	if acquired, _ := same["acquired"].(bool); acquired {
		t.Fatalf("refused re-acquire must not also report acquired, got %+v", same)
	}

	// A differently-named tick on the same lane under the same identity likewise.
	other := dispatchAcquireGateway(t, dir, "resolve-gateway-2")
	if refused, _ := other["refused"].(bool); !refused {
		t.Fatalf("a second tick on a live lane must refuse regardless of shared identity, got %+v", other)
	}
	if other["rung"] != regionadmit.RungSameLane {
		t.Fatalf("second-tick refusal rung = %v, want %s", other["rung"], regionadmit.RungSameLane)
	}

	// And the live lease is untouched by the two refusals: still one lane, still bound to
	// the running owner. A silent renew would have rewritten it instead.
	rec, ok, err := leaseref.NewInDir(dir).Get(context.Background(), "resolve-gateway")
	if err != nil || !ok {
		t.Fatalf("read back the held lease: ok=%v err=%v", ok, err)
	}
	if rec.SessionID != "dispatch-sess-live" || rec.Holder != "worker-on-gateway" {
		t.Fatalf("held lease = holder %q session %q, want the original worker's record", rec.Holder, rec.SessionID)
	}
}

// TestDispatchLaneLeaseSkipsUnusableSessionID is the fail-open control. Binding is an
// improvement on a best-effort path, so an absent or unusable session id must degrade to
// EXACTLY the pre-#5566 record (empty session_id, peer-unknown, not reclaimable) and never
// to a failed acquire or a record pointing at a ref that cannot exist.
func TestDispatchLaneLeaseSkipsUnusableSessionID(t *testing.T) {
	t.Run("unset binds nothing and stays peer-unknown", func(t *testing.T) {
		dir := dispatchSessionLeaseFixture(t, "")

		lease := dispatchAcquireGateway(t, dir, "resolve-gateway")
		if acquired, _ := lease["acquired"].(bool); !acquired {
			t.Fatalf("a missing session id must not break the acquire, got %+v", lease)
		}
		rec, ok, err := leaseref.NewInDir(dir).Get(context.Background(), "resolve-gateway")
		if err != nil || !ok {
			t.Fatalf("read back the written lease: ok=%v err=%v", ok, err)
		}
		if rec.SessionID != "" {
			t.Fatalf("written lease session_id = %q, want empty when no session env is set", rec.SessionID)
		}
		got := dispatchClassifiedLease(t, dir, "resolve-gateway", time.Now())
		if got.Liveness != leaseref.LivenessPeerUnknown || got.Reclaimable {
			t.Fatalf("liveness = %q reclaimable = %v, want peer-unknown and not reclaimable", got.Liveness, got.Reclaimable)
		}
		if got.EvidenceKind != leaseref.EvidenceNoBinding {
			t.Fatalf("evidence_kind = %q, want %q — the remedy this names is the acquire call site", got.EvidenceKind, leaseref.EvidenceNoBinding)
		}
	})

	t.Run("malformed value is skipped not written", func(t *testing.T) {
		// A harness that exports a value which cannot be one ref segment must cost the
		// binding, nothing more: the lane still acquires.
		dir := dispatchSessionLeaseFixture(t, "refs/heads/not a segment")

		lease := dispatchAcquireGateway(t, dir, "resolve-gateway")
		if acquired, _ := lease["acquired"].(bool); !acquired {
			t.Fatalf("a malformed session id must not break the acquire, got %+v", lease)
		}
		rec, ok, err := leaseref.NewInDir(dir).Get(context.Background(), "resolve-gateway")
		if err != nil || !ok {
			t.Fatalf("read back the written lease: ok=%v err=%v", ok, err)
		}
		if rec.SessionID != "" {
			t.Fatalf("written lease session_id = %q, want empty — a malformed id must be skipped, not published", rec.SessionID)
		}
	})

	t.Run("id rule confines to one safe ref segment", func(t *testing.T) {
		for _, id := range []string{
			"guard-abc123",
			"dispatch-sess-live",
			"0f8a5b2c-1d3e-4f60-8a91-2b3c4d5e6f70",
			"a.b_c-d",
		} {
			if !validDispatchLeaseSessionID(id) {
				t.Errorf("validDispatchLeaseSessionID(%q) = false, want true", id)
			}
		}
		for _, id := range []string{
			"",
			"session-abc",            // would address session-session-abc
			"-lead",                  // misparses as a flag
			".lead",                  // ref-illegal
			"has space",              //
			"refs/heads/main",        // escapes the namespace
			"tilde~1",                // ref-special
			strings.Repeat("x", 201), // over the length bound
		} {
			if validDispatchLeaseSessionID(id) {
				t.Errorf("validDispatchLeaseSessionID(%q) = true, want false", id)
			}
		}
	})
}
