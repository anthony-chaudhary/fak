package gardenbundle

import "testing"

// Session-aware garden trigger witness for #5003 (#4990 root cause #3).
//
// The gap this pins: `fak leaseref audit` is a READ-ONLY detector, so it emits
// `ok: true` unconditionally (reporting IS the pass working) and carries its finding
// in `verdict: ACTION`. Interpret derived State from `ok` alone, so the stale_leases
// member folded to "ok" no matter what it found, PlanTick read "ok: nothing to
// remediate", and ActReap — registered for this member since #1386 — could never fire.
// A backlog made purely of expired SESSION descriptors (the 7,977-ref accretion in
// #4990) therefore sat there until a human ran `fak leaseref reap` by hand.
//
// Every case below drives the REAL `stale_leases` member out of the default `Members`
// bundle rather than a hand-built Member. That is deliberate: a synthetic member with
// ActOnVerdict set by the test would pass even with the production bundle unfixed, so
// it would witness nothing. Reading the shipped member is what makes this a failing-
// before/passing-after proof of the trigger the garden actually runs.

// staleLeasesMember returns the shipped stale_leases member from the default bundle.
func staleLeasesMember(t *testing.T) Member {
	t.Helper()
	for _, m := range Members {
		if m.Key == "stale_leases" {
			return m
		}
	}
	t.Fatalf("stale_leases is not in the default Members bundle; the garden no longer runs the lease audit")
	return Member{}
}

// auditEnvelope builds a fak.leaseref-audit-control-pane.v1 payload the way
// runLeaserefAudit emits it: ok is ALWAYS true, and the finding rides on verdict.
func auditEnvelope(verdict, reason string, extra map[string]any) map[string]any {
	env := map[string]any{
		"schema":                   "fak.leaseref-audit-control-pane.v1",
		"ok":                       true,
		"verdict":                  verdict,
		"reason":                   reason,
		"live_count":               float64(0),
		"expired_count":            float64(0),
		"expired_descriptor_count": float64(0),
		"expired_intent_count":     float64(0),
		"age_stale_count":          float64(0),
	}
	for k, v := range extra {
		env[k] = v
	}
	return env
}

// reapDecision returns the tick's decision row for stale_leases.
func reapDecision(t *testing.T, res MemberResult) ActDecision {
	t.Helper()
	plan := PlanTick([]MemberResult{res}, false)
	for _, d := range plan.Decisions {
		if d.Key == "stale_leases" {
			return d
		}
	}
	t.Fatalf("PlanTick produced no decision for stale_leases: %+v", plan.Decisions)
	return ActDecision{}
}

// TestSessionOnlyBacklogFiresActReap5003 is the headline witness: a backlog of ONLY
// expired session descriptors — zero expired lock leases — must reach ActReap.
func TestSessionOnlyBacklogFiresActReap5003(t *testing.T) {
	m := staleLeasesMember(t)
	env := auditEnvelope("ACTION",
		"0 live, 0 EXPIRED lease(s), 3 EXPIRED session descriptor(s) under refs/fak/locks/* — run `fak leaseref reap`",
		map[string]any{
			"expired_descriptor_count": float64(3),
			"expired_descriptor_ids":   []any{"session-a", "session-b", "session-c"},
		})

	res := Interpret(m, env, 0, "")
	if res.State != "action" {
		t.Fatalf("session-only backlog folded to state %q, want \"action\" — the audit reports ok=true and "+
			"carries its finding in verdict:ACTION, so a state derived from ok alone leaves ActReap unreachable (#5003)", res.State)
	}
	// Advisory, never red: the detector surfacing a condition must not gate the garden.
	if !res.OK {
		t.Fatalf("stale_leases must stay OK=true (advisory); got %+v", res)
	}

	d := reapDecision(t, res)
	if d.Act != ActReap {
		t.Fatalf("act = %q, want %q", d.Act, ActReap)
	}
	if !d.Perform {
		t.Fatalf("session-only backlog did not set Perform: %+v — the garden tick still would not reap it", d)
	}
	if plan := PlanTick([]MemberResult{res}, false); plan.ToReap != 1 {
		t.Fatalf("plan.ToReap = %d, want 1", plan.ToReap)
	}

	// The advisory condition must NOT move the garden gate (Fold keeps OK=true).
	if p := Fold([]MemberResult{res}, "/ws", "abc123"); !p.OK {
		t.Fatalf("an advisory stale_leases condition gated the garden red: %+v", p)
	}
}

// TestIntentOnlyBacklogFiresActReap5003 covers the third ref kind under the same
// namespace split: lapsed intents, which `fak leaseref reap` already collects.
func TestIntentOnlyBacklogFiresActReap5003(t *testing.T) {
	m := staleLeasesMember(t)
	env := auditEnvelope("ACTION",
		"0 live, 0 EXPIRED lease(s), 2 LAPSED intent(s) under refs/fak/locks/* — run `fak leaseref reap`",
		map[string]any{
			"expired_intent_count": float64(2),
			"expired_intent_ids":   []any{"intent-x", "intent-y"},
		})

	res := Interpret(m, env, 0, "")
	if res.State != "action" {
		t.Fatalf("intent-only backlog folded to state %q, want \"action\" (#5003)", res.State)
	}
	if d := reapDecision(t, res); !d.Perform || d.Act != ActReap {
		t.Fatalf("intent-only backlog did not reach ActReap: %+v", d)
	}
}

// TestLockLeaseBacklogStillFiresActReap5003 is the no-regression leg for the original
// lock-lease path named in the issue's acceptance contract.
func TestLockLeaseBacklogStillFiresActReap5003(t *testing.T) {
	m := staleLeasesMember(t)
	env := auditEnvelope("ACTION",
		"0 live, 2 EXPIRED lease(s) under refs/fak/locks/* — run `fak leaseref reap`",
		map[string]any{
			"expired_count": float64(2),
			"expired_ids":   []any{"lane-a", "lane-b"},
		})

	res := Interpret(m, env, 0, "")
	if res.State != "action" {
		t.Fatalf("lock-lease backlog folded to state %q, want \"action\"", res.State)
	}
	if d := reapDecision(t, res); !d.Perform || d.Act != ActReap {
		t.Fatalf("lock-lease backlog did not reach ActReap: %+v", d)
	}
}

// TestCleanAuditDoesNotReap5003 is the other half of the fence: a clean audit must
// stay "ok" and must NOT trigger a reap. Without this, "always act" would pass the
// cases above while making the garden run the reaper on every single tick.
func TestCleanAuditDoesNotReap5003(t *testing.T) {
	m := staleLeasesMember(t)
	env := auditEnvelope("OK", "4 live lease(s), 0 expired under refs/fak/locks/*", nil)

	res := Interpret(m, env, 0, "")
	if res.State != "ok" {
		t.Fatalf("clean audit folded to state %q, want \"ok\"", res.State)
	}
	d := reapDecision(t, res)
	if d.Perform {
		t.Fatalf("clean audit triggered a reap: %+v — the trigger must fire on a finding, not on every tick", d)
	}
	if plan := PlanTick([]MemberResult{res}, false); plan.ToReap != 0 {
		t.Fatalf("plan.ToReap = %d on a clean audit, want 0", plan.ToReap)
	}
}

// TestErroredAuditDoesNotReap5003 keeps the unmeasured-condition fence: a member that
// could not run is never acted on, even now that a verdict can drive the state.
func TestErroredAuditDoesNotReap5003(t *testing.T) {
	m := staleLeasesMember(t)
	res := Interpret(m, nil, -1, "timed out")
	if res.State != "errored" {
		t.Fatalf("state = %q, want \"errored\"", res.State)
	}
	if d := reapDecision(t, res); d.Perform {
		t.Fatalf("acted on an unmeasured condition: %+v", d)
	}
}

// TestOtherMembersUnchangedByVerdictOptIn5003 pins the blast radius: the opt-in is
// per-member, so a NON-opted-in member reporting ok=true with an advisory ACTION
// verdict (growthgate, whose remediation DELETES files) must still fold to "ok".
func TestOtherMembersUnchangedByVerdictOptIn5003(t *testing.T) {
	for _, m := range Members {
		if m.Key == "stale_leases" || m.Kind != "envelope" {
			continue
		}
		if m.ActOnVerdict {
			t.Fatalf("member %q also opted into verdict-driven action; #5003 arms stale_leases ONLY "+
				"(a blanket opt-in would newly arm file-deleting remediations)", m.Key)
		}
		env := auditEnvelope("ACTION", "advisory condition", nil)
		if res := Interpret(m, env, 0, ""); res.State != "ok" {
			t.Fatalf("member %q folded to %q on an ok=true/ACTION payload, want \"ok\" — "+
				"the opt-in leaked beyond stale_leases", m.Key, res.State)
		}
	}
}
