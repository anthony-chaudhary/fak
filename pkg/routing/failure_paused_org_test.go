package routing

import (
	"errors"
	"testing"
)

// pausedOrgBody is the verbatim body Hive AI returns while the organization is
// paused. It is the wire evidence the whole router-only change turns on: with no
// signature match this body would classify 405 -> PAYLOAD_REJECTED -> ActionFatal
// and kill the session instead of failing over.
const pausedOrgBody = `{"status_code":405,"message":"Your Organization is currently paused. Please check your account balance."}`

// TestPausedOrg405FailsOverInsteadOfDieing is the load-bearing severity witness:
// routing a call through the router only helps if the router MOVES ON when the
// Hive organization is paused. Before this change a 405 classified
// PAYLOAD_REJECTED (ActionFatal), which is how every session on the paused
// provider died on 2026-10-02.
func TestPausedOrg405FailsOverInsteadOfDieing(t *testing.T) {
	f := Classify(405, pausedOrgBody)
	if f.Class != FailureCreditExhausted {
		t.Fatalf("Classify(405, paused) class = %s, want %s", f.Class, FailureCreditExhausted)
	}
	if f.Action != ActionFailoverNext {
		t.Errorf("action = %s, want %s", f.Action, ActionFailoverNext)
	}
	if f.Retries != 0 {
		t.Errorf("retries = %d, want 0 (a walled org cannot answer a retry)", f.Retries)
	}
	// The retry budget must authorize the FULL credential headroom so a
	// keyring-backed walk rolls to a DIFFERENT key rather than pinning itself.
	if got := f.RetryBudget(3); got != 3 {
		t.Errorf("RetryBudget(3) = %d, want 3 (roll to another credential)", got)
	}
}

// TestPausedOrgBodyClassifiesOnItsOwnChannels is the org-pause BODY rung: the
// wall is named where a pause is the provider's meaning — a 405 and an in-band
// error on a 2xx. Other statuses are deliberately excluded; see the shadow test.
func TestPausedOrgBodyClassifiesOnItsOwnChannels(t *testing.T) {
	for _, status := range []int{200, 299, 405} {
		if got := Classify(status, pausedOrgBody); got.Class != FailureCreditExhausted {
			t.Errorf("Classify(%d, paused) class = %s, want %s", status, got.Class, FailureCreditExhausted)
		}
	}
}

// TestBareOrgPaused405FailsOverWithoutRetiringTheKey pins the mirror case the
// catalog names PROVIDER_TRANSIENT_405. The repo's own authority record says the
// provider plane uses 405 exclusively for an org pause "often with an empty
// body", and a live direct probe on this host returned exactly that: 405 with an
// EMPTY body. Leaving the bare case FATAL (the old PAYLOAD_REJECTED verdict)
// therefore killed every routed call on the paused provider. It must now fail
// over AND stay clear of the auth/credit verdicts, because those are what retire
// a credential — and the prescribed recovery is "retry; do NOT tombstone the key".
func TestBareOrgPaused405FailsOverWithoutRetiringTheKey(t *testing.T) {
	for _, body := range []string{
		"",
		`{"error":"Method Not Allowed"}`,
		`{"status_code":405,"message":"The requested path is not routable on this host."}`,
	} {
		f := Classify(405, body)
		if f.Action != ActionFailoverNext {
			t.Errorf("Classify(405, %q) action = %s, want %s", body, f.Action, ActionFailoverNext)
		}
		if f.Class == FailureAuthRejected || f.Class == FailureCreditExhausted {
			t.Errorf("Classify(405, %q) class = %s; a bare 405 must not retire the credential", body, f.Class)
		}
		if f.Class == FailureBadPayload {
			t.Errorf("Classify(405, %q) class = %s; a bare 405 must not be fatal", body, f.Class)
		}
	}
}

// TestPausedOrgSignatureDoesNotShadowAuthOrCreditOnOtherStatuses keeps the pause
// signature from hijacking a verdict that belongs to a different wall. A 401/403
// carrying the paused body is an AUTH verdict (which retires the key); a 402 is a
// credit wall. Only 405 is the org-pause channel on this plane.
func TestPausedOrgSignatureDoesNotShadowAuthOrCreditOnOtherStatuses(t *testing.T) {
	if got := Classify(401, pausedOrgBody).Class; got != FailureAuthRejected {
		t.Errorf("Classify(401, paused) class = %s, want %s", got, FailureAuthRejected)
	}
	if got := Classify(403, pausedOrgBody).Class; got != FailureAuthRejected {
		t.Errorf("Classify(403, paused) class = %s, want %s", got, FailureAuthRejected)
	}
	if got := Classify(402, pausedOrgBody).Class; got != FailureCreditExhausted {
		t.Errorf("Classify(402, paused) class = %s, want %s", got, FailureCreditExhausted)
	}
	if got := Classify(429, pausedOrgBody).Class; got != FailureRateLimited {
		t.Errorf("Classify(429, paused) class = %s, want %s; a limit is not an org wall", got, FailureRateLimited)
	}
}

// TestPausedOrgBodyDoesNotShadowRetryableRateLimits guards the same hazard the
// usage-limit entry guards: the phrase is kept org-scoped, so an ordinary
// rate-limit body that merely mentions pausing stays retryable.
func TestPausedOrgBodyDoesNotShadowRetryableRateLimits(t *testing.T) {
	bodies := []string{
		`{"error":{"message":"Rate limit reached for requests per min"}}`,
		`{"error":{"message":"429 rate_limit_exceeded; the upstream is paused for maintenance"}}`,
		`{"error":{"message":"Too many requests"}}`,
	}
	for _, body := range bodies {
		if got := Classify(429, body); got.Class != FailureRateLimited {
			t.Errorf("Classify(429, %q) class = %s, want %s", body, got.Class, FailureRateLimited)
		}
	}
}

// TestPausedOrgStatusMapEntryIsFailoverNotFatal documents the exact shape of the
// 405 change: the status table still NAMES a class for 405 (the "every status
// names a class" contract holds and Classify agrees with it), and that class is
// now failover rather than fatal.
func TestPausedOrgStatusMapEntryIsFailoverNotFatal(t *testing.T) {
	got, ok := standardStatusClasses[405]
	if !ok {
		t.Fatal("standardStatusClasses must still name 405")
	}
	if got != FailureUnavailable {
		t.Errorf("standardStatusClasses[405] = %s, want %s", got, FailureUnavailable)
	}
	if f := Classify(405, ""); f.Action != ActionFailoverNext {
		t.Errorf("Classify(405, bare) action = %s, want %s", f.Action, ActionFailoverNext)
	}
}

// TestPausedOrgViaClassifyError covers the relay path: a router that only ever
// sees the Go error wrapping the body must reach the same verdict, because
// errorSignatures shares the credit list.
func TestPausedOrgViaClassifyError(t *testing.T) {
	err := errors.New(`upstream 405: ` + pausedOrgBody)
	f := ClassifyError(err)
	if f.Class != FailureCreditExhausted {
		t.Fatalf("ClassifyError class = %s, want %s", f.Class, FailureCreditExhausted)
	}
	if f.Action != ActionFailoverNext {
		t.Errorf("ClassifyError action = %s, want %s", f.Action, ActionFailoverNext)
	}
}

// TestPausedOrg405ActuallyFailsOver is the INVOKABILITY witness for the severity
// change. A classifier whose verdict nothing consumes is presence, not behavior
// (docs/PRESENCE-VS-INVOKABILITY.md), so this drives the real Failover path: the
// request must land on the NEXT engine, never on ReasonFailureFatal.
//
// This is the whole operational point of routing through the router. Before the
// change, a paused provider 405 meant the router REFUSED the request outright,
// so a direct-path death became a routed-path death; now the ladder moves on.
func TestPausedOrg405ActuallyFailsOver(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T-405", "docs"), Role: "worker"}

	for _, tc := range []struct {
		name      string
		body      string
		wantClass FailureClass
	}{
		{"paused body", pausedOrgBody, FailureCreditExhausted},
		{"bare body (the observed case)", "", FailureUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := Classify(405, tc.body)
			if f.Class != tc.wantClass {
				t.Fatalf("class = %s, want %s", f.Class, tc.wantClass)
			}
			a, _, u, err := Failover(req, "primary", f, failoverEngines(), DefaultDegradationChain(), DefaultPolicy())
			if err != nil {
				t.Fatalf("Failover() error = %v", err)
			}
			if u.Reason == ReasonFailureFatal {
				t.Fatalf("a 405 ended the request as fatal: %+v; it must move to the next engine", u)
			}
			if a.EngineID != "fallback" {
				t.Errorf("assigned engine = %q, want %q", a.EngineID, "fallback")
			}
			if a.EngineID == "primary" {
				t.Error("the failed engine was re-assigned")
			}
			if !auditContains(a.AssignmentAudit, "FAILOVER_NEXT") {
				t.Errorf("audit does not name the failover action: %v", a.AssignmentAudit)
			}
		})
	}
}

// TestPausedOrgLeavesTheClosedVocabularyClosed pins the closed-vocabulary size
// so a class addition is always a deliberate, witnessed change: the paused-org
// fix reused an existing class (the 14th), and the certificate rung added the
// 15th.
func TestPausedOrgLeavesTheClosedVocabularyClosed(t *testing.T) {
	if len(FailureClasses()) != 15 {
		t.Errorf("FailureClasses() has %d members, want the closed vocabulary of 15", len(FailureClasses()))
	}
	if !FailureCreditExhausted.IsKnown() {
		t.Error("FailureCreditExhausted left the closed vocabulary")
	}
}
