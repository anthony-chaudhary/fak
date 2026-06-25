package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// TestComplainSetAdmitsNonReadShapedDefaultDeny is the #670 anchor: a tool in the
// per-tool complain set has its DEFAULT_DENY downgraded to an admit-and-log Allow
// EVEN WHEN it is not read-shaped — the global PostureAdmitAndLog path only admits
// read-shaped names.
func TestComplainSetAdmitsNonReadShapedDefaultDeny(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Complain: map[string]bool{"provision_widget": true},
	})

	// "provision_widget" is NOT read-shaped (no read_/get_/… prefix) and not allowed,
	// so it would default-deny — but it is in the complain set, so it admits-and-logs.
	v := a.Adjudicate(ctx, inlineCall("provision_widget", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("complain-set tool: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "admit_and_log" {
		t.Fatalf("complain admit must carry posture=admit_and_log, got %v", v.Meta)
	}
	if v.Reason != abi.ReasonNone {
		t.Fatalf("an admitted call must not carry a refusal reason, got %s", abi.ReasonName(v.Reason))
	}

	// A different non-read tool NOT in the complain set still fails closed.
	if v := a.Adjudicate(ctx, inlineCall("provision_other", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("non-complain tool: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestComplainEmptySetByteIdenticalToHead pins the drop-in guarantee: a Policy with a
// nil/empty Complain set behaves exactly as before — read-shaped admit still requires
// PostureAdmitAndLog, and everything else default-denies.
func TestComplainEmptySetByteIdenticalToHead(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{}) // zero policy: nil Complain
	if a.policy.complainFor("anything") {
		t.Fatal("nil Complain set must admit nothing")
	}
	// Fail-closed default deny for an unknown tool (HEAD behavior).
	if v := a.Adjudicate(ctx, inlineCall("provision_widget", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("zero policy: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestComplainDoesNotDowngradeHardRefusal confirms complain mode only touches the
// default-deny rung: an EXPLICIT Deny on a complain-set tool still fails closed (the
// deny rung returns before defaultDeny ever runs).
func TestComplainDoesNotDowngradeHardRefusal(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Deny:     map[string]abi.ReasonCode{"exfiltrate": abi.ReasonSecretExfil},
		Complain: map[string]bool{"exfiltrate": true},
	})
	v := a.Adjudicate(ctx, inlineCall("exfiltrate", `{}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("explicit deny on complain-set tool: got %v/%s, want Deny/SECRET_EXFIL (hard refusal stays closed)",
			v.Kind, abi.ReasonName(v.Reason))
	}
}
