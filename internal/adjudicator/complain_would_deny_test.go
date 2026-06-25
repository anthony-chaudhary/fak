package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// TestComplainAdmitCarriesWouldDeny is the #671 anchor: a complain-mode admit must
// record would_deny = the suppressed rung's reason name (DEFAULT_DENY here), the
// forensic field the promotion ledger folds. The global read-shaped posture admit
// carries the same field identically.
func TestComplainAdmitCarriesWouldDeny(t *testing.T) {
	ctx := context.Background()

	complain := New(Policy{Complain: map[string]bool{"provision_widget": true}})
	v := complain.Adjudicate(ctx, inlineCall("provision_widget", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("complain admit: got %v, want Allow", v.Kind)
	}
	if v.Meta["posture"] != "admit_and_log" || v.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("complain admit record = %v, want posture=admit_and_log + would_deny=DEFAULT_DENY", v.Meta)
	}

	// Parity: the global read-shaped posture admit carries the same forensic fields.
	posture := New(Policy{Posture: PostureAdmitAndLog})
	rv := posture.Adjudicate(ctx, inlineCall("read_report", `{}`))
	if rv.Meta["posture"] != "admit_and_log" || rv.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("read-shaped admit record = %v, want posture + would_deny=DEFAULT_DENY", rv.Meta)
	}
}

// TestComplainKeepsHardRefusalsClosed is the #671 safety guarantee: complain mode
// admits ONLY the default-deny rung. A complain-set tool that trips a HARD refusal —
// an explicit Deny, or a write-shaped self-modify into a guarded tree — still fails
// closed, because those rungs return before defaultDeny is ever consulted.
func TestComplainKeepsHardRefusalsClosed(t *testing.T) {
	ctx := context.Background()
	a := New(Policy{
		Deny:            map[string]abi.ReasonCode{"exfiltrate": abi.ReasonSecretExfil},
		SelfModifyGlobs: []string{"internal/abi/"},
		// Put BOTH a hard-deny tool and a write-shaped tool in the complain set: neither
		// may be admitted-and-logged.
		Complain: map[string]bool{"exfiltrate": true, "write_kernel": true},
	})

	// Explicit deny stays closed.
	if v := a.Adjudicate(ctx, inlineCall("exfiltrate", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("explicit deny under complain: got %v/%s, want Deny/SECRET_EXFIL", v.Kind, abi.ReasonName(v.Reason))
	}
	// Self-modify (write-shaped target in a guarded tree) stays closed.
	if v := a.Adjudicate(ctx, inlineCall("write_kernel", `{"path":"internal/abi/x.go"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify under complain: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
	// The would_deny carried on an admit is never a hard reason: only DEFAULT_DENY is
	// ever suppressed by admit-and-log.
	if v := a.Adjudicate(ctx, inlineCall("benign_complain_tool", `{}`)); v.Kind == abi.VerdictAllow && v.Meta["would_deny"] != "DEFAULT_DENY" && v.Meta["would_deny"] != "" {
		t.Fatalf("admit would_deny must be DEFAULT_DENY, got %q", v.Meta["would_deny"])
	}
}
