package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestToolsPosture(t *testing.T) {
	t.Cleanup(func() {
		SetConfiguredPosture(adjudicator.PostureDefaultOpen)
		Configure()
	})

	// 1. PostureDefaultOpen is the default posture.
	if got := ConfiguredPosture(); got != adjudicator.PostureDefaultOpen {
		t.Fatalf("ConfiguredPosture() = %v, want PostureDefaultOpen (%v)", got, adjudicator.PostureDefaultOpen)
	}

	Configure()
	ctx := context.Background()

	// 2. Unlisted tools are ALLOWED by default with meta["posture"]="default_open".
	v := adjudicator.Default.Adjudicate(ctx, guardedCall("unlisted_custom_tool", `{"action":"test"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("unlisted tool: got Kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "default_open" {
		t.Fatalf("unlisted tool: got Meta[posture]=%q, want %q", v.Meta["posture"], "default_open")
	}

	// 3. Explicit deny (delete_account) is strictly DENIED.
	v = adjudicator.Default.Adjudicate(ctx, guardedCall("delete_account", `{"user_id":"mia"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("delete_account: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 4. Self-modify writes (direct write into internal/abi/) are strictly DENIED.
	v = adjudicator.Default.Adjudicate(ctx, guardedCall("write_file", `{"path":"internal/abi/types.go"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify write_file: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}

	// 5. Self-modify writes via shell (e.g. into internal/abi/) are strictly DENIED.
	v = adjudicator.Default.Adjudicate(ctx, guardedCall("Bash", `{"command":"sed -i 's/a/b/' internal/abi/types.go"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify Bash: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}

	// 6. Dangerous gotchas (e.g. rm -rf /) remain strictly DENIED.
	v = adjudicator.Default.Adjudicate(ctx, guardedCall("Bash", `{"command":"rm -rf /"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("dangerous gotcha: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 7. Switching to PostureFailClosed denies unlisted tools.
	SetConfiguredPosture(adjudicator.PostureFailClosed)
	if got := ConfiguredPosture(); got != adjudicator.PostureFailClosed {
		t.Fatalf("ConfiguredPosture() after switch = %v, want PostureFailClosed", got)
	}
	Configure()

	v = adjudicator.Default.Adjudicate(ctx, guardedCall("unlisted_custom_tool", `{"action":"test"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unlisted tool under PostureFailClosed: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}
