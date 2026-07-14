package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestCompanionDefinitionsCommitted is the regression witness for #3862:
// committed callers in this package once referenced definitions that existed
// only in peer-untracked companion files, so a clean checkout of the trunk
// could not build even though every live tree compiled. Binding the companion
// definitions into the committed test graph makes that failure mode loud at
// the seam where it happened: if any of these definitions leaves the committed
// tree while a caller still needs it, this file stops compiling from a clean
// checkout instead of only failing in CI preflight.
func TestCompanionDefinitionsCommitted(t *testing.T) {
	// messages_transform.go and anthropic_elide_stale.go carry request-shaping
	// state; referencing them as method/function values is the compile-time
	// witness without invoking a served request.
	for name, ref := range map[string]any{
		"(*Server).readAnthropicMessagesRequest":  (*Server).readAnthropicMessagesRequest,
		"(*Server).prepareServedAnthropicRequest": (*Server).prepareServedAnthropicRequest,
		"repetitionLoopSteer":                     repetitionLoopSteer,
		"agent.ElideStaleReadsWithOutcome":        agent.ElideStaleReadsWithOutcome,
	} {
		if ref == nil {
			t.Fatalf("companion definition %s missing", name)
		}
	}

	// messages_resultnotes.go helpers have total, input-free zero cases; assert
	// their committed no-activity semantics directly.
	if got := fakExtFrom(nil, nil); got != nil {
		t.Fatalf("fakExtFrom(nil, nil) = %+v, want nil (fak key omitted on a turn with no tool activity)", got)
	}
	if anyRepaired(nil) {
		t.Fatal("anyRepaired(nil) = true, want false with no adjudications")
	}
	if anyLivelock(nil) {
		t.Fatal("anyLivelock(nil) = true, want false with no adjudications")
	}
}
