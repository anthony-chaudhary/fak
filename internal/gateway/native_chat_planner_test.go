package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestSelectChatPlannerNativeChatPlannerSeam(t *testing.T) {
	native := agent.NewMockPlanner("inkling2")
	logf := func(string, ...any) {}
	got, side, _, _, err := selectChatPlanner(Config{NativeChatPlanner: native}, "inkling2", nil, logf, nil)
	if err != nil || got != agent.Planner(native) || side != localitySelfHosted {
		t.Fatalf("native seam: planner=%T side=%v err=%v", got, side, err)
	}
	proxied, _, _, _, err := selectChatPlanner(Config{NativeChatPlanner: native}, "m", []string{"http://127.0.0.1:1"}, logf, nil)
	if err != nil || proxied == agent.Planner(native) {
		t.Fatalf("a proxy upstream must win over the native seam: planner=%T err=%v", proxied, err)
	}
}
