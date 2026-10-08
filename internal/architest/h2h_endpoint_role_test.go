package architest

import "testing"

// TestH2HChatEndpointRemainsOffPath pins the boundary that earns this benchmark
// its endpoint declaration; it must not become a second live planner.
// fak-test:runtime fast est=1s lane=default
func TestH2HChatEndpointRemainsOffPath(t *testing.T) {
	t.Parallel()
	if role, ok := chatEndpointRole["h2hbench"]; !ok || role == "" {
		t.Error("h2hbench must declare its off-path benchmark endpoint role")
	}
	if requestPathClosure(t, internalDir(t))["h2hbench"] {
		t.Fatal("h2hbench must remain outside the live registrations closure; use agent.HTTPPlanner for live planning")
	}
}
