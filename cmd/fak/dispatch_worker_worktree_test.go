package main

import "testing"

// TestWorkerWorktreeEnabledGrammar pins the #3168 opt-in switch: unset and every
// off-ish value are OFF (shared-trunk spawn, today's behavior byte-for-byte); any
// other value is ON. Mirrors guardDisabled's truthy/falsy grammar, inverted.
func TestWorkerWorktreeEnabledGrammar(t *testing.T) {
	off := []string{"", "0", "off", "OFF", "false", "False", "no", "disable", "disabled", "  off  "}
	on := []string{"1", "on", "true", "yes", "enable", "enabled", "worktree"}

	// Unset -> off.
	t.Run("unset", func(t *testing.T) {
		if workerWorktreeEnabled() {
			t.Fatal("unset FLEET_WORKER_WORKTREE must be OFF")
		}
	})
	for _, v := range off {
		v := v
		t.Run("off/"+v, func(t *testing.T) {
			t.Setenv("FLEET_WORKER_WORKTREE", v)
			if workerWorktreeEnabled() {
				t.Fatalf("value %q must be OFF", v)
			}
		})
	}
	for _, v := range on {
		v := v
		t.Run("on/"+v, func(t *testing.T) {
			t.Setenv("FLEET_WORKER_WORKTREE", v)
			if !workerWorktreeEnabled() {
				t.Fatalf("value %q must be ON", v)
			}
		})
	}
}
