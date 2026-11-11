package main

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// standardHarnessSurfaces are the in-scope standard-harness tool names the
// 2026-11-07 transcript census found DEFAULT_DENY'd by the embedded guard floor
// (cmd/fak/guard-default-policy.json). Each name was observed emitted by a
// first-class harness; the trailing comment notes which one. These belong on
// the public floor: native Claude Code tools, pi's native subagent tool, and
// fak-owned MCP server spellings.
var standardHarnessSurfaces = []string{
	"ReadNotifications",                         // claude
	"ListAgents",                                // claude
	"SendUserFile",                              // claude
	"SendFeedback",                              // claude
	"subagent",                                  // pi
	"mcp__fak-mcp__dos_post",                    // fak-mcp
	"mcp__fak-mcp__dos_read",                    // fak-mcp
	"mcp__fak-mcp__fak_ticket_create",           // fak-mcp
	"mcp__fak-mcp__fak_ops_api",                 // fak-mcp
	"mcp__plugin_dos-kernel_dos__dos_arbitrate", // dos-plugin
	"mcp__plugin_dos-kernel_dos__dos_doctor",    // dos-plugin
}

// pinnedOffFloorNames are census names the repo deliberately keeps OFF the
// public floor. Artifact publishes session output to claude.ai: issue #2255
// decided it stays off-floor unless an operator opts in with a custom policy,
// and that verdict is pinned twice — the Claude Code profile's DeniedTools
// entry in guard_harness_profiles_test.go and
// TestHarnessProfileCoverageDetectsForbiddenArtifactTool. Widening the floor
// past this pin is a coordinator/operator decision, not a census follow-through.
var pinnedOffFloorNames = []string{"Artifact"}

// personalMCPNames are the 42 operator-personal MCP tool names the same census
// found DEFAULT_DENY'd. They are NOT public-floor material: each belongs to a
// host-local operator allow overlay (`fak guard allow --prefix`), so they are
// asserted only by TestGuardHostOverlayProbe, never against the bare floor.
var personalMCPNames = []string{
	"mcp__ccd_turn__update_plan",
	"mcp__ccd_turn__wrap_up",
	"mcp__ccd_session_mgmt__list_events",
	"mcp__ccd_session_mgmt__send_message",
	"mcp__ccd_session_mgmt__search_session_transcripts",
	"mcp__ccd_session_mgmt__list_sessions",
	"mcp__ccd_session_mgmt__get_session",
	"mcp__ccd_session_mgmt__get_usage",
	"mcp__ccd_session_mgmt__export_transcript",
	"mcp__ccd_session_mgmt__stop_session",
	"mcp__ccd_session__spawn_task",
	"mcp__ccd_session__start_session",
	"mcp__ccd_session__dismiss_task",
	"mcp__ccd_session__mark_chapter",
	"mcp__ccd_session__list_start_targets",
	"mcp__ccd_host__request_keep_awake",
	"mcp__ccd_host__sync_with_base_branch",
	"mcp__ccd_directory__change_directory",
	"mcp__ccd_view__show_pane",
	"mcp__ccd_settings__get_settings",
	"mcp__hearthbot__update_status",
	"mcp__hearthbot__no_reply_needed",
	"mcp__hearthbot__reply",
	"mcp__hearthbot__read_project_file",
	"mcp__hearthbot__react",
	"mcp__hearthbot__list_project_files",
	"mcp__hearthbot__unreact",
	"mcp__hearthbot__update_message",
	"mcp__hearthbot__list_thread_sessions",
	"mcp__hearthbot__fetch_thread",
	"mcp__hearthbot__send_message",
	"mcp__claude-code-remote__send_message",
	"mcp__scheduled-tasks__create_scheduled_task",
	"mcp__scheduled-tasks__run_scheduled_task",
	"mcp__terminal__read_terminal",
	"mcp__terminal__run_in_terminal",
	"mcp__Claude_Browser__computer",
	"mcp__Claude_Browser__navigate",
	"mcp__Claude_Browser__javascript_tool",
	"mcp__Claude_Browser__get_page_text",
	"mcp__Claude_Browser__preview_start",
	"mcp__1a59c906-04da-521d-bda7-7f71b9f9e01c__read",
}

// allObservedDeniedNames is the full 54-name census set: the 11 in-scope
// public-floor names, the 1 pinned-off-floor name, and the 42 personal-MCP
// names.
var allObservedDeniedNames = append(append(append([]string(nil), standardHarnessSurfaces...), pinnedOffFloorNames...), personalMCPNames...)

// fak-test:runtime fast est=1s
func TestGuardFloorAdmitsStandardHarnessSurfaces(t *testing.T) {
	isolateGuardFloorOverlays(t)
	ctx := context.Background()
	for _, posture := range []string{"", "fail_closed"} {
		rt, _, _, _ := loadGuardCapabilityFloor("", posture)
		adj := adjudicator.New(rt.Adjudicator)
		for _, tool := range standardHarnessSurfaces {
			v := adj.Adjudicate(ctx, guardToolCall(t, tool, map[string]any{}))
			if v.Kind != abi.VerdictAllow {
				t.Errorf("posture %q: %s = %v/%s, want ALLOW", posture, tool, v.Kind, abi.ReasonName(v.Reason))
			}
		}
		// Danger spot-checks: the allow-list widening must not reopen any
		// genuine-danger class. The triggers below are the policy's actual
		// arg-rules in guard-default-policy.json: rm -rf (Bash), recursive/forced
		// Remove-Item (PowerShell — the floor ships no sudo rule on PowerShell, so
		// this is the closest existing danger-rule trigger), and the fork-bomb
		// shape (Bash).
		dangers := []struct {
			tool string
			args map[string]any
		}{
			{"Bash", map[string]any{"command": "rm -rf /"}},
			{"PowerShell", map[string]any{"command": `Remove-Item -Recurse -Force C:\work`}},
			{"Bash", map[string]any{"command": ":(){ :|:& };:"}},
		}
		for _, d := range dangers {
			v := adj.Adjudicate(ctx, guardToolCall(t, d.tool, d.args))
			if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
				t.Errorf("posture %q: %s danger call = %v/%s, want DENY/POLICY_BLOCK", posture, d.tool, v.Kind, abi.ReasonName(v.Reason))
			}
		}
		// Boundary spot-check: the census's twelfth name stays closed. #2255
		// keeps Artifact off the public floor; since the default-open flip it is
		// pinned by an EXPLICIT deny entry in guard-default-policy.json, which
		// answers POLICY_BLOCK under every posture — stronger than the old
		// DEFAULT_DENY pin that only bit under fail_closed.
		for _, tool := range pinnedOffFloorNames {
			v := adj.Adjudicate(ctx, guardToolCall(t, tool, map[string]any{}))
			if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
				t.Errorf("posture %q: %s = %v/%s, want DENY/POLICY_BLOCK (pinned off-floor, #2255)", posture, tool, v.Kind, abi.ReasonName(v.Reason))
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestGuardHostOverlayProbe(t *testing.T) {
	if os.Getenv("FAK_GUARD_HOST_PROBE") != "1" {
		t.Skip("set FAK_GUARD_HOST_PROBE=1 to probe host-local operator allow overlays against the 54 observed-denied names (CI must not depend on host-local config)")
	}
	ov, layers, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatalf("loadGuardAllowOverlayLayers: %v", err)
	}
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("ParseRuntime(guardDefaultPolicyJSON): %v", err)
	}
	added := guardApplyAllowOverlay(&rt, ov)
	// Fail-closed posture: under default_open the default-deny rung returns Allow
	// for every name that passes the danger rungs, so the probe would pass with
	// zero overlays and witness nothing. The census's DEFAULT_DENY observations
	// come from the strict side, where only floor ∪ overlay admits — which is
	// exactly what this probe must verify.
	rt.Adjudicator.Posture = adjudicator.PostureFailClosed
	rt.PolicyContext.Posture = adjudicator.PostureFailClosed
	adj := adjudicator.New(rt.Adjudicator)
	ctx := context.Background()
	pinned := map[string]bool{}
	for _, tool := range pinnedOffFloorNames {
		pinned[tool] = true
	}
	var stillDenied []string
	for _, tool := range allObservedDeniedNames {
		v := adj.Adjudicate(ctx, guardToolCall(t, tool, map[string]any{}))
		if pinned[tool] {
			// Off by design: #2255 keeps these off the floor and out of the
			// default overlay path; an explicit operator opt-in would admit them.
			if v.Kind != abi.VerdictDeny {
				t.Errorf("%s admitted although pinned off-floor (#2255) — got %v/%s, want DENY", tool, v.Kind, abi.ReasonName(v.Reason))
			}
			continue
		}
		if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonDefaultDeny {
			stillDenied = append(stillDenied, tool)
		}
	}
	probeSet := len(allObservedDeniedNames) - len(pinnedOffFloorNames)
	t.Logf("host probe: %d/%d overlay-eligible observed-denied names admitted (overlay layers=%d, new entries=%d; %d pinned off-floor)",
		probeSet-len(stillDenied), probeSet, len(layers), added, len(pinnedOffFloorNames))
	for _, tool := range stillDenied {
		t.Errorf("%s still DEFAULT_DENY with host overlays applied", tool)
	}
}
