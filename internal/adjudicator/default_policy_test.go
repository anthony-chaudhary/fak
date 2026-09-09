package adjudicator

import (
	"context"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestDefaultPolicyProductionCapabilityFloor(t *testing.T) {
	p := DefaultPolicy()

	categories := map[string][]string{
		"standard coding tools": {
			"Bash", "BashOutput", "KillShell", "PowerShell", "Read", "Edit", "Write",
			"NotebookEdit", "Glob", "Grep", "LS", "TodoWrite", "Task", "WebFetch",
			"WebSearch", "ExitPlanMode", "Skill", "SlashCommand",
		},
		"agent orchestration": {
			"Agent", "AskUserQuestion", "DeferredToolPlaceholder", "EnterPlanMode",
			"Monitor", "ReportFindings", "ScheduleWakeup", "SendMessage",
			"StructuredOutput", "TaskCreate", "TaskGet", "TaskList", "TaskOutput",
			"TaskStop", "TaskUpdate", "ToolSearch", "shell_command", "exec_command",
			"update_plan", "write_stdin", "request_user_input", "view_image",
			"get_goal", "create_goal", "update_goal", "emit_context_tip",
			"list_mcp_resources", "read_mcp_resource", "parallel",
		},
		"workflows": {
			"Workflow", "EnterWorktree", "ExitWorktree", "CronCreate", "CronList",
			"CronDelete", "PushNotification", "RemoteTrigger", "DesignSync",
			"spawn_agent", "send_input", "wait", "wait_agent", "close_agent",
		},
		"DOS tools": {
			"dos_verify", "dos_arbitrate", "dos_recall", "dos_review", "dos_status",
			"dos_doctor", "dos_answer", "dos_check_reason", "dos_refuse_reasons",
			"dos_commit_audit", "dos_citation_resolve",
		},
		"FAK tools": {
			"fak_adjudicate", "fak_admit", "fak_syscall", "fak_read", "fak_changes",
			"fak_memory_drivers", "fak_memory_explain", "fak_memory_run", "fak_grep", "fak_glob",
		},
		"safe inspection / build / test / safe sinks": {
			"git_status", "git_diff", "git_log", "go_build", "go_test", "run_tests",
			"ship_release", "transfer_to_human_agents",
		},
		"preserve test/benchmark compatibility": {
			"book_reservation", "update_reservation_flights", "send_certificate",
		},
	}

	for cat, tools := range categories {
		for _, tool := range tools {
			if !p.Allow[tool] {
				t.Errorf("category %q: tool %q missing from DefaultPolicy.Allow", cat, tool)
			}
		}
	}

	wantPrefixes := []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"}
	if !slices.Equal(p.AllowPrefix, wantPrefixes) {
		t.Errorf("DefaultPolicy.AllowPrefix = %v, want %v", p.AllowPrefix, wantPrefixes)
	}

	if p.Deny["shell_rm_rf"] != abi.ReasonPolicyBlock {
		t.Errorf("DefaultPolicy.Deny[shell_rm_rf] = %v, want %v", p.Deny["shell_rm_rf"], abi.ReasonPolicyBlock)
	}
	if p.Deny["exfiltrate"] != abi.ReasonSecretExfil {
		t.Errorf("DefaultPolicy.Deny[exfiltrate] = %v, want %v", p.Deny["exfiltrate"], abi.ReasonSecretExfil)
	}

	wantGlobs := []string{
		"internal/abi/", "internal/kernel/", "internal/adjudicator/", "internal/architest/",
		"internal/shipgate/", "dos.toml", ".dos/", "fak/internal/",
	}
	if !slices.Equal(p.SelfModifyGlobs, wantGlobs) {
		t.Errorf("DefaultPolicy.SelfModifyGlobs = %v, want %v", p.SelfModifyGlobs, wantGlobs)
	}

	wantRedact := []string{"password", "secret", "api_key", "token", "authorization"}
	if !slices.Equal(p.RedactFields, wantRedact) {
		t.Errorf("DefaultPolicy.RedactFields = %v, want %v", p.RedactFields, wantRedact)
	}
}

func TestDefaultPolicyAdjudicationVerdicts(t *testing.T) {
	a := New(DefaultPolicy())
	ctx := context.Background()

	// Standard coding and agent tools allowed for benign arguments
	allowedTools := []string{
		"Bash", "Read", "Write", "Edit", "Glob", "Grep", "dos_verify", "dos_arbitrate",
		"fak_adjudicate", "fak_syscall", "git_status", "go_test",
	}
	for _, tool := range allowedTools {
		v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("tool %s: got verdict %v/%s, want VerdictAllow", tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// Unknown tools denied fail-closed
	vUnknown := a.Adjudicate(ctx, inlineCall("totally_unknown_custom_tool", `{}`))
	if vUnknown.Kind != abi.VerdictDeny || vUnknown.Reason != abi.ReasonDefaultDeny {
		t.Errorf("unknown tool: got %v/%s, want Deny/DEFAULT_DENY", vUnknown.Kind, abi.ReasonName(vUnknown.Reason))
	}

	// Explicit deny rules honored
	vDeny := a.Adjudicate(ctx, inlineCall("shell_rm_rf", `{}`))
	if vDeny.Kind != abi.VerdictDeny || vDeny.Reason != abi.ReasonPolicyBlock {
		t.Errorf("shell_rm_rf: got %v/%s, want Deny/POLICY_BLOCK", vDeny.Kind, abi.ReasonName(vDeny.Reason))
	}

	// Self-modify guarded trees denied
	vSelfMod := a.Adjudicate(ctx, inlineCall("Write", `{"file_path":"internal/abi/types.go","content":"package abi"}`))
	if vSelfMod.Kind != abi.VerdictDeny || vSelfMod.Reason != abi.ReasonSelfModify {
		t.Errorf("write to internal/abi/: got %v/%s, want Deny/SELF_MODIFY", vSelfMod.Kind, abi.ReasonName(vSelfMod.Reason))
	}

	// Secret field redaction triggers transform
	vRedact := a.Adjudicate(ctx, inlineCall("book_reservation", `{"password":"secret123"}`))
	if vRedact.Kind != abi.VerdictTransform {
		t.Errorf("redact secret arg: got %v/%s, want VerdictTransform", vRedact.Kind, abi.ReasonName(vRedact.Reason))
	}
}
