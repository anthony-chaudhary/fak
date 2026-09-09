package adjudicator

import "github.com/anthony-chaudhary/fak/internal/abi"

// interpreterEvalSpec pairs a general-purpose interpreter with the inline-program flags
// whose presence as a TOKEN means it runs code from an opaque string argument able to
// write a file directly.
type InlineEvalSpec struct {
	Interp string
	Flags  []string
}

type interpreterEvalSpec struct {
	interp string   // the interpreter named as a command word (`ruby`, `node`, …)
	flags  []string // its inline-eval flags (`-e`, `--eval`, `-c`, `-p`/`--print`)
}

// interpreterEvalFlags is the adjudicator's inline-eval write floor (#172 Hole 1
// residual): the general-purpose interpreters most likely on a coding agent's PATH —
// python, node, and ruby — each paired with the inline-program flags that run an opaque
// program string. It is the interpreter analogue of shellWriteVerbs; commandWrites ranges
// over it and treats such a command as write-shaped, routing it through the SAME
// commandSelfModify guard the shell floor uses.
//
// ruby joins python/node to close an asymmetry the rulesynth RSI loop (internal/rsiloop,
// run with -harness rulesynth) mined from the near-miss corpus and the keep-bit KEPT:
// `ruby -i` (an in-place edit) is already caught by shellWriteVerbs, but `ruby -e
// 'File.write("internal/adjudicator/decide.go", …)'` — the EVAL flag doing the same
// self-edit — slipped every rung. `perl -e`, `php -r`, and `lua -e` are the same shape and
// remain the residual queue the loop's corpus drives next, so they are deliberately not
// listed here yet.
//
// Detection is by TOKEN, not a fixed `<interp> <flag> ` prefix (see interpreterEvalMatch):
// the interpreter need only appear as a word and the flag as its own argument, so the
// idiomatic no-space, quoted, and `=`-joined spellings (`ruby -e'…'`, `node --eval=…`) and
// intervening flags (`ruby -rjson -e …`) are all caught — closing the porous-prefix gap a
// fixed-spelling table leaves open. The identifier name `interpreterEvalFlags` is pinned by
// architest (TestInlineEvalFloorWiredInCommandWrites); rename only with that gate's constant.
var interpreterEvalFlags = []interpreterEvalSpec{
	{"python3", []string{"-c"}},
	{"python", []string{"-c"}},
	{"node", []string{"-e", "--eval", "-p", "--print"}},
	{"ruby", []string{"-e", "--eval"}},
}

// DefaultPolicy returns the production capability floor: allowing standard
// agent tools, coding tools, read-only utilities, and safe execution tools,
// denying dangerous self-modify globs and destructive actions, and redacting
// common secret arg fields.
func DefaultPolicy() Policy {
	return Policy{
		Allow: map[string]bool{
			// Standard coding tools
			"Bash": true, "BashOutput": true, "KillShell": true, "PowerShell": true,
			"Read": true, "Edit": true, "Write": true, "NotebookEdit": true,
			"Glob": true, "Grep": true, "LS": true, "TodoWrite": true,
			"Task": true, "WebFetch": true, "WebSearch": true,
			"ExitPlanMode": true, "Skill": true, "SlashCommand": true,

			// Agent orchestration
			"Agent": true, "AskUserQuestion": true, "DeferredToolPlaceholder": true,
			"EnterPlanMode": true, "Monitor": true, "ReportFindings": true,
			"ScheduleWakeup": true, "SendMessage": true, "StructuredOutput": true,
			"TaskCreate": true, "TaskGet": true, "TaskList": true,
			"TaskOutput": true, "TaskStop": true, "TaskUpdate": true,
			"ToolSearch": true, "shell_command": true, "exec_command": true,
			"update_plan": true, "write_stdin": true, "request_user_input": true,
			"view_image": true, "get_goal": true, "create_goal": true,
			"update_goal": true, "emit_context_tip": true,
			"list_mcp_resources": true, "read_mcp_resource": true, "parallel": true,

			// Workflows
			"Workflow": true, "EnterWorktree": true, "ExitWorktree": true,
			"CronCreate": true, "CronList": true, "CronDelete": true,
			"PushNotification": true, "RemoteTrigger": true, "DesignSync": true,
			"spawn_agent": true, "send_input": true, "wait": true,
			"wait_agent": true, "close_agent": true,

			// DOS tools
			"dos_verify": true, "dos_arbitrate": true, "dos_recall": true,
			"dos_review": true, "dos_status": true, "dos_doctor": true,
			"dos_answer": true, "dos_check_reason": true, "dos_refuse_reasons": true,
			"dos_commit_audit": true, "dos_citation_resolve": true,

			// FAK tools
			"fak_adjudicate": true, "fak_admit": true, "fak_syscall": true,
			"fak_read": true, "fak_changes": true, "fak_memory_drivers": true,
			"fak_memory_explain": true, "fak_memory_run": true,
			"fak_grep": true, "fak_glob": true,

			// Safe inspection / build / test / safe sinks
			"git_status": true, "git_diff": true, "git_log": true,
			"go_build": true, "go_test": true, "run_tests": true,
			"ship_release": true, "transfer_to_human_agents": true,

			// Preserve test/benchmark compatibility
			"book_reservation": true, "update_reservation_flights": true,
			"send_certificate": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"shell_rm_rf": abi.ReasonPolicyBlock,
			"exfiltrate":  abi.ReasonSecretExfil,
		},
		// Witness machinery (#172 Hole 2): internal/architest + internal/shipgate
		// are covered by the fak/internal/ catch-all, but dos.toml (the lane
		// taxonomy + stamp grammar the referee binds to) is not — name it so a
		// self-improving loop cannot quietly rewrite the gate that witnesses it.
		SelfModifyGlobs: []string{"internal/abi/", "internal/kernel/", "internal/adjudicator/", "internal/architest/", "internal/shipgate/", "dos.toml", ".dos/", "fak/internal/"},
		RedactFields:    []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// DevAgentPolicy is the deployable floor for a CODING agent that runs against this
// repo (the dogfood case — fak passing its own CICD test). It differs from the
// permissive bench DefaultPolicy in three load-bearing ways:
//
//   - it DENIES the shared-history git mutations (push / merge / tag): a dev agent
//     adapts code, it never moves the branch on its own say-so;
//   - it bounds writes AWAY from the spine — a write whose target touches the ABI,
//     the kernel, the adjudicator, the policy loader, the defconfig, or the .git
//     dir is a SELF_MODIFY (ESCALATE), not a silent edit;
//   - it ALLOWS a single high-level `ship_release` action that is NOT free: the
//     shipgate adjudicator holds it behind the require-witness rung, so it proceeds
//     only when a claimed ship is corroborated by git evidence the agent did not
//     author (an unwitnessed ship is refused).
//
// Mirrored on disk as examples/dev-agent-policy.json (round-trips through the
// policy manifest loader), so an adopter selects it with `--policy` — no fork.
func DevAgentPolicy() Policy {
	return Policy{
		Posture: PostureDefaultOpen,
		Allow: map[string]bool{
			// safe inspect / build / test tools a coding agent drives
			"Read":       true,
			"fak_read":   true,
			"git_status": true, "git_diff": true, "git_log": true,
			"go_build": true, "go_test": true, "run_tests": true,
			// the high-level ship action: allowed at the floor, but witness-gated by
			// the shipgate adjudicator (a claimed ship must be git-corroborated).
			"ship_release": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"git_push":   abi.ReasonPolicyBlock,
			"git_merge":  abi.ReasonPolicyBlock,
			"git_tag":    abi.ReasonPolicyBlock,
			"exfiltrate": abi.ReasonSecretExfil,
		},
		SelfModifyGlobs: []string{
			"internal/abi/", "internal/kernel/", "internal/adjudicator/",
			"internal/policy/", "internal/registrations/",
			// Witness machinery (#172 Hole 2): the trees that, if self-edited,
			// would let an RSI loop grade its own homework — the architest gates,
			// the shipgate require-witness rung, and the workspace's dos.toml
			// (lane taxonomy + stamp grammar the referee binds to).
			"internal/architest/", "internal/shipgate/", "dos.toml",
			".git/", ".dos/", "policy.json", "VERSION", "id_rsa",
		},
		RedactFields: []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// Default is the registered instance.
var Default = New(DefaultPolicy())
