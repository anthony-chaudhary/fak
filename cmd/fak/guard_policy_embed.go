package main

import (
	_ "embed"
	"time"
)

const guardResourceSampleInterval = 2 * time.Second

// guardDefaultPolicyJSON is the day-to-day capability floor `fak guard` enforces when
// the operator names no --policy. It is embedded in the binary so `fak guard` works
// from ANY directory (a repo or not, an installed binary with no source tree). It
// allows the standard coding-agent tool set and denies the genuine-danger classes:
// destructive removal, privilege escalation, disk wipe, fork bomb, RCE pipe,
// `../`-relative tree-escape redirects, and writes into credential/SSH/secret paths.
// This is a DENYLIST of named-dangerous patterns (`self_modify_globs` matched by
// substring + a handful of `../`-relative Bash arg-rules), NOT a working-tree
// boundary: an absolute-path write outside the repo that matches no guarded glob and
// is not spelled `../` is still ALLOWED — the floor does not confine a wrapped agent
// to the repo. Making that boundary real is the optional hardening follow-on in #1466.
// Print or fork it with `fak guard --dump-policy`.
//
// The allow-list also admits the host harness's ORCHESTRATION + deferred-tool-loading +
// read-only-MCP surface (Agent/Task*/SendMessage/Monitor/ScheduleWakeup/EnterPlanMode/
// AskUserQuestion/ToolSearch/Read*McpResource*). It also admits the Codex-native
// names for the same host plumbing (update_plan, request_user_input, get/update_goal,
// list/read MCP resources, tool_search_tool, hosted web/image built-ins, and
// shell_command with the same danger arg-rules as Bash/PowerShell), including the
// namespace-qualified spellings some Codex surfaces expose (`functions.shell_command`,
// `tool_search.tool_search_tool`, `multi_tool_use.parallel`, `web.run`,
// `image_gen.imagegen`). That keeps `fak guard -- codex` from DEFAULT_DENYing the
// harness's own planning seam before the agent can do real work. These are NOT
// capability grants: a
// subagent the floor lets the agent SPAWN makes its own tool calls back through this same
// gateway, so every real effect is re-adjudicated downstream — the floor is unchanged, the
// agent just keeps its task/subagent/plan plumbing. ToolSearch in particular is load-bearing
// on harnesses that defer tool schemas: deny it and the agent cannot even reach WebFetch /
// WebSearch / MCP tools, so a bare floor silently bricks the wrapped agent. This was the
// dominant friction a historical-session replay surfaced ("align_policy_with_real_tool_shapes"
// across every audited Claude Code session: the floor was DEFAULT_DENYing Task*/SendMessage/
// ToolSearch — the harness's own tools). The genuine-danger classes above are untouched.
//
// The allow-list also admits the broader ULTRACODE orchestration surface (Workflow,
// EnterWorktree/ExitWorktree, Cron*, PushNotification, RemoteTrigger, DesignSync) and the
// READ-ONLY DOS verbs (mcp__dos__dos_verify/arbitrate/recall/review/status/doctor/answer/
// check_reason/refuse_reasons/commit_audit/citation_resolve, acme_lane_hint). The
// work-spawners (Workflow, EnterWorktree, Cron*) carry the same re-adjudication property as
// Agent/Task*: the agents and future prompts they create make their own tool calls back
// through this gateway, so every real effect still crosses the floor. The DOS verbs are pure
// reads of git/journal state. This means a turn that advertises the full ultracode toolset is
// never left with those names as silent prune-candidates — and matches the operator posture
// that an ultracode/debug session is governed by re-adjudication of EFFECTS, not by withholding
// orchestration plumbing. The genuine-danger classes (destructive Bash args, self-modify globs)
// are still untouched, so widening the orchestration surface never widens the danger floor.
//
// The allow-list also admits fak's OWN self-service MCP verbs (fak_adjudicate/fak_admit/
// fak_syscall/fak_read/fak_changes/fak_memory_drivers/fak_memory_explain/fak_trajquery) —
// real guarded sessions were DEFAULT_DENYing the guard's own appeal channel (fak_admit),
// which is self-defeating: these are pure reads or kernel-re-adjudicated executions, so
// admitting the NAME grants nothing the floor doesn't re-check downstream. fak_memory_run
// stays OUT: with apply=true it is an effectful memory write (the pinned deny in
// guard_test.go); an operator overlay is the sanctioned place to grant it.
// Likewise the harness's ReportFindings (code-review output) and DeferredToolPlaceholder
// (deferred-schema plumbing), both witnessed as DEFAULT_DENY friction in session journals.
// The Confluence DELETE verbs are explicitly denied so a coarse operator overlay grant
// (`fak guard allow --prefix mcp__atlassian__`) never admits irreversible external
// destruction — a name-deny outranks every allow layer.
//
// The `\bsudo\b` arg-rule is decided STRUCTURALLY (internal/adjudicator/sudo_local.go,
// same recogniser pattern as rm_rf/rce_pipe): only a LOCAL escalation at a resolved
// command-word position is refused. `ssh gpu-box 'sudo systemctl …'` — the dominant
// false POLICY_BLOCK in real remote-GPU bring-up trajectories — is allowed, because the
// floor governs this host, and the remote host's own controls govern that one.
//
//go:embed guard-default-policy.json
var guardDefaultPolicyJSON []byte

// cmdGuard — run any agent command with the kernel adjudicating every tool call it
// proposes. This is the one-command, cross-platform, productized form of the dogfood
// path: it starts the SAME gateway `fak serve` runs (in-process, on a private loopback
// port), points the child agent's provider base URL at it via a child-ONLY env var
// (never the parent shell, never a config file), execs the agent interactively, and on
// exit prints what the kernel allowed vs blocked before tearing the gateway down.
//
// The default upstream is the real Anthropic API in passthrough mode, so
// `fak guard -- claude` wraps your normal Claude Code: your credential and prompt-cache
// breakpoints flow through untouched (the gateway forwards the request bytes verbatim),
// but every tool call Claude proposes crosses the capability floor first. For
// --provider anthropic, fak uses your Claude Pro/Max SUBSCRIPTION by DEFAULT — it
// sources the OAuth token and sends it upstream as a bearer token. This holds even when
// ANTHROPIC_API_KEY is exported (a global SDK key no longer silently switches you to API
// billing); pass --api-key-env ANTHROPIC_API_KEY to opt INTO API billing, or
// --anthropic-oauth to force the subscription path and fail loud if no token is found.
//
// It also turns the durable DECISION JOURNAL on by default: every verdict the kernel
// reaches this session is appended to a tamper-evident, hash-chained file you can
// later replay with `fak audit verify`. fak is the disinterested referee, and the
// journal is the verifiable record of what it allowed vs blocked — a self-report is
// not a witness. Point it with --audit PATH, or turn it off with --no-audit.
