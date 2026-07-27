package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// TestGuardSessionStartEmitsAffordance asserts the #3092 affordance: the SessionStart hook
// emits a valid additionalContext envelope naming the fak entry verbs.
func TestGuardSessionStartEmitsAffordance(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
	}

	// Valid JSON with the exact Claude Code SessionStart envelope shape.
	var env struct {
		HookSpecificOutput struct {
			HookEventName     string `json:"hookEventName"`
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("stdout is not valid JSON envelope: %v\n%s", err, out.String())
	}
	if env.HookSpecificOutput.HookEventName != "SessionStart" {
		t.Fatalf("hookEventName = %q, want SessionStart", env.HookSpecificOutput.HookEventName)
	}
	ctx := env.HookSpecificOutput.AdditionalContext
	for _, verb := range []string{"fak_index_work", "fak_admit", "fak_tools_search"} {
		if !strings.Contains(ctx, verb) {
			t.Fatalf("affordance did not name entry verb %q: %s", verb, ctx)
		}
	}
}

// TestGuardSessionStartIdentity is the #4113 acceptance gate: a SessionStart hook holding both
// ids — the guard trace via --trace and the transcript UUID via CLAUDE_CODE_SESSION_ID — appends
// exactly one uuid<->trace join row to resume_identity.jsonl, and FoldIdentity (via LoadIdentity)
// resolves the UUID to the trace and back. This is the A1 store's first producer (#4112).
func TestGuardSessionStartIdentity(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	const uuid = "11111111-2222-3333-4444-555555555555"
	const trace = "trace-abc"
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", trace}); code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
	}

	traceByUUID, uuidByTrace := resume.LoadIdentity(regDir)
	if got := traceByUUID[uuid]; got != trace {
		t.Fatalf("traceByUUID[%q] = %q, want %q", uuid, got, trace)
	}
	if got := uuidByTrace[trace]; got != uuid {
		t.Fatalf("uuidByTrace[%q] = %q, want %q", trace, got, uuid)
	}

	// Exactly one row on a fresh start.
	raw, err := os.ReadFile(resume.IdentityLedgerPath(regDir))
	if err != nil {
		t.Fatalf("read identity store: %v", err)
	}
	if n := len(strings.Split(strings.TrimSpace(string(raw)), "\n")); n != 1 {
		t.Fatalf("want exactly one join row, got %d:\n%s", n, raw)
	}
}

// TestGuardSessionStartIdentityFailOpen asserts the hook's fail-open contract: a resumed child
// (CLAUDE_CODE_SESSION_ID stripped, so the UUID is blank) writes NO half row — a join needs both
// endpoints — yet still exits 0.
func TestGuardSessionStartIdentityFailOpen(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "") // resumed child: transcript UUID absent

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", "trace-abc"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if _, err := os.Stat(resume.IdentityLedgerPath(regDir)); !os.IsNotExist(err) {
		t.Fatalf("a half row (missing UUID) must not be written; stat err = %v", err)
	}
}

// TestGuardSessionStartArgsThreadsTrace pins the install-side half: a non-empty trace threads
// --trace into the hook argv (so the running hook holds it), an empty one threads nothing.
func TestGuardSessionStartArgsThreadsTrace(t *testing.T) {
	joined := strings.Join(guardSessionStartArgs(true, "trace-xyz"), " ")
	if !strings.Contains(joined, "--trace trace-xyz") {
		t.Fatalf("args missing threaded --trace: %q", joined)
	}
	if !strings.Contains(joined, "--managed") {
		t.Fatalf("args dropped --managed: %q", joined)
	}
	if strings.Contains(strings.Join(guardSessionStartArgs(false, ""), " "), "--trace") {
		t.Fatalf("an empty trace must not emit a --trace flag")
	}
}

// TestInstallGuardSessionStartHookThreadsTrace asserts the trace threads all the way through
// install into the written SessionStart hook settings the child actually loads.
func TestInstallGuardSessionStartHookThreadsTrace(t *testing.T) {
	dir := t.TempDir()
	_, install, err := installGuardSessionStartHookAt([]string{"claude", "-p", "x"}, "on", true, "fak", dir, "", "trace-xyz")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	raw, err := os.ReadFile(install.SettingsPath)
	if err != nil {
		t.Fatalf("read written settings: %v", err)
	}
	if !strings.Contains(string(raw), "trace-xyz") {
		t.Fatalf("written SessionStart hook did not thread the trace id: %s", raw)
	}
}

// TestGuardSessionStartManagedInjectsRule asserts the spine (#3512): a --managed (headless)
// SessionStart injection carries the long-horizon persistence + managed-context rule ON TOP of
// the base affordance, while a plain (attended) injection carries the affordance alone.
func TestGuardSessionStartManagedInjectsRule(t *testing.T) {
	readCtx := func(argv []string) string {
		var out, errb bytes.Buffer
		if code := runGuardSessionStart(&out, &errb, argv); code != 0 {
			t.Fatalf("exit = %d for %v", code, argv)
		}
		var env struct {
			HookSpecificOutput struct {
				AdditionalContext string `json:"additionalContext"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &env); err != nil {
			t.Fatalf("not valid JSON for %v: %v", argv, err)
		}
		return env.HookSpecificOutput.AdditionalContext
	}

	managed := readCtx([]string{"--mode", "on", "--managed"})
	for _, want := range []string{"fak_index_work", "managed context is ON", "CHECKPOINT", "REBUILD"} {
		if !strings.Contains(managed, want) {
			t.Fatalf("managed injection missing %q: %s", want, managed)
		}
	}

	plain := readCtx([]string{"--mode", "on"})
	if !strings.Contains(plain, "fak_index_work") {
		t.Fatalf("plain injection dropped the base affordance: %s", plain)
	}
	if strings.Contains(plain, "managed context is ON") {
		t.Fatalf("plain (attended) injection must NOT carry the long-horizon rule: %s", plain)
	}
}

// TestSessionStartRulePositiveVoice pins the #3566 emit-time guarantee at the SessionStart
// boundary: the additionalContext fak injects is already in positive-voice NORMAL FORM — routing
// it through the deterministic reframe again is a no-op (fixed point) — while every load-bearing
// token (the MCP entry verbs and the managed-context directives) survives the pass byte-for-byte.
func TestSessionStartRulePositiveVoice(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--managed"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	var env struct {
		HookSpecificOutput struct {
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	ctx := env.HookSpecificOutput.AdditionalContext
	// Fixed point: what fak injects has already been through the reframe, so a second pass changes
	// nothing. This is the always-on invariant — no injected string reaches the model un-normalized.
	if reframed := negframe.Reframe(ctx); reframed != ctx {
		t.Fatalf("injected context is not a positive-voice fixed point:\n have: %q\n want: %q", ctx, reframed)
	}
	// The reframe preserved the load-bearing structure it must never mangle.
	for _, tok := range []string{"fak_index_work", "managed context is ON", "CHECKPOINT", "REBUILD"} {
		if !strings.Contains(ctx, tok) {
			t.Fatalf("reframe dropped load-bearing token %q:\n%s", tok, ctx)
		}
	}
}

// TestGuardSessionStartManagedDefault locks the default-on admission (#3512): a headless
// `claude -p` fleet worker is admitted MANAGED (gets the long-horizon rule) by default, while
// an attended interactive `claude` TUI is not. This is the switch that makes the persistence
// posture on-by-default exactly where a human is not present to keep the session going.
func TestGuardSessionStartManagedDefault(t *testing.T) {
	if !guardSessionStartManaged([]string{"claude", "-p", "do the work"}) {
		t.Fatalf("a headless `claude -p` worker must default to MANAGED")
	}
	if guardSessionStartManaged([]string{"claude"}) {
		t.Fatalf("an attended `claude` TUI must not be forced onto the managed posture")
	}
}

// TestGuardSessionStartOffSuppresses asserts the off knob emits nothing (a lean harness
// opts out).
func TestGuardSessionStartOffSuppresses(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "off"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if out.Len() != 0 {
		t.Fatalf("off mode should emit nothing, got: %q", out.String())
	}
}

// TestGuardSessionStartDefaultsOn asserts an empty mode defaults to on (the affordance is
// the fix, so it is on by default).
func TestGuardSessionStartDefaultsOn(t *testing.T) {
	if got := normalizeGuardSessionStartMode(""); got != guardSessionStartModeOn {
		t.Fatalf("empty mode = %q, want on", got)
	}
	if got := normalizeGuardSessionStartMode("OFF"); got != guardSessionStartModeOff {
		t.Fatalf("OFF (case-insensitive) = %q, want off", got)
	}
}

// TestGuardSessionStartSettingsRoundTrip asserts the settings writer emits a SessionStart
// hook entry the merge path can read back, and that the merge preserves a sibling hook.
func TestGuardSessionStartSettingsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/settings.json"

	// Seed a settings file that already carries a Stop hook (a sibling the merge must keep).
	seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
		"Stop": guardStopHookMatchers("fak"),
	}}
	seedData, _ := json.MarshalIndent(seed, "", "  ")
	if err := os.WriteFile(path, seedData, 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := mergeGuardSessionStartIntoSettings(path, "fak", false, ""); err != nil {
		t.Fatalf("merge: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	var got guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("parse merged: %v", err)
	}
	if _, ok := got.Hooks["SessionStart"]; !ok {
		t.Fatalf("merged settings missing SessionStart hook: %s", raw)
	}
	if _, ok := got.Hooks["Stop"]; !ok {
		t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
	}
	// The SessionStart hook must invoke the guard-sessionstart verb.
	if !strings.Contains(string(raw), "guard-sessionstart") {
		t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
	}
}

// TestInstallGuardSessionStartHookAtWiring covers the install path cmdGuard actually invokes
// (guard.go -> installGuardSessionStartHook -> installGuardSessionStartHookAt) for the #3092
// affordance. The actuator/merge tests above exercise emission and merge in isolation; this
// asserts the install BRANCHING that reaches the child — a claude launcher gets the SessionStart
// hook wired into its --settings, a non-claude child and the off knob stay no-ops, and merging
// into an existing guard settings file preserves the sibling hooks. Without this, a regression
// that stops wiring the affordance (re-inerting the fak verbs) would pass every existing test.
func TestInstallGuardSessionStartHookAtWiring(t *testing.T) {
	t.Run("claude child gets a fresh settings file and a --settings repoint", func(t *testing.T) {
		dir := t.TempDir()
		cmd := []string{"claude", "-p", "do the work"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", dir, "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied {
			t.Fatalf("expected Applied=true for a claude child, got %+v", install)
		}
		if !install.Managed {
			t.Fatalf("expected Managed=true for a headless claude -p child, got %+v", install)
		}
		if install.SettingsPath == "" {
			t.Fatalf("expected a written settings path, got empty")
		}
		// The launcher must stay first and gain `--settings <path>` so Claude Code loads the hook.
		if len(out) == 0 || out[0] != "claude" {
			t.Fatalf("launcher token moved or dropped: %v", out)
		}
		if !strings.Contains(strings.Join(out, " "), "--settings "+install.SettingsPath) {
			t.Fatalf("command missing --settings %s: %v", install.SettingsPath, out)
		}
		raw, err := os.ReadFile(install.SettingsPath)
		if err != nil {
			t.Fatalf("read written settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse written settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("written settings missing SessionStart hook: %s", raw)
		}
		if !strings.Contains(string(raw), "guard-sessionstart") {
			t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
		}
		// A managed (headless) install must carry --managed so the injected context includes
		// the long-horizon persistence + managed-context rule (#3512).
		if !strings.Contains(string(raw), "--managed") {
			t.Fatalf("managed SessionStart hook missing --managed arg: %s", raw)
		}
	})

	t.Run("non-claude child is a no-op", func(t *testing.T) {
		cmd := []string{"bash", "-c", "echo hi"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", t.TempDir(), "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("expected no-op for a non-claude child, got %+v", install)
		}
		if install.Reason != "non-claude-child" {
			t.Fatalf("reason = %q, want non-claude-child", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("non-claude command was mutated: %v", out)
		}
	})

	t.Run("off mode stays a no-op even for a claude child", func(t *testing.T) {
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "off", true, "fak", t.TempDir(), "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("off mode should not apply, got %+v", install)
		}
		if install.Reason != "disabled" {
			t.Fatalf("reason = %q, want disabled", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("off mode mutated the command: %v", out)
		}
	})

	t.Run("merges into an existing settings file without re-pointing the command", func(t *testing.T) {
		dir := t.TempDir()
		existing := dir + "/settings.json"
		seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop": guardStopHookMatchers("fak"),
		}}
		seedData, _ := json.MarshalIndent(seed, "", "  ")
		if err := os.WriteFile(existing, seedData, 0o600); err != nil {
			t.Fatalf("seed: %v", err)
		}
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", "", existing, "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied || install.SettingsPath != existing {
			t.Fatalf("expected merge into %s, got %+v", existing, install)
		}
		// The merge branch reuses the existing --settings file, so the command is not repointed.
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("merge branch should not append --settings again: %v", out)
		}
		raw, err := os.ReadFile(existing)
		if err != nil {
			t.Fatalf("read merged settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse merged settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("merge missing SessionStart hook: %s", raw)
		}
		if _, ok := got.Hooks["Stop"]; !ok {
			t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
		}
	})
}

func TestGuardSessionStartHintPositiveFirst(t *testing.T) {
	if !strings.HasPrefix(guardSessionStartHint, "Reach for the fak substrate verbs") {
		t.Fatalf("hint does not lead with affordance: %q", guardSessionStartHint)
	}
	for _, forbidden := range []string{"before working as", "must invoke", "will not", "do not", "never"} {
		if strings.Contains(strings.ToLower(guardSessionStartHint), forbidden) {
			t.Fatalf("hint retains negation-first clause %q: %q", forbidden, guardSessionStartHint)
		}
	}
	for _, token := range []string{"`mcp__fak__fak_index_work`", "`mcp__fak__fak_admit`", "`mcp__fak__fak_adjudicate`", "`mcp__fak__fak_memory_run`", "`mcp__fak__fak_tools_search`"} {
		if !strings.Contains(guardSessionStartHint, token) {
			t.Fatalf("hint dropped %s", token)
		}
	}
	if got := negframe.Reframe(guardSessionStartHint); got != guardSessionStartHint {
		t.Fatalf("positive source is not reframe-idempotent:\n got %q\nwant %q", got, guardSessionStartHint)
	}
}

// TestGuardSessionStartWritesNegframeJournal is #5365's witness for the halves #3568 deferred.
// Before this, the read side (guardNegframeSummaryLine) was structurally silent because NOTHING
// wrote guardNegframeJournalRel, and the emit called negframe.ReframeFakOnly unconditionally so
// #3546's control arm shipped reframed prose anyway. Both are asserted end-to-end here, through
// the real hook actuator rather than the helper in isolation:
//
//   - a SessionStart emit leaves a foldable row, so the exit summary has something to report;
//   - the row names the arm the FAK_ABLATE_NEGFRAME_REFRAME lever actually selected;
//   - on the control arm the injected prose is byte-identical to the raw fragment.
func TestGuardSessionStartWritesNegframeJournal(t *testing.T) {
	// guardNegframeJournalRel is workspace-relative, so each emit runs inside a scratch tree —
	// that also keeps the assertion off this repo's real .fak/ journal.
	emit := func(t *testing.T, argv ...string) (ctx, summary string) {
		t.Helper()
		t.Chdir(t.TempDir())
		var out, errb bytes.Buffer
		if code := runGuardSessionStart(&out, &errb, append([]string{"--mode", "on"}, argv...)); code != 0 {
			t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
		}
		var env struct {
			HookSpecificOutput struct {
				AdditionalContext string `json:"additionalContext"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &env); err != nil {
			t.Fatalf("not valid JSON: %v\n%s", err, out.String())
		}
		return env.HookSpecificOutput.AdditionalContext, guardNegframeSummaryLine(guardNegframeJournalRel)
	}

	t.Run("treatment arm leaves a foldable row", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "1")
		ctx, summary := emit(t, "--managed")
		if summary == "" {
			t.Fatal("SessionStart wrote no journal row — the exit-summary fold is still silent (#5365 item 2)")
		}
		if !strings.Contains(summary, "reframe on") {
			t.Fatalf("row did not record the treatment arm:\n%s", summary)
		}
		if !strings.Contains(ctx, "fak_index_work") {
			t.Fatalf("routing through the lever dropped the affordance:\n%s", ctx)
		}
	})

	// The control arm is the whole point of #3546: with the lever off, the bytes fak injects are
	// the RAW fragment, not a quietly-reframed one, and the row says so.
	t.Run("control arm records the ablated arm and ships raw bytes", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "0")
		ctx, summary := emit(t)
		if summary == "" {
			t.Fatal("control arm wrote no journal row")
		}
		if !strings.Contains(summary, "reframe OFF") {
			t.Fatalf("row did not record the ablated arm:\n%s", summary)
		}
		if ctx != guardSessionStartHint {
			t.Fatalf("control arm rewrote the injected prose:\n got %q\nwant %q", ctx, guardSessionStartHint)
		}
	})

	// Begin, not append: a second SessionStart is a new session boundary, so the fold reports
	// THIS session rather than accumulating the workspace's whole history.
	t.Run("each session start resets the fold", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "1")
		dir := t.TempDir()
		t.Chdir(dir)
		var out, errb bytes.Buffer
		for range 3 {
			if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"}); code != 0 {
				t.Fatalf("exit = %d, want 0", code)
			}
		}
		raw, err := os.ReadFile(guardNegframeJournalRel)
		if err != nil {
			t.Fatalf("read journal: %v", err)
		}
		if n := len(strings.Split(strings.TrimSpace(string(raw)), "\n")); n != 1 {
			t.Fatalf("three session starts left %d rows, want 1 (the boundary must truncate):\n%s", n, raw)
		}
	})
}
