package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

func TestCodexCompactHookConfigAndTrustHash(t *testing.T) {
	args := guardCodexCompactConfigArgs()
	if len(args) != 6 {
		t.Fatalf("guardCodexCompactConfigArgs len = %d, want 6", len(args))
	}

	preCompactArg := ""
	postCompactArg := ""
	stateArg := ""
	for i := 0; i < len(args); i += 2 {
		if args[i] != "-c" {
			t.Errorf("args[%d] = %q, want -c", i, args[i])
		}
		val := args[i+1]
		switch {
		case strings.HasPrefix(val, "hooks.PreCompact="):
			preCompactArg = val
		case strings.HasPrefix(val, "hooks.PostCompact="):
			postCompactArg = val
		case strings.HasPrefix(val, "hooks.state="):
			stateArg = val
		}
	}

	wantPre := `hooks.PreCompact=[{hooks=[{type="command",command="fak sessions codex-compact-hook --pre",timeout=10}]}]`
	if preCompactArg != wantPre {
		t.Errorf("PreCompact arg = %q, want %q", preCompactArg, wantPre)
	}

	wantPost := `hooks.PostCompact=[{hooks=[{type="command",command="fak sessions codex-compact-hook --post",timeout=10}]}]`
	if postCompactArg != wantPost {
		t.Errorf("PostCompact arg = %q, want %q", postCompactArg, wantPost)
	}

	if !strings.HasPrefix(stateArg, "hooks.state={") || !strings.HasSuffix(stateArg, "}") {
		t.Errorf("stateArg = %q, want hooks.state={...}", stateArg)
	}

	preHash := guardCodexCompactTrustedHash("pre_compact", guardCodexPreCompactCommand, guardCodexCompactTimeoutSeconds)
	if !strings.HasPrefix(preHash, "sha256:") {
		t.Fatalf("preHash = %q, want sha256: prefix", preHash)
	}
	postHash := guardCodexCompactTrustedHash("post_compact", guardCodexPostCompactCommand, guardCodexCompactTimeoutSeconds)
	if !strings.HasPrefix(postHash, "sha256:") {
		t.Fatalf("postHash = %q, want sha256: prefix", postHash)
	}

	posixPreKey := guardCodexHookKeyForOS("pre_compact", "linux")
	wantPosixPreKey := `/<session-flags>/config.toml:pre_compact:0:0`
	if posixPreKey != wantPosixPreKey {
		t.Errorf("posixPreKey = %q, want %q", posixPreKey, wantPosixPreKey)
	}

	winPreKey := guardCodexHookKeyForOS("pre_compact", "windows")
	wantWinPreKey := `C:\<session-flags>\config.toml:pre_compact:0:0`
	if winPreKey != wantWinPreKey {
		t.Errorf("winPreKey = %q, want %q", winPreKey, wantWinPreKey)
	}

	profile, ok := harnessprofile.Lookup("codex")
	if !ok {
		t.Fatalf("harnessprofile.Lookup(codex) failed")
	}

	t.Run("installGuardCodexConfigForProfile adds hooks before exec", func(t *testing.T) {
		cmd := []string{"codex", "exec", "task"}
		out, install := installGuardCodexConfigForProfile(cmd, profile, true, "http://127.0.0.1:8137", "")
		if !install.Applied {
			t.Fatalf("install not applied")
		}
		execIdx := indexOf(out, "exec")
		if execIdx < 0 {
			t.Fatalf("exec not found in out: %v", out)
		}
		preIdx := indexOf(out, wantPre)
		postIdx := indexOf(out, wantPost)
		if preIdx < 0 || preIdx > execIdx {
			t.Errorf("PreCompact hook must precede exec: preIdx=%d, execIdx=%d", preIdx, execIdx)
		}
		if postIdx < 0 || postIdx > execIdx {
			t.Errorf("PostCompact hook must precede exec: postIdx=%d, execIdx=%d", postIdx, execIdx)
		}
	})

	t.Run("installGuardCodexConfigForProfile merges hooks.state when present", func(t *testing.T) {
		cmd := []string{"codex", "-c", `hooks.state={"/<session-flags>/config.toml:session_start:0:0"={trusted_hash="sha256:start"}}`, "exec", "task"}
		out, _ := installGuardCodexConfigForProfile(cmd, profile, true, "http://127.0.0.1:8137", "")
		stateCount := 0
		for _, arg := range out {
			if strings.HasPrefix(arg, "hooks.state=") {
				stateCount++
				if !strings.Contains(arg, "session_start:0:0") {
					t.Errorf("state arg missing session_start: %q", arg)
				}
				if !strings.Contains(arg, "pre_compact:0:0") {
					t.Errorf("state arg missing pre_compact: %q", arg)
				}
				if !strings.Contains(arg, "post_compact:0:0") {
					t.Errorf("state arg missing post_compact: %q", arg)
				}
			}
		}
		if stateCount != 1 {
			t.Errorf("hooks.state count = %d, want exactly 1 merged state table", stateCount)
		}
	})
}

func TestCodexCompactPreAndPostAuditDrop(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	sessionID := "test-sess-compact-001"

	dosDir := filepath.Join(cwd, ".dos")
	if err := os.MkdirAll(dosDir, 0o700); err != nil {
		t.Fatalf("mkdir .dos: %v", err)
	}
	leaseContent := `{"lane":"gateway","tree_globs":["internal/gateway/**"]}` + "\n"
	if err := os.WriteFile(filepath.Join(dosDir, "leases.jsonl"), []byte(leaseContent), 0o600); err != nil {
		t.Fatalf("write leases.jsonl: %v", err)
	}

	goalContent := strings.Join([]string{
		"---",
		"loop: loop-1",
		"witness: commit-audit",
		"lane: gateway",
		"---",
		"# Objective",
		"feat(guard): Codex native PreCompact/PostCompact lifecycle hooks",
		"",
		"# Non-goals",
		"- do not edit sibling packages",
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(cwd, "GOAL.md"), []byte(goalContent), 0o600); err != nil {
		t.Fatalf("write GOAL.md: %v", err)
	}

	promptFile := filepath.Join(home, "compact_prompt.txt")
	t.Setenv("EXPERIMENTAL_COMPACT_PROMPT_FILE", promptFile)

	prePayload, _ := json.Marshal(codexCompactHookInput{
		SessionID: sessionID,
		TurnID:    "turn-10",
		Cwd:       cwd,
	})

	var stdoutPre, stderrPre bytes.Buffer
	code := sessionsCodexCompactHook(&stdoutPre, &stderrPre, bytes.NewReader(prePayload), []string{"--pre", "--codex-home", home})
	if code != 0 {
		t.Fatalf("pre-compact exit = %d, stderr=%s", code, stderrPre.String())
	}

	manifestPath, err := codexPrecompactInvariantsPath(home, sessionID)
	if err != nil {
		t.Fatalf("codexPrecompactInvariantsPath: %v", err)
	}
	rawManifest, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read invariants_precompact.json: %v", err)
	}
	var manifest codexInvariantsPrecompact
	if err := json.Unmarshal(rawManifest, &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}

	if len(manifest.ActiveLeases) == 0 || manifest.ActiveLeases[0].Lane != "gateway" {
		t.Errorf("manifest ActiveLeases = %+v, want lane gateway", manifest.ActiveLeases)
	}
	hasABI := false
	for _, f := range manifest.FrozenSubtrees {
		if f == "internal/abi" {
			hasABI = true
			break
		}
	}
	if !hasABI {
		t.Errorf("manifest FrozenSubtrees = %v, want internal/abi", manifest.FrozenSubtrees)
	}
	if manifest.GoalState.Objective != "feat(guard): Codex native PreCompact/PostCompact lifecycle hooks" {
		t.Errorf("manifest Objective = %q", manifest.GoalState.Objective)
	}
	if manifest.GoalState.Witness != "commit-audit" {
		t.Errorf("manifest Witness = %q", manifest.GoalState.Witness)
	}

	promptData, err := os.ReadFile(promptFile)
	if err != nil {
		t.Fatalf("read promptFile: %v", err)
	}
	if !strings.Contains(string(promptData), "internal/abi") || !strings.Contains(string(promptData), "gateway") {
		t.Errorf("promptFile content missing invariants: %s", string(promptData))
	}

	transcriptPath := filepath.Join(home, "rollout.jsonl")
	transcriptContent := strings.Join([]string{
		`{"timestamp":"2026-09-05T10:00:00Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-09-05T10:05:00Z","type":"compacted","payload":{"message":"Generic compaction summary","replacement_history":[{"text":"Did some generic work, refactored code without mentioning objective or lane."}]}}`,
		`{"timestamp":"2026-09-05T10:05:01Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(transcriptPath, []byte(transcriptContent), 0o600); err != nil {
		t.Fatalf("write transcript: %v", err)
	}

	auditJournalPath := filepath.Join(home, "audit.jsonl")
	t.Setenv("FAK_AUDIT_JOURNAL", auditJournalPath)

	postPayload, _ := json.Marshal(codexCompactHookInput{
		SessionID:      sessionID,
		TurnID:         "turn-10",
		TranscriptPath: transcriptPath,
		Cwd:            cwd,
	})

	var stdoutPost, stderrPost bytes.Buffer
	code = sessionsCodexCompactHook(&stdoutPost, &stderrPost, bytes.NewReader(postPayload), []string{"--post", "--codex-home", home})
	if code != 0 {
		t.Fatalf("post-compact exit = %d, stderr=%s", code, stderrPost.String())
	}

	pendingPath, err := codexPendingRestorationPath(home, sessionID)
	if err != nil {
		t.Fatalf("codexPendingRestorationPath: %v", err)
	}
	rawPending, err := os.ReadFile(pendingPath)
	if err != nil {
		t.Fatalf("pending_restoration.json not created: %v", err)
	}
	var pending codexPendingRestoration
	if err := json.Unmarshal(rawPending, &pending); err != nil {
		t.Fatalf("unmarshal pending_restoration: %v", err)
	}
	if len(pending.Dropped) == 0 {
		t.Fatalf("pending.Dropped is empty, want dropped invariants")
	}

	auditData, err := os.ReadFile(auditJournalPath)
	if err != nil {
		t.Fatalf("audit journal not written: %v", err)
	}
	if !strings.Contains(string(auditData), "COMPACTION_INVARIANT_DROPPED") {
		t.Errorf("audit journal missing COMPACTION_INVARIANT_DROPPED: %s", string(auditData))
	}
}

func TestCodexCompactRestorativeContextInjection(t *testing.T) {
	home := t.TempDir()
	sessionID := "test-sess-compact-002"
	t.Setenv(guardActiveEnv, "1")

	pending := codexPendingRestoration{
		SessionID: sessionID,
		TurnID:    "turn-11",
		Dropped:   []string{"objective", "lane_lease:gateway", "frozen_subtree:internal/abi"},
		Objective: "feat(guard): Codex native PreCompact/PostCompact lifecycle hooks",
		Witness:   "commit-audit",
		LaneLeases: []codexLaneLease{
			{Lane: "gateway", TreeGlobs: []string{"internal/gateway/**"}},
		},
		FrozenSubtrees: []string{"internal/abi"},
		DirtyFiles:     []string{"cmd/fak/guard_codex.go"},
	}
	pending.RestorationNote = formatRestorativeContext(pending)

	pendingPath, err := codexPendingRestorationPath(home, sessionID)
	if err != nil {
		t.Fatalf("codexPendingRestorationPath: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(pendingPath), 0o700); err != nil {
		t.Fatalf("mkdir pending: %v", err)
	}
	pendingData, _ := json.MarshalIndent(pending, "", "  ")
	if err := os.WriteFile(pendingPath, append(pendingData, '\n'), 0o600); err != nil {
		t.Fatalf("write pending: %v", err)
	}

	submitPayload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit","turn_id":"turn-11","prompt":"continue work"}`
	var stdout, stderr bytes.Buffer
	code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(submitPayload), []string{"codex-loop-hook", "--codex-home", home})
	if code != 0 {
		t.Fatalf("codex-loop-hook exit = %d, stderr=%s", code, stderr.String())
	}

	var output codexLoopHookOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		t.Fatalf("decode hook output: %v\nstdout=%s", err, stdout.String())
	}
	if output.HookSpecificOutput == nil {
		t.Fatalf("HookSpecificOutput is nil: %+v", output)
	}
	ctx := output.HookSpecificOutput.AdditionalContext
	for _, want := range []string{
		"[FAK RESTORATIVE INVARIANT RESTORATION]",
		"Context compaction omitted critical system invariants. Re-injecting:",
		"- Objective: feat(guard): Codex native PreCompact/PostCompact lifecycle hooks",
		"- Witness Exit Gate: commit-audit",
		"- Lane Lease: gateway (Tree: internal/gateway/**)",
		"- Frozen Subtrees: internal/abi must not be modified",
		"- In-flight Files: cmd/fak/guard_codex.go",
	} {
		if !strings.Contains(ctx, want) {
			t.Errorf("AdditionalContext missing %q:\n%s", want, ctx)
		}
	}

	if _, err := os.Stat(pendingPath); !os.IsNotExist(err) {
		t.Errorf("pending_restoration.json was not cleared after injection")
	}

	var stdout2, stderr2 bytes.Buffer
	code2 := runSessionsWithStdin(&stdout2, &stderr2, strings.NewReader(submitPayload), []string{"codex-loop-hook", "--codex-home", home})
	if code2 != 0 {
		t.Fatalf("second turn exit = %d", code2)
	}
	if strings.Contains(stdout2.String(), "[FAK RESTORATIVE INVARIANT RESTORATION]") {
		t.Errorf("restorative note re-injected on subsequent turn: %s", stdout2.String())
	}
}
