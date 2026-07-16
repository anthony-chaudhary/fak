package main

import (
	"os"
	"strings"
	"testing"
)

// #3056: the headless/no-continue seed handback. When --continue reattach does not serve a child
// (a foreign binary, or a deliberately fresh-session `claude -p`), the on-disk carryover seed used
// to rot write-only. These are the witness for guardSeedPromptRelaunchCommand + its bounding: the
// bounded seed_text must reach the child argv via the agent's known prompt entrypoint, truncation
// past the token budget must be LOGGED with the dropped count, an unrecognized agent must stay a
// no-op, and the recorded continuity verdict must read handback=seed-prompt.

func TestGuardSeedPromptInjectReachesChildArgv(t *testing.T) {
	got, handback, injected := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", "resume the triage task", nil)
	if !injected || handback != guardRestartHandbackSeedPrompt {
		t.Fatalf("injected=%v handback=%q", injected, handback)
	}
	flag := seedPromptArgIndex(got, "--append-system-prompt-file")
	if flag < 0 || flag+1 >= len(got) {
		t.Fatalf("seed file flag missing: %v", got)
	}
	raw, err := os.ReadFile(got[flag+1])
	if err != nil || string(raw) != "resume the triage task" {
		t.Fatalf("seed file=%q err=%v argv=%v", raw, err, got)
	}
}

func TestGuardSeedPromptInjectTruncationLogged(t *testing.T) {
	// A seed well past the token budget must be truncated AND the drop logged — no silent loss.
	big := strings.Repeat("x", (guardSeedPromptTokenBudget*4)+4000) // ~1000 approx-tokens over budget
	var log strings.Builder
	got, _, injected := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", big, &log)
	if !injected {
		t.Fatalf("expected injection for an oversized seed")
	}
	// The bounded seed on argv must be strictly shorter than the original.
	injectedSeed := got[len(got)-1]
	if len(injectedSeed) >= len(big) {
		t.Fatalf("seed was not truncated: argv seed len=%d, original len=%d", len(injectedSeed), len(big))
	}
	if guardApproxTokens(injectedSeed) > guardSeedPromptTokenBudget {
		t.Fatalf("bounded seed = %d approx-tokens, exceeds budget %d", guardApproxTokens(injectedSeed), guardSeedPromptTokenBudget)
	}
	// The full seed must NOT be present on argv.
	if strings.Contains(strings.Join(got, " "), big) {
		t.Fatalf("full un-truncated seed leaked onto argv")
	}
	// The drop must be logged with both a dropped-token and a dropped-byte count.
	line := log.String()
	if !strings.Contains(line, "seed-prompt") || !strings.Contains(line, "dropped") {
		t.Fatalf("truncation was not logged: %q", line)
	}
	wantTokens := guardApproxTokens(big) - guardApproxTokens(injectedSeed)
	if wantTokens <= 0 {
		t.Fatalf("test setup: expected a positive dropped-token count, got %d", wantTokens)
	}
	if !strings.Contains(line, "approx-tokens") || !strings.Contains(line, "bytes") {
		t.Fatalf("log must name the dropped approx-token AND byte count: %q", line)
	}
}

func TestGuardSeedPromptInjectNoTruncationLogWhenFits(t *testing.T) {
	var log strings.Builder
	got, _, injected := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", "short seed", &log)
	if !injected {
		t.Fatal("short seed should inject")
	}
	flag := seedPromptArgIndex(got, "--append-system-prompt-file")
	raw, err := os.ReadFile(got[flag+1])
	if err != nil || string(raw) != "short seed" {
		t.Fatalf("seed file=%q err=%v", raw, err)
	}
	if log.Len() != 0 {
		t.Fatalf("in-budget seed logged truncation: %q", log.String())
	}
}

func TestGuardSeedPromptInjectUnknownAgentNoOp(t *testing.T) {
	// fak never guesses a foreign tool's prompt syntax: an unrecognized agent is a no-op and the
	// seed is left on disk (the command is returned unchanged, injected=false).
	cmd := []string{"codex", "run"}
	got, handback, injected := guardSeedPromptRelaunchCommand(cmd, "codex", "resume the triage task", nil)
	if injected || handback != "" {
		t.Fatalf("unrecognized agent must be a no-op, got injected=%v handback=%q", injected, handback)
	}
	if strings.Join(got, " ") != strings.Join(cmd, " ") {
		t.Fatalf("unrecognized agent command must be unchanged, got %v", got)
	}
}

func TestGuardSeedPromptInjectEmptySeedNoOp(t *testing.T) {
	cmd := []string{"claude", "-p"}
	got, handback, injected := guardSeedPromptRelaunchCommand(cmd, "claude", "   ", nil)
	if injected || handback != "" {
		t.Fatalf("empty seed must be a no-op, got injected=%v handback=%q", injected, handback)
	}
	if strings.Join(got, " ") != strings.Join(cmd, " ") {
		t.Fatalf("empty-seed command must be unchanged, got %v", got)
	}
}

func TestGuardSeedPromptInjectIdempotent(t *testing.T) {
	once, _, _ := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", "first seed", nil)
	twice, _, _ := guardSeedPromptRelaunchCommand(once, "claude", "second seed", nil)
	if n := seedPromptArgCount(twice, "--append-system-prompt-file"); n != 1 {
		t.Fatalf("file flag count=%d argv=%v", n, twice)
	}
	flag := seedPromptArgIndex(twice, "--append-system-prompt-file")
	raw, err := os.ReadFile(twice[flag+1])
	if err != nil || string(raw) != "second seed" {
		t.Fatalf("second seed=%q err=%v", raw, err)
	}
}

func TestGuardSeedPromptHopContinuityVerdict(t *testing.T) {
	// Acceptance criterion #4: the continuity verdict (the hop's handback field) reflects the
	// prompt-inject path as seed-prompt, with an engaged (ok) status because the seed reached argv.
	ev := guardBudgetRestartEvent{
		FromTraceID: "gw-1",
		ToTraceID:   "gw-2",
		SeedFile:    "/tmp/fak-guard-reset-1/reset-gw-1-to-gw-2.json",
		SeedText:    "resume the triage task",
	}
	hop := guardRestartHopFromEventHandback(ev, 1, "claude", guardRestartHandbackSeedPrompt)
	if hop.Handback != guardRestartHandbackSeedPrompt {
		t.Fatalf("hop.Handback = %q, want %q", hop.Handback, guardRestartHandbackSeedPrompt)
	}
	if hop.Status != "ok" {
		t.Fatalf("a seed-prompt hop with a present seed must be status=ok, got %q", hop.Status)
	}
	if !strings.Contains(guardRestartHopOneLiner(hop), "handback=seed-prompt") {
		t.Fatalf("one-liner must carry handback=seed-prompt, got %q", guardRestartHopOneLiner(hop))
	}
	// An empty override still derives the legacy continue handback for a recognized agent, so the
	// #3055 path is untouched.
	legacy := guardRestartHopFromEventHandback(ev, 1, "claude", "")
	if legacy.Handback != guardRestartHandbackContinue {
		t.Fatalf("empty override must derive continue for claude, got %q", legacy.Handback)
	}
}

func TestGuardStripContinueFlag(t *testing.T) {
	// Recognized agent: --continue is removed; the binary and every other arg are preserved.
	if got := guardStripContinueFlag([]string{"claude", "--continue", "-p", "task"}, "claude"); strings.Join(got, " ") != "claude -p task" {
		t.Fatalf("strip = %v, want [claude -p task]", got)
	}
	// No --continue present: unchanged.
	if got := guardStripContinueFlag([]string{"claude", "-p"}, "claude"); strings.Join(got, " ") != "claude -p" {
		t.Fatalf("no-op strip changed the command: %v", got)
	}
	// Unrecognized agent: fak knows no resume flag for it, so it never strips (no-op) — it must not
	// guess that a foreign tool's "--continue" means the same thing.
	if got := guardStripContinueFlag([]string{"codex", "--continue"}, "codex"); strings.Join(got, " ") != "codex --continue" {
		t.Fatalf("unrecognized agent must be a no-op, got %v", got)
	}
	// The input command must not be mutated in place.
	in := []string{"claude", "--continue", "run"}
	_ = guardStripContinueFlag(in, "claude")
	if strings.Join(in, " ") != "claude --continue run" {
		t.Fatalf("input command was mutated: %v", in)
	}
}

func TestGuardSeedHandbackBootsFreshStrippingContinue(t *testing.T) {
	// Fix #2 (authoritative restart seed): a command carrying a stale --continue from a prior fallback
	// restart must, on the seed path, BOTH gain the bounded seed prompt AND lose --continue, so the
	// child boots fresh on the distilled seed instead of reattaching — and re-inflating — the exhausted
	// transcript. This mirrors exactly what the supervision loop does on a restart.
	command := []string{"claude", "-p", "--continue"}
	next, hb, injected := guardSeedPromptRelaunchCommand(command, "claude", "resume the triage task", nil)
	if !injected || hb != guardRestartHandbackSeedPrompt {
		t.Fatalf("expected seed-prompt injection, got injected=%v hb=%q", injected, hb)
	}
	next = guardStripContinueFlag(next, "claude")
	joined := strings.Join(next, " ")
	if strings.Contains(joined, "--continue") {
		t.Fatalf("--continue must be stripped on the seed path (no transcript re-inflation): %v", next)
	}
	if n := seedPromptArgCount(next, "--append-system-prompt-file"); n != 1 {
		t.Fatalf("file-backed seed prompt count = %d, want 1: %v", n, next)
	}
	flag := seedPromptArgIndex(next, "--append-system-prompt-file")
	raw, err := os.ReadFile(next[flag+1])
	if err != nil {
		t.Fatalf("read file-backed seed prompt: %v", err)
	}
	if got := string(raw); got != "resume the triage task" {
		t.Fatalf("bounded seed file = %q, want authoritative restart seed", got)
	}
}

func TestGuardBoundSeedPrompt(t *testing.T) {
	// Under budget: untouched, zero dropped.
	if b, d := guardBoundSeedPrompt("hello", 100); b != "hello" || d != 0 {
		t.Fatalf("in-budget seed = (%q,%d), want (hello,0)", b, d)
	}
	// Over budget: bounded to the byte budget, positive drop, and the sum accounts for every token.
	seed := strings.Repeat("a", 800) // 200 approx-tokens
	b, d := guardBoundSeedPrompt(seed, 50)
	if guardApproxTokens(b) > 50 {
		t.Fatalf("bounded to %d approx-tokens, want <= 50", guardApproxTokens(b))
	}
	if d != guardApproxTokens(seed)-guardApproxTokens(b) {
		t.Fatalf("dropped count %d != total-bounded %d", d, guardApproxTokens(seed)-guardApproxTokens(b))
	}
	if d <= 0 {
		t.Fatalf("over-budget seed must report a positive drop, got %d", d)
	}
	// A multi-byte rune must never be split at the cut boundary: the result stays valid UTF-8.
	runes := strings.Repeat("é", 400) // 2 bytes each = 800 bytes, 200 approx-tokens
	rb, _ := guardBoundSeedPrompt(runes, 50)
	if !isValidUTF8Prefix(rb) {
		t.Fatalf("bounded seed split a multi-byte rune: %q", rb)
	}
}

// isValidUTF8Prefix reports whether s ends on a rune boundary (no trailing continuation byte left
// dangling), the property guardBoundSeedPrompt must preserve when it cuts.
func isValidUTF8Prefix(s string) bool {
	if s == "" {
		return true
	}
	// The last byte must not be a lone leading byte of an unfinished multi-byte rune, and the cut
	// must have landed on a rune start. strings.ToValidUTF8 replacing nothing is the simplest proof.
	return strings.ToValidUTF8(s, "�") == s
}
