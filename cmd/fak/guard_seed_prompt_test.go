package main

import (
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
	cmd := []string{"claude", "-p"}
	got, handback, injected := guardSeedPromptRelaunchCommand(cmd, "claude", "resume the triage task", nil)
	if !injected {
		t.Fatalf("recognized agent with a seed must inject, got injected=false")
	}
	if handback != guardRestartHandbackSeedPrompt {
		t.Fatalf("handback = %q, want %q", handback, guardRestartHandbackSeedPrompt)
	}
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "--append-system-prompt resume the triage task") {
		t.Fatalf("seed did not reach the child argv via the prompt entrypoint: %v", got)
	}
	// The prompt flag must be immediately followed by the seed value (argv adjacency).
	found := false
	for i := 0; i+1 < len(got); i++ {
		if got[i] == "--append-system-prompt" && got[i+1] == "resume the triage task" {
			found = true
		}
	}
	if !found {
		t.Fatalf("--append-system-prompt must be followed by the seed value, got %v", got)
	}
	// The input command must not be mutated in place.
	if strings.Join(cmd, " ") != "claude -p" {
		t.Fatalf("input command was mutated: %v", cmd)
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
		t.Fatalf("expected injection")
	}
	if got[len(got)-1] != "short seed" {
		t.Fatalf("an in-budget seed must pass through untouched, got %q", got[len(got)-1])
	}
	if strings.Contains(log.String(), "dropped") {
		t.Fatalf("no truncation should be logged for an in-budget seed, got %q", log.String())
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
	// A second restart in the same session must not stack a second flag; it replaces the prior
	// seed value with the fresher one.
	once, _, _ := guardSeedPromptRelaunchCommand([]string{"claude", "-p"}, "claude", "first seed", nil)
	twice, _, _ := guardSeedPromptRelaunchCommand(once, "claude", "second seed", nil)
	n := 0
	for _, a := range twice {
		if a == "--append-system-prompt" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("--append-system-prompt must appear exactly once across repeated restarts, got %d in %v", n, twice)
	}
	if twice[len(twice)-1] != "second seed" {
		t.Fatalf("the fresher seed must replace the prior value, got %q", twice[len(twice)-1])
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
