package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This file is the adversarial / edge-case sweep for the Codex direct-continuation
// hook (#3903, follow-on to #3023). The hook is deliberately FAIL-OPEN: any input it
// cannot turn into a resolvable direct-provider session allows the turn (exit 0, no
// block), and only an active session whose model_provider is non-empty and not "fak"
// is blocked with a JSON {"decision":"block"} on stdout. Every case below captures
// that expected block/allow decision so a regression that either (a) starts hard-
// failing on a malformed input or (b) stops blocking a real direct-provider session
// is caught. The load-bearing invariant is the second one: no adversarial input may
// weaken the direct-provider refusal.

const codexHookEdgeSessionID = "019f3903-aaaa-7001-b559-000000000001"

// codexHookMetaLine renders one session_meta JSONL record carrying an arbitrary
// model_provider. It marshals via encoding/json so a hostile provider string
// (embedded quotes, control bytes) can never corrupt the fixture itself.
func codexHookMetaLine(t *testing.T, provider string) string {
	t.Helper()
	rec := map[string]any{
		"timestamp": "2026-07-10T17:00:00.000Z",
		"type":      "session_meta",
		"payload": map[string]any{
			"session_id":     codexHookEdgeSessionID,
			"originator":     "codex-tui",
			"cli_version":    "0.142.5",
			"model_provider": provider,
			"git":            map[string]any{"branch": "main"},
		},
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal session_meta: %v", err)
	}
	return string(b)
}

// writeCodexHookSessionFromLines builds a codex-home holding exactly one session
// transcript whose JSONL body is `lines`, and returns the home plus the session id
// its filename carries (the resolver matches by filename substring).
func writeCodexHookSessionFromLines(t *testing.T, lines []string) (home, sessionID string) {
	t.Helper()
	home = filepath.Join(t.TempDir(), "codex-home")
	sessionID = codexHookEdgeSessionID
	dir := filepath.Join(home, "sessions", "2026", "07", "10")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "rollout-2026-07-10T10-00-00-"+sessionID+".jsonl")
	writeCodexLoopFixture(t, path, lines)
	return home, sessionID
}

func runCodexLoopHookForTest(t *testing.T, home, payload string) (code int, stdout, stderr string) {
	t.Helper()
	var outBuf, errBuf bytes.Buffer
	argv := []string{"codex-loop-hook"}
	if home != "" {
		argv = append(argv, "--codex-home", home)
	}
	code = runSessionsWithStdin(&outBuf, &errBuf, strings.NewReader(payload), argv)
	return code, outBuf.String(), errBuf.String()
}

func assertCodexHookAllow(t *testing.T, code int, stdout, stderr string) {
	t.Helper()
	if code != 0 {
		t.Fatalf("hook exit = %d, want 0 (allow); stderr=%s", code, stderr)
	}
	if strings.TrimSpace(stdout) != "" {
		t.Fatalf("allow path emitted a block decision on stdout: %s", stdout)
	}
}

func assertCodexHookBlock(t *testing.T, code int, stdout, wantProviderInReason string) {
	t.Helper()
	if code != 0 {
		t.Fatalf("hook exit = %d, want 0 with a JSON block decision", code)
	}
	var got codexLoopHookBlock
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("decode hook response: %v\nstdout=%s", err, stdout)
	}
	if got.Decision != "block" {
		t.Fatalf("decision = %q, want block: %+v", got.Decision, got)
	}
	if wantProviderInReason != "" && !strings.Contains(got.Reason, "model_provider="+wantProviderInReason) {
		t.Errorf("block reason missing model_provider=%q: %s", wantProviderInReason, got.Reason)
	}
}

func codexHookEdgePayload(sessionID string) string {
	return `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit","turn_id":"turn-next"}`
}

// TestCodexLoopHookMalformedPayload: a hook payload that cannot be decoded into the
// input struct is fail-open — the turn is allowed even when a real direct-provider
// session is present, because the hook never learns which session to inspect.
func TestCodexLoopHookMalformedPayload(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	// A real blocking (openai) session exists; only the malformed PAYLOAD forces allow.
	home, _ := writeCodexHookSessionFromLines(t, []string{codexHookMetaLine(t, "openai")})

	for _, tc := range []struct {
		name    string
		payload string
	}{
		{"empty", ""},
		{"whitespace only", "   \n\t "},
		{"truncated open brace", "{"},
		{"truncated mid key", `{"session_id":`},
		{"not json", "not json at all"},
		{"json array", "[]"},
		{"json null", "null"},
		{"json number", "12345"},
		{"session_id wrong type int", `{"session_id":123}`},
		{"session_id wrong type object", `{"session_id":{"nested":"x"}}`},
		{"trailing garbage after object", `{"session_id":"x"} <<< not json`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, stdout, stderr := runCodexLoopHookForTest(t, home, tc.payload)
			assertCodexHookAllow(t, code, stdout, stderr)
		})
	}
}

// TestCodexLoopHookMissingSessionFile: a well-formed payload whose session_id is
// absent, blank, or resolves to no transcript on disk is allowed (the hook cannot
// diagnose a session it cannot find).
func TestCodexLoopHookMissingSessionFile(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	// A home that DOES contain an openai session, so "unknown id" exercises the
	// resolver's not-found path rather than an empty tree.
	home, _ := writeCodexHookSessionFromLines(t, []string{codexHookMetaLine(t, "openai")})
	emptyHome := filepath.Join(t.TempDir(), "empty-codex-home")

	for _, tc := range []struct {
		name    string
		home    string
		payload string
	}{
		{"absent session_id", home, `{"hook_event_name":"UserPromptSubmit"}`},
		{"empty session_id", home, `{"session_id":""}`},
		{"whitespace session_id", home, `{"session_id":"   "}`},
		{"unknown session_id", home, `{"session_id":"nonexistent-session-zzzz"}`},
		{"home with no sessions dir", emptyHome, `{"session_id":"` + codexHookEdgeSessionID + `"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, stdout, stderr := runCodexLoopHookForTest(t, tc.home, tc.payload)
			assertCodexHookAllow(t, code, stdout, stderr)
		})
	}
}

// TestCodexLoopHookOversizedPayload: the hook reads at most 1 MiB of stdin. A JSON
// object larger than that is truncated and fails to decode -> allow. But an oversized
// stream whose leading object is small and valid still decodes (json reads a single
// value), so trailing bloat can neither force fail-open nor bypass the block of a real
// direct-provider session.
func TestCodexLoopHookOversizedPayload(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	// Neutralize an ambient FAK_GUARD_ACTIVE: inside a `fak guard` session the hook
	// allow-silents before it ever reads the transcript, so the block subtest below
	// would see empty stdout. CI runners have no such variable, which is the only
	// reason this omission stayed invisible.
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSessionFromLines(t, []string{codexHookMetaLine(t, "openai")})

	t.Run("object exceeds 1MiB limit is allowed", func(t *testing.T) {
		giant := `{"session_id":"` + strings.Repeat("a", (1<<20)+4096) + `"}`
		code, stdout, stderr := runCodexLoopHookForTest(t, home, giant)
		assertCodexHookAllow(t, code, stdout, stderr)
	})

	t.Run("small valid object with megabytes of trailing junk still blocks", func(t *testing.T) {
		payload := codexHookEdgePayload(sessionID) + strings.Repeat("x", 2<<20)
		code, stdout, _ := runCodexLoopHookForTest(t, home, payload)
		assertCodexHookBlock(t, code, stdout, "openai")
	})
}

// TestCodexLoopHookPartialFinalLine: a truncated or garbage trailing JSONL record in
// the transcript is skipped by the diagnoser, so it never masks the provider recorded
// in an earlier session_meta. The direct-provider block must survive a partial tail.
func TestCodexLoopHookPartialFinalLine(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "") // ambient `fak guard` would allow-silently; see above

	t.Run("openai meta then partial final line still blocks", func(t *testing.T) {
		home, sessionID := writeCodexHookSessionFromLines(t, []string{
			codexHookMetaLine(t, "openai"),
			`{"timestamp":"2026-07-10T17:01:00Z","type":"response_item","payload":{"type":"func`,
		})
		code, stdout, _ := runCodexLoopHookForTest(t, home, codexHookEdgePayload(sessionID))
		assertCodexHookBlock(t, code, stdout, "openai")
	})

	t.Run("openai meta with interleaved garbage lines still blocks", func(t *testing.T) {
		home, sessionID := writeCodexHookSessionFromLines(t, []string{
			codexHookMetaLine(t, "openai"),
			"",
			"{ this is not json",
			"   ",
			`{"type":"event_msg","payload":{"type":"token_count"`,
		})
		code, stdout, _ := runCodexLoopHookForTest(t, home, codexHookEdgePayload(sessionID))
		assertCodexHookBlock(t, code, stdout, "openai")
	})

	t.Run("only a partial line with no valid meta is allowed", func(t *testing.T) {
		home, sessionID := writeCodexHookSessionFromLines(t, []string{
			`{"timestamp":"2026-07-10T17:00:00Z","type":"session_meta","payl`,
		})
		code, stdout, stderr := runCodexLoopHookForTest(t, home, codexHookEdgePayload(sessionID))
		assertCodexHookAllow(t, code, stdout, stderr)
	})

	t.Run("guarded fak meta then partial final line is allowed", func(t *testing.T) {
		home, sessionID := writeCodexHookSessionFromLines(t, []string{
			codexHookMetaLine(t, "fak"),
			`{"type":"response_item","payload":{"type":"func`,
		})
		code, stdout, _ := runCodexLoopHookForTest(t, home, codexHookEdgePayload(sessionID))
		assertCodexHookBlock(t, code, stdout, "fak")
	})
}

// TestCodexLoopHookEdgeProviderStrings: the guard/direct decision keys only on
// TrimSpace(model_provider) != "" && !EqualFold("fak"). Hostile provider strings —
// mixed case, surrounding whitespace, fak-lookalikes, and a value that tries to inject
// a fake "decision":"allow" — must resolve to the correct block/allow and can never
// forge the emitted decision (the reason string is JSON-escaped).
func TestCodexLoopHookEdgeProviderStrings(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "") // ambient `fak guard` would allow-silently; see above

	for _, tc := range []struct {
		name           string
		provider       string
		wantBlock      bool
		providerInText string // substring expected in the block reason (block cases only)
	}{
		{name: "openai blocks", provider: "openai", wantBlock: true, providerInText: "openai"},
		{name: "mixed case OpenAI blocks", provider: "OpenAI", wantBlock: true, providerInText: "OpenAI"},
		{name: "uppercase ANTHROPIC blocks", provider: "ANTHROPIC", wantBlock: true, providerInText: "ANTHROPIC"},
		{name: "fak without witness blocks", provider: "fak", wantBlock: true, providerInText: "fak"},
		{name: "uppercase FAK without witness blocks", provider: "FAK", wantBlock: true, providerInText: "FAK"},
		{name: "titlecase Fak without witness blocks", provider: "Fak", wantBlock: true, providerInText: "Fak"},
		{name: "padded fak without witness blocks", provider: "  fak  ", wantBlock: true, providerInText: "fak"},
		{name: "empty provider allows", provider: "", wantBlock: false},
		{name: "whitespace provider allows", provider: "   ", wantBlock: false},
		{name: "fak lookalike hyphen blocks", provider: "fak-direct", wantBlock: true, providerInText: "fak-direct"},
		{name: "fak prefix suffix blocks", provider: "fakish", wantBlock: true, providerInText: "fakish"},
		{name: "trailing newline trims then blocks", provider: "openai\n", wantBlock: true, providerInText: "openai"},
		{name: "decision injection cannot forge allow", provider: `openai","decision":"allow","x":"`, wantBlock: true, providerInText: `openai","decision":"allow`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home, sessionID := writeCodexHookSessionFromLines(t, []string{codexHookMetaLine(t, tc.provider)})
			code, stdout, stderr := runCodexLoopHookForTest(t, home, codexHookEdgePayload(sessionID))
			if tc.wantBlock {
				assertCodexHookBlock(t, code, stdout, tc.providerInText)
			} else {
				assertCodexHookAllow(t, code, stdout, stderr)
			}
		})
	}
}
