package sessionaudit

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseGeminiChatFileSyntheticFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "gemini_chat.json")

	// a) ParseGeminiChatFile parses synthetic fixture
	s, err := ParseGeminiChatFile(fixturePath)
	if err != nil {
		t.Fatalf("ParseGeminiChatFile(%q) error: %v", fixturePath, err)
	}
	if s.Error != "" {
		t.Fatalf("Session.Error unexpectedly non-empty: %s", s.Error)
	}

	// b) Model name, turn counts, prompt counts, and token counts
	if s.Session != "gemini-chat-sample-10738" {
		t.Errorf("s.Session = %q, want %q", s.Session, "gemini-chat-sample-10738")
	}
	if s.Kind != KindGemini {
		t.Errorf("s.Kind = %q, want %q", s.Kind, KindGemini)
	}
	if s.AssistantTurns != 4 {
		t.Errorf("s.AssistantTurns = %d, want 4", s.AssistantTurns)
	}
	if s.NPrompts != 2 {
		t.Errorf("s.NPrompts = %d, want 2", s.NPrompts)
	}
	if len(s.Prompts) != 2 {
		t.Errorf("len(s.Prompts) = %d, want 2", len(s.Prompts))
	}
	if s.Prompts[0].Text != "List the files in src and check for any syntax errors." {
		t.Errorf("s.Prompts[0].Text = %q", s.Prompts[0].Text)
	}
	if s.Prompts[1].Text != "No, that is all. Thank you!" {
		t.Errorf("s.Prompts[1].Text = %q", s.Prompts[1].Text)
	}

	// Token counts
	const wantPromptTokens = int64(1240) // 120 + 250 + 380 + 490
	const wantOutputTokens = int64(180)  // 40 + 55 + 65 + 20
	if s.Tokens.Input != wantPromptTokens {
		t.Errorf("s.Tokens.Input = %d, want %d", s.Tokens.Input, wantPromptTokens)
	}
	if s.Tokens.Output != wantOutputTokens {
		t.Errorf("s.Tokens.Output = %d, want %d", s.Tokens.Output, wantOutputTokens)
	}
	if s.TotalInputTokens != wantPromptTokens {
		t.Errorf("s.TotalInputTokens = %d, want %d", s.TotalInputTokens, wantPromptTokens)
	}

	// Models & PerModel
	if s.Models["gemini-2.5-flash"] != 4 {
		t.Errorf("s.Models[gemini-2.5-flash] = %d, want 4", s.Models["gemini-2.5-flash"])
	}
	pm, ok := s.PerModel["gemini-2.5-flash"]
	if !ok {
		t.Fatalf("missing PerModel entry for gemini-2.5-flash")
	}
	if pm.Turns != 4 || pm.Input != wantPromptTokens || pm.Output != wantOutputTokens {
		t.Errorf("PerModel = %+v, want Turns:4, Input:%d, Output:%d", pm, wantPromptTokens, wantOutputTokens)
	}

	// Tool calls & ReadOnly fractions
	if s.NToolUse != 2 {
		t.Errorf("s.NToolUse = %d, want 2", s.NToolUse)
	}
	if s.Tools["Glob"] != 1 || s.Tools["Read"] != 1 {
		t.Errorf("s.Tools = %+v, want Glob:1, Read:1", s.Tools)
	}
	if s.ReadOnlyToolCalls != 2 {
		t.Errorf("s.ReadOnlyToolCalls = %d, want 2", s.ReadOnlyToolCalls)
	}
	if s.ReadOnlyFrac == nil || math.Abs(*s.ReadOnlyFrac-1.0) > 1e-9 {
		t.Errorf("s.ReadOnlyFrac = %v, want 1.0", s.ReadOnlyFrac)
	}
	if s.NToolResult != 2 {
		t.Errorf("s.NToolResult = %d, want 2", s.NToolResult)
	}

	// Timestamps
	if s.TSMin != "2026-08-20T10:00:00Z" {
		t.Errorf("s.TSMin = %q, want %q", s.TSMin, "2026-08-20T10:00:00Z")
	}
	if s.TSMax != "2026-08-20T10:05:00Z" {
		t.Errorf("s.TSMax = %q, want %q", s.TSMax, "2026-08-20T10:05:00Z")
	}
	if s.WallSeconds == nil || *s.WallSeconds != 300 {
		t.Errorf("s.WallSeconds = %v, want 300", s.WallSeconds)
	}
}

func TestAnalyzeDelegatesToGeminiParser(t *testing.T) {
	fixturePath := filepath.Join("testdata", "gemini_chat.json")

	// a) Analyze delegates to Gemini parser when path ends in .json
	s := Analyze(fixturePath)
	if s.Error != "" {
		t.Fatalf("Analyze(%q) error: %s", fixturePath, s.Error)
	}

	if s.Session != "gemini-chat-sample-10738" {
		t.Errorf("s.Session = %q, want %q", s.Session, "gemini-chat-sample-10738")
	}
	if s.AssistantTurns != 4 {
		t.Errorf("s.AssistantTurns = %d, want 4", s.AssistantTurns)
	}
	if s.NPrompts != 2 {
		t.Errorf("s.NPrompts = %d, want 2", s.NPrompts)
	}
	if s.Tokens.Input != 1240 {
		t.Errorf("s.Tokens.Input = %d, want 1240", s.Tokens.Input)
	}
	if s.Tokens.Output != 180 {
		t.Errorf("s.Tokens.Output = %d, want 180", s.Tokens.Output)
	}
	if s.TotalInputTokens != 1240 {
		t.Errorf("s.TotalInputTokens = %d, want 1240", s.TotalInputTokens)
	}
	if s.NToolUse != 2 {
		t.Errorf("s.NToolUse = %d, want 2", s.NToolUse)
	}
}

func TestParseGeminiChatsDirectoryFixtures(t *testing.T) {
	chatsDir := filepath.Join("testdata", "gemini_chats")

	// 1. Sample with tool calls, thoughts, and usageMetadata
	samplePath := filepath.Join(chatsDir, "chat-sample.json")
	s1 := Analyze(samplePath)
	if s1.Error != "" {
		t.Fatalf("chat-sample.json failed: %v", s1.Error)
	}
	if s1.Session != "gemini-sample-001" {
		t.Errorf("s1.Session = %q, want gemini-sample-001", s1.Session)
	}
	if s1.AssistantTurns != 2 {
		t.Errorf("s1.AssistantTurns = %d, want 2", s1.AssistantTurns)
	}
	if s1.NPrompts != 2 {
		t.Errorf("s1.NPrompts = %d, want 2", s1.NPrompts)
	}
	if s1.Models["gemini-2.5-pro"] != 2 {
		t.Errorf("s1.Models = %+v, want gemini-2.5-pro: 2", s1.Models)
	}
	// Turn 1: 1000 prompt, 300 cached -> fresh 700; output 100
	// Turn 2: 1500 prompt, 1000 cached -> fresh 500; output 50
	// Total fresh input: 1200; CacheRead: 1300; Output: 150; Total input: 2500
	if s1.Tokens.Input != 1200 {
		t.Errorf("s1.Tokens.Input = %d, want 1200", s1.Tokens.Input)
	}
	if s1.Tokens.CacheRead != 1300 {
		t.Errorf("s1.Tokens.CacheRead = %d, want 1300", s1.Tokens.CacheRead)
	}
	if s1.Tokens.Output != 150 {
		t.Errorf("s1.Tokens.Output = %d, want 150", s1.Tokens.Output)
	}
	if s1.TotalInputTokens != 2500 {
		t.Errorf("s1.TotalInputTokens = %d, want 2500", s1.TotalInputTokens)
	}
	if s1.CacheHitFrac == nil || math.Abs(*s1.CacheHitFrac-(1300.0/2500.0)) > 1e-6 {
		t.Errorf("s1.CacheHitFrac = %v, want 0.52", s1.CacheHitFrac)
	}
	if s1.NToolUse != 3 {
		t.Errorf("s1.NToolUse = %d, want 3", s1.NToolUse)
	}
	if s1.NToolResult != 3 {
		t.Errorf("s1.NToolResult = %d, want 3", s1.NToolResult)
	}
	if s1.ReadOnlyToolCalls != 2 {
		t.Errorf("s1.ReadOnlyToolCalls = %d, want 2 (read_file, list_directory)", s1.ReadOnlyToolCalls)
	}
	if s1.NThinking != 1 {
		t.Errorf("s1.NThinking = %d, want 1", s1.NThinking)
	}

	// 2. Turns with parts, functionCall, and functionResponse
	turnsPath := filepath.Join(chatsDir, "chat-turns-parts.json")
	s2 := Analyze(turnsPath)
	if s2.Error != "" {
		t.Fatalf("chat-turns-parts.json failed: %v", s2.Error)
	}
	if s2.Session != "gemini-turns-002" {
		t.Errorf("s2.Session = %q, want gemini-turns-002", s2.Session)
	}
	if s2.AssistantTurns != 2 {
		t.Errorf("s2.AssistantTurns = %d, want 2", s2.AssistantTurns)
	}
	if s2.NPrompts != 1 {
		t.Errorf("s2.NPrompts = %d, want 1", s2.NPrompts)
	}
	if s2.Models["gemini-1.5-pro"] != 2 {
		t.Errorf("s2.Models = %+v, want gemini-1.5-pro: 2", s2.Models)
	}
	// Turn 1: 500 prompt, 100 cached -> fresh 400; output 45
	// Turn 2: 600 prompt, 500 cached -> fresh 100; output 20
	// Total fresh input: 500; CacheRead: 600; Output: 65; Total input: 1100
	if s2.Tokens.Input != 500 {
		t.Errorf("s2.Tokens.Input = %d, want 500", s2.Tokens.Input)
	}
	if s2.Tokens.CacheRead != 600 {
		t.Errorf("s2.Tokens.CacheRead = %d, want 600", s2.Tokens.CacheRead)
	}
	if s2.Tokens.Output != 65 {
		t.Errorf("s2.Tokens.Output = %d, want 65", s2.Tokens.Output)
	}
	if s2.NToolUse != 1 || s2.Tools["Glob"] != 1 {
		t.Errorf("s2.NToolUse = %d, tools = %+v", s2.NToolUse, s2.Tools)
	}

	// 3. Tokens field format
	tokensPath := filepath.Join(chatsDir, "chat-tokens.json")
	s3 := Analyze(tokensPath)
	if s3.Error != "" {
		t.Fatalf("chat-tokens.json failed: %v", s3.Error)
	}
	if s3.Session != "gemini-tokens-003" {
		t.Errorf("s3.Session = %q, want gemini-tokens-003", s3.Session)
	}
	if s3.AssistantTurns != 1 {
		t.Errorf("s3.AssistantTurns = %d, want 1", s3.AssistantTurns)
	}
	// Input 3000, cached 1000 -> fresh 2000; Output 60 + thoughts 20 = 80
	if s3.Tokens.Input != 2000 {
		t.Errorf("s3.Tokens.Input = %d, want 2000", s3.Tokens.Input)
	}
	if s3.Tokens.CacheRead != 1000 {
		t.Errorf("s3.Tokens.CacheRead = %d, want 1000", s3.Tokens.CacheRead)
	}
	if s3.Tokens.Output != 80 {
		t.Errorf("s3.Tokens.Output = %d, want 80", s3.Tokens.Output)
	}
	if s3.NToolUse != 1 || s3.Tools["run_shell_command"] != 1 {
		t.Errorf("s3.NToolUse = %d, tools = %+v", s3.NToolUse, s3.Tools)
	}

	// 4. Corrupted JSON file
	corruptedPath := filepath.Join(chatsDir, "corrupted.json")
	s4 := Analyze(corruptedPath)
	if s4.Error == "" {
		t.Fatalf("expected s4.Error on corrupted JSON, got empty")
	}
}

func TestParseGeminiTokensFixture(t *testing.T) {
	fixturePath := filepath.Join("testdata", "gemini_chat_tokens.json")
	s := Analyze(fixturePath)
	if s.Error != "" {
		t.Fatalf("Analyze failed: %v", s.Error)
	}

	if s.Session != "gemini-tokens-session-002" {
		t.Errorf("session ID = %q, want %q", s.Session, "gemini-tokens-session-002")
	}
	if s.Kind != KindGemini {
		t.Errorf("session Kind = %q, want %q", s.Kind, KindGemini)
	}
	if s.NPrompts != 1 {
		t.Errorf("n_prompts = %d, want 1", s.NPrompts)
	}
	if s.AssistantTurns != 1 {
		t.Errorf("assistant turns = %d, want 1", s.AssistantTurns)
	}
	if s.Models["gemini-2.5-flash"] != 1 {
		t.Errorf("models[gemini-2.5-flash] = %d, want 1", s.Models["gemini-2.5-flash"])
	}

	// Tokens: input 5000, cached 2000 -> fresh 3000; output 50 + thoughts 30 = 80
	if s.Tokens.Input != 3000 {
		t.Errorf("tokens.input = %d, want 3000", s.Tokens.Input)
	}
	if s.Tokens.CacheRead != 2000 {
		t.Errorf("tokens.cache_read = %d, want 2000", s.Tokens.CacheRead)
	}
	if s.Tokens.Output != 80 {
		t.Errorf("tokens.output = %d, want 80", s.Tokens.Output)
	}
	if s.TotalInputTokens != 5000 {
		t.Errorf("total_input_tokens = %d, want 5000", s.TotalInputTokens)
	}
	if s.CacheHitFrac == nil || math.Abs(*s.CacheHitFrac-0.4) > 1e-6 {
		t.Errorf("cache_hit_frac = %v, want 0.4", s.CacheHitFrac)
	}

	// Tool: list_directory is read-only
	if s.NToolUse != 1 {
		t.Errorf("n_tool_use = %d, want 1", s.NToolUse)
	}
	if s.ReadOnlyToolCalls != 1 {
		t.Errorf("read_only_tool_calls = %d, want 1", s.ReadOnlyToolCalls)
	}
	if s.ReadOnlyFrac == nil || *s.ReadOnlyFrac != 1.0 {
		t.Errorf("read_only_frac = %v, want 1.0", s.ReadOnlyFrac)
	}
}

func TestParseGeminiSessionCorruptedJSON(t *testing.T) {
	// Corrupted/invalid JSON returns an error or Session.Error cleanly
	cases := []struct {
		name    string
		content string
	}{
		{"malformed syntax", `{"sessionId": "test", "turns": [{"role": "user"`},
		{"empty string", ``},
		{"only whitespace", `   \n\t `},
		{"not a gemini transcript", `{"foo": "bar", "count": 42}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, err := ParseGeminiSession(strings.NewReader(tc.content), "corrupt.json")
			if err == nil && s.Error == "" {
				t.Fatalf("expected error for %s, got err=nil, s.Error=%q", tc.name, s.Error)
			}
		})
	}

	// Missing file through ParseGeminiChatFile
	missingPath := filepath.Join(t.TempDir(), "nonexistent.json")
	s, err := ParseGeminiChatFile(missingPath)
	if err == nil {
		t.Fatalf("expected os.Open error for missing file, got nil")
	}
	if s.Error == "" {
		t.Fatalf("expected s.Error to be populated for missing file")
	}

	// Analyze on missing file
	analyzed := Analyze(missingPath)
	if analyzed.Error == "" {
		t.Fatalf("expected Analyze(missingPath).Error to be populated")
	}
}

func TestParseGeminiSessionArrayOfTurns(t *testing.T) {
	raw := `[
		{
			"role": "user",
			"parts": [{"text": "Hello Gemini"}],
			"timestamp": "2026-08-20T12:00:00Z"
		},
		{
			"role": "model",
			"model": "gemini-1.5-pro",
			"parts": [{"text": "Hello! How can I assist you today?"}],
			"usageMetadata": {
				"promptTokenCount": 15,
				"candidatesTokenCount": 8,
				"totalTokenCount": 23
			},
			"timestamp": "2026-08-20T12:00:02Z"
		}
	]`

	s, err := ParseGeminiSession(strings.NewReader(raw), "array_session.json")
	if err != nil {
		t.Fatalf("ParseGeminiSession error: %v", err)
	}
	if s.AssistantTurns != 1 {
		t.Errorf("AssistantTurns = %d, want 1", s.AssistantTurns)
	}
	if s.NPrompts != 1 {
		t.Errorf("NPrompts = %d, want 1", s.NPrompts)
	}
	if s.Tokens.Input != 15 || s.Tokens.Output != 8 {
		t.Errorf("Tokens = Input:%d Output:%d, want 15/8", s.Tokens.Input, s.Tokens.Output)
	}
	if s.Models["gemini-1.5-pro"] != 1 {
		t.Errorf("Models = %+v, want gemini-1.5-pro:1", s.Models)
	}
}

func TestParseGeminiSessionTopLevelUsage(t *testing.T) {
	raw := `{
		"sessionId": "top-level-usage-sess",
		"model": "gemini-2.5-flash",
		"usageMetadata": {
			"promptTokenCount": 100,
			"candidatesTokenCount": 50,
			"totalTokenCount": 150
		},
		"turns": [
			{
				"role": "user",
				"text": "What is the weather?"
			},
			{
				"role": "model",
				"text": "Sunny and 72F."
			}
		]
	}`

	s, err := ParseGeminiSession(strings.NewReader(raw), "top_level.json")
	if err != nil {
		t.Fatalf("ParseGeminiSession error: %v", err)
	}
	if s.Tokens.Input != 100 || s.Tokens.Output != 50 {
		t.Errorf("Tokens = Input:%d Output:%d, want 100/50", s.Tokens.Input, s.Tokens.Output)
	}
	if s.AssistantTurns != 1 || s.NPrompts != 1 {
		t.Errorf("AssistantTurns:%d NPrompts:%d, want 1/1", s.AssistantTurns, s.NPrompts)
	}
}

func TestParseGeminiSessionPricedModel(t *testing.T) {
	// A model with a published card (e.g. deepseek-v4-pro) calculates CostUSD
	raw := `{
		"sessionId": "priced-test",
		"turns": [
			{
				"role": "user",
				"parts": [{"text": "hello"}]
			},
			{
				"role": "model",
				"model": "deepseek-v4-pro",
				"parts": [{"text": "response"}],
				"usageMetadata": {
					"promptTokenCount": 1000000,
					"candidatesTokenCount": 1000000
				}
			}
		]
	}`

	s, err := ParseGeminiSession(strings.NewReader(raw), "priced.json")
	if err != nil {
		t.Fatalf("ParseGeminiSession error: %v", err)
	}
	// deepseek-v4-pro: input $0.435 / MTok, output $0.87 / MTok => total 1.305
	wantCost := 0.435 + 0.87
	if math.Abs(s.CostUSD-wantCost) > 1e-6 {
		t.Errorf("s.CostUSD = %.6f, want %.6f", s.CostUSD, wantCost)
	}
}

func TestGeminiDiscovery(t *testing.T) {
	root := t.TempDir()

	// Write mock Gemini sessions
	// root/proj-a/chats/session-a.json
	// root/proj-b/chats/session-b.json
	// root/proj-b/chats/sub1/session-sub.json
	chatA := filepath.Join(root, "proj-a", "chats", "session-a.json")
	chatB := filepath.Join(root, "proj-b", "chats", "session-b.json")
	chatSub := filepath.Join(root, "proj-b", "chats", "sub1", "session-sub.json")

	writeJSONFile(t, chatA, map[string]any{
		"sessionId": "sess-a",
		"messages": []map[string]any{
			{"type": "user", "content": "hello from a"},
			{"type": "gemini", "model": "gemini-2.5-pro", "content": "hi from a", "usageMetadata": map[string]any{"promptTokenCount": 100, "candidatesTokenCount": 20}},
		},
	})
	writeJSONFile(t, chatB, map[string]any{
		"sessionId": "sess-b",
		"messages": []map[string]any{
			{"type": "user", "content": "hello from b"},
			{"type": "gemini", "model": "gemini-2.5-flash", "content": "hi from b", "usageMetadata": map[string]any{"promptTokenCount": 200, "candidatesTokenCount": 40}},
		},
	})
	writeJSONFile(t, chatSub, map[string]any{
		"sessionId": "sess-sub",
		"messages": []map[string]any{
			{"type": "user", "content": "sub task"},
			{"type": "gemini", "model": "gemini-2.5-flash", "content": "sub done", "usageMetadata": map[string]any{"promptTokenCount": 50, "candidatesTokenCount": 10}},
		},
	})

	// 1. Discover without subagents
	recs, err := Discover(DiscoverOptions{Roots: []string{root}})
	if err != nil {
		t.Fatalf("Discover failed: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("discovered %d records, want 2 (top-level only)", len(recs))
	}
	for _, r := range recs {
		if r.Kind != KindGemini {
			t.Errorf("record kind = %q, want %q", r.Kind, KindGemini)
		}
		if r.NS != "proj-a" && r.NS != "proj-b" {
			t.Errorf("unexpected NS %q", r.NS)
		}
	}

	// 2. Discover with subagents
	recsAll, err := Discover(DiscoverOptions{Roots: []string{root}, IncludeSubagents: true})
	if err != nil {
		t.Fatalf("Discover with subagents failed: %v", err)
	}
	if len(recsAll) != 3 {
		t.Fatalf("discovered %d records with subagents, want 3", len(recsAll))
	}
	var foundSub bool
	for _, r := range recsAll {
		if r.Kind == KindSpawned {
			foundSub = true
			if r.NS != "proj-b" {
				t.Errorf("subagent NS = %q, want proj-b", r.NS)
			}
		}
	}
	if !foundSub {
		t.Error("expected to find KindSpawned record for nested chat")
	}

	// 3. Namespace filtering
	recsA, err := Discover(DiscoverOptions{Roots: []string{root}, NamespacePrefix: "proj-a"})
	if err != nil {
		t.Fatalf("Discover with ns prefix failed: %v", err)
	}
	if len(recsA) != 1 || recsA[0].NS != "proj-a" {
		t.Fatalf("expected 1 record for proj-a, got %d", len(recsA))
	}

	// 4. Test DiscoverGemini specifically
	gRecs, err := DiscoverGemini(DiscoverOptions{GeminiRoots: []string{root}})
	if err != nil {
		t.Fatalf("DiscoverGemini failed: %v", err)
	}
	if len(gRecs) != 2 {
		t.Fatalf("DiscoverGemini found %d records, want 2", len(gRecs))
	}
}

func TestDefaultDiscoveryHonorsGeminiHome(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("GEMINI_HOME", tmpDir)

	chatPath := filepath.Join(tmpDir, "tmp", "myproj", "chats", "session-1.json")
	writeJSONFile(t, chatPath, map[string]any{
		"sessionId": "sess-default-1",
		"messages": []map[string]any{
			{"type": "user", "content": "test prompt"},
			{"type": "gemini", "model": "gemini-2.5-pro", "content": "response"},
		},
	})

	roots := DefaultGeminiRoots()
	if len(roots) != 1 || roots[0] != filepath.Join(tmpDir, "tmp") {
		t.Fatalf("DefaultGeminiRoots() = %v, want [%s]", roots, filepath.Join(tmpDir, "tmp"))
	}

	recs, err := Discover(DiscoverOptions{})
	if err != nil {
		t.Fatalf("Discover default failed: %v", err)
	}
	var found bool
	for _, r := range recs {
		if r.NS == "myproj" && r.Kind == KindGemini {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("Discover default did not find Gemini session under GEMINI_HOME: %+v", recs)
	}
}

func TestAnalyzeGeminiJSONL(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "session-test.jsonl")

	lines := []string{
		`{"sessionId":"sess-jsonl-1","projectHash":"hash123","startTime":"2026-06-01T00:00:00Z","kind":"main"}`,
		`{"id":"m1","timestamp":"2026-06-01T00:00:01Z","type":"user","content":[{"text":"say hi"}]}`,
		`{"id":"m2","timestamp":"2026-06-01T00:00:05Z","type":"gemini","model":"gemini-2.5-flash","content":"hi!","tokens":{"input":100,"output":10,"cached":0,"total":110}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	s := Analyze(path)
	if s.Error != "" {
		t.Fatalf("Analyze JSONL failed: %v", s.Error)
	}
	if s.Session != "sess-jsonl-1" {
		t.Errorf("session = %q, want sess-jsonl-1", s.Session)
	}
	if s.Kind != KindGemini {
		t.Errorf("kind = %q, want %q", s.Kind, KindGemini)
	}
	if s.NPrompts != 1 {
		t.Errorf("n_prompts = %d, want 1", s.NPrompts)
	}
	if s.AssistantTurns != 1 {
		t.Errorf("assistant turns = %d, want 1", s.AssistantTurns)
	}
	if s.Tokens.Input != 100 || s.Tokens.Output != 10 {
		t.Errorf("tokens = %+v, want input=100 output=10", s.Tokens)
	}
}

func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}
