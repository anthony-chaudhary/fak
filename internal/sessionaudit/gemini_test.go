package sessionaudit

import (
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

func TestParseGeminiSessionCorruptedJSON(t *testing.T) {
	// c) Corrupted/invalid JSON returns an error or Session.Error cleanly
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

func TestDiscoverGeminiAndDiscoverWithGemini(t *testing.T) {
	dir := t.TempDir()
	// Create layout: dir/proj-hash/chats/chat1.json
	wsDir := filepath.Join(dir, "proj-hash", "chats")
	if err := os.MkdirAll(wsDir, 0755); err != nil {
		t.Fatalf("MkdirAll error: %v", err)
	}
	chatPath := filepath.Join(wsDir, "session_alpha.json")
	content := `{
		"sessionId": "session-alpha",
		"model": "gemini-2.5-flash",
		"turns": [
			{"role": "user", "text": "hello"},
			{"role": "model", "text": "hi", "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
		]
	}`
	if err := os.WriteFile(chatPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}

	// 1. DiscoverGemini
	transcripts, err := DiscoverGemini(DiscoverOptions{Roots: []string{dir}})
	if err != nil {
		t.Fatalf("DiscoverGemini error: %v", err)
	}
	if len(transcripts) != 1 {
		t.Fatalf("DiscoverGemini found %d transcripts, want 1", len(transcripts))
	}
	if transcripts[0].Path != chatPath {
		t.Errorf("transcript.Path = %q, want %q", transcripts[0].Path, chatPath)
	}
	if transcripts[0].NS != "proj-hash" {
		t.Errorf("transcript.NS = %q, want %q", transcripts[0].NS, "proj-hash")
	}

	// 2. Discover with Roots pointing to dir
	allTranscripts, err := Discover(DiscoverOptions{Roots: []string{dir}})
	if err != nil {
		t.Fatalf("Discover error: %v", err)
	}
	if len(allTranscripts) != 1 {
		t.Fatalf("Discover found %d transcripts, want 1", len(allTranscripts))
	}
	if allTranscripts[0].Path != chatPath {
		t.Errorf("allTranscripts[0].Path = %q, want %q", allTranscripts[0].Path, chatPath)
	}

	// 3. Analyze through the discovered path
	sess := Analyze(allTranscripts[0].Path)
	if sess.Error != "" {
		t.Fatalf("Analyze on discovered path failed: %s", sess.Error)
	}
	if sess.AssistantTurns != 1 || sess.NPrompts != 1 {
		t.Errorf("sess turns = %d / %d, want 1 / 1", sess.AssistantTurns, sess.NPrompts)
	}
	if sess.Tokens.Input != 10 || sess.Tokens.Output != 5 {
		t.Errorf("sess tokens = %d / %d, want 10 / 5", sess.Tokens.Input, sess.Tokens.Output)
	}
}
