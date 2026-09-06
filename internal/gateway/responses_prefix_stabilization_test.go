package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// makeLargeStandingInstructions generates standing developer instructions aligned to a 1024-token block boundary.
func makeLargeStandingInstructions(targetTokens int, tools []responsesTool) string {
	base := "You are an autonomous software engineering assistant operating in a secure sandbox.\n" +
		"Strictly adhere to repository boundaries. Never mutate unassigned files.\n" +
		"Write reproduction tests first before implementing bug fixes.\n" +
		"Verify all code modifications with deterministic unit tests and package-scoped checks.\n" +
		"Preserve git commit message conventions, DCO sign-offs, and trailer rules.\n" +
		"Always respect tool capability floors, rate limits, and least-privilege principles.\n"
	var sb strings.Builder
	for EstimateTokensBytes(RenderInvariantPrefix(sb.String()+base, tools)) <= targetTokens {
		sb.WriteString(base)
	}
	inst := sb.String()
	for EstimateTokensBytes(RenderInvariantPrefix(inst+"X", tools)) <= targetTokens {
		inst += "X"
	}
	if rem := EstimateTokensBytes(RenderInvariantPrefix(inst, tools)) % 1024; rem != 0 {
		for EstimateTokensBytes(RenderInvariantPrefix(inst, tools))%1024 != 0 {
			inst += "X"
		}
	}
	return inst
}

func sampleTestTools() []responsesTool {
	return []responsesTool{
		{
			Type:        "function",
			Name:        "search_code",
			Description: "Search code in repository by regex pattern",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"}}}`),
		},
		{
			Type:        "function",
			Name:        "edit_file",
			Description: "Edit existing file with exact string replacements",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"filePath":{"type":"string"}}}`),
		},
		{
			Type:        "function",
			Name:        "bash",
			Description: "Execute shell command within sandbox",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`),
		},
		{
			Type:        "function",
			Name:        "fetch_url",
			Description: "Fetch web content by URL",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"url":{"type":"string"}}}`),
		},
		{
			Type:        "function",
			Name:        "git_commit",
			Description: "Record staged changes to git history",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"message":{"type":"string"}}}`),
		},
	}
}

// TestResponsesPrefixStabilizationAcrossCompactionBoundary tests Turn N (pre-compaction)
// and Turn N+1 (post-compaction with synthetic summary and altered timestamp):
// 1. Byte stream from Byte Offset 0 through final tool schema is bit-identical between Turn N and Turn N+1.
// 2. >=98% of the initial post-compaction prompt tokens align with 1024-token block boundaries of Turn N.
func TestResponsesPrefixStabilizationAcrossCompactionBoundary(t *testing.T) {
	toolsN := sampleTestTools() // in order: search_code, edit_file, bash, fetch_url, git_commit
	standingBase := makeLargeStandingInstructions(4096, toolsN)

	// Turn N: Pre-compaction request with Turn N timestamp and arbitrary tool order
	turnNInstructions := standingBase + "\nToday is Friday, Sep 04, 2026. Current turn: 120."
	msgsN := []agent.Message{
		{Role: agent.RoleSystem, Content: turnNInstructions},
		{Role: agent.RoleUser, Content: "Please review the authentication module."},
		{Role: agent.RoleAssistant, Content: "I will inspect internal/auth/session.go."},
		{Role: agent.RoleUser, Content: "Proceed with the inspection."},
	}

	// Turn N+1: Post-compaction request crossing 96k token boundary
	// The client injects a synthetic compaction summary and an updated timestamp at the head
	compactionSummary := "<compaction_summary>\n" +
		"Turns 1-118 summarized:\n" +
		"- User requested auth module review.\n" +
		"- Agent identified session token expiry issue.\n" +
		"- Unit test was added and passed.\n" +
		"</compaction_summary>"

	turnN1Instructions := compactionSummary + "\n" +
		standingBase + "\n" +
		"Today is Saturday, Sep 05, 2026. Current turn: 121."

	// Reverse tool order to simulate client permutation
	toolsN1 := []responsesTool{
		toolsN[4], toolsN[3], toolsN[2], toolsN[1], toolsN[0],
	}

	// Pruned history preserving only recent turn sequence
	msgsN1 := []agent.Message{
		{Role: agent.RoleSystem, Content: turnN1Instructions},
		{Role: agent.RoleUser, Content: "Continue with the remaining test coverage."},
	}

	// Apply CanonicalizeResponsesPrefix to Turn N
	cleanInstN, sortedToolsN, stabilizedMsgsN, ratioN := CanonicalizeResponsesPrefix(turnNInstructions, toolsN, msgsN)
	if ratioN < 0.98 {
		t.Fatalf("Turn N ratio = %f, want >= 0.98", ratioN)
	}

	// Apply CanonicalizeResponsesPrefix to Turn N+1
	cleanInstN1, sortedToolsN1, stabilizedMsgsN1, ratioN1 := CanonicalizeResponsesPrefix(turnN1Instructions, toolsN1, msgsN1)
	if ratioN1 < 0.98 {
		t.Fatalf("Turn N+1 ratio = %f, want >= 0.98", ratioN1)
	}

	// 1. Invariant Prefix Bit-Identical Verification:
	// The byte stream from Byte Offset 0 through final tool schema MUST be bit-identical.
	bytesN := RenderInvariantPrefix(cleanInstN, sortedToolsN)
	bytesN1 := RenderInvariantPrefix(cleanInstN1, sortedToolsN1)

	if !bytes.Equal(bytesN, bytesN1) {
		t.Fatalf("Byte stream from Byte Offset 0 through final tool schema is NOT bit-identical between Turn N and Turn N+1:\nTurn N bytes: %d, Turn N+1 bytes: %d", len(bytesN), len(bytesN1))
	}

	if cleanInstN != cleanInstN1 {
		t.Fatalf("Cleaned instructions differ:\nTurn N: %q\nTurn N+1: %q", cleanInstN, cleanInstN1)
	}

	if len(sortedToolsN) != len(sortedToolsN1) {
		t.Fatalf("Sorted tools count mismatch: %d vs %d", len(sortedToolsN), len(sortedToolsN1))
	}
	for i := range sortedToolsN {
		if sortedToolsN[i].Name != sortedToolsN1[i].Name {
			t.Fatalf("Tool[%d] name mismatch: %s vs %s", i, sortedToolsN[i].Name, sortedToolsN1[i].Name)
		}
	}

	// Verify tools are strictly in alphabetical order: bash, edit_file, fetch_url, git_commit, search_code
	expectedToolOrder := []string{"bash", "edit_file", "fetch_url", "git_commit", "search_code"}
	for i, expected := range expectedToolOrder {
		if sortedToolsN1[i].Name != expected {
			t.Errorf("Tool[%d] = %s, want %s", i, sortedToolsN1[i].Name, expected)
		}
	}

	// 2. >=98% 1024-token block boundary alignment verification:
	tokN := EstimateTokensBytes(bytesN)
	tokN1 := EstimateTokensBytes(bytesN1)
	if tokN < 1024 || tokN1 < 1024 {
		t.Fatalf("Test setup error: expected >= 1024 prefix tokens, got tokN=%d, tokN1=%d", tokN, tokN1)
	}

	blockAlignment := ComputePrefixBlockAlignment(bytesN, bytesN1)
	if blockAlignment < 0.98 {
		t.Fatalf("Prefix block alignment = %f, want >= 0.98", blockAlignment)
	}

	// 3. Volatile session metadata relocation:
	// Verify volatile timestamps were removed from instructions and relocated to suffix of latest user message.
	if strings.Contains(cleanInstN, "Today is") || strings.Contains(cleanInstN, "Current turn") {
		t.Errorf("Turn N instructions still contain volatile metadata: %s", cleanInstN)
	}
	if strings.Contains(cleanInstN1, "Today is") || strings.Contains(cleanInstN1, "Current turn") {
		t.Errorf("Turn N+1 instructions still contain volatile metadata: %s", cleanInstN1)
	}

	lastUserMsgN := stabilizedMsgsN[len(stabilizedMsgsN)-1]
	if !strings.Contains(lastUserMsgN.Content, "Today is Friday, Sep 04, 2026") {
		t.Errorf("Turn N latest user message missing relocated timestamp: %s", lastUserMsgN.Content)
	}

	lastUserMsgN1 := stabilizedMsgsN1[len(stabilizedMsgsN1)-1]
	if !strings.Contains(lastUserMsgN1.Content, "Today is Saturday, Sep 05, 2026") {
		t.Errorf("Turn N+1 latest user message missing relocated timestamp: %s", lastUserMsgN1.Content)
	}

	// 4. Compaction summary re-anchored strictly after Tier 0:
	if strings.Contains(cleanInstN1, "<compaction_summary>") {
		t.Errorf("Turn N+1 Tier 0 standing instructions still contain compaction summary")
	}

	// The summary block must be anchored strictly after Tier 0 as root of pruned history (at index 1)
	if len(stabilizedMsgsN1) < 2 {
		t.Fatalf("Turn N+1 stabilized messages length = %d, want at least 2", len(stabilizedMsgsN1))
	}
	if !strings.Contains(stabilizedMsgsN1[1].Content, "<compaction_summary>") {
		t.Errorf("Turn N+1 message at index 1 does not contain compaction summary: %v", stabilizedMsgsN1[1])
	}

	// 5. Negative reproduction witness: Without prefix stabilization, Byte Offset 0 shifts
	// and 100% of 1024-token prompt cache blocks are busted.
	rawBytesN := RenderInvariantPrefix(turnNInstructions, toolsN)
	rawBytesN1 := RenderInvariantPrefix(turnN1Instructions, toolsN1)

	if bytes.Equal(rawBytesN, rawBytesN1) {
		t.Fatalf("Raw unstabilized Turn N and N+1 unexpectedly matched")
	}

	rawAlignment := ComputePrefixBlockAlignment(rawBytesN, rawBytesN1)
	if rawAlignment > 0.05 {
		t.Fatalf("Raw unstabilized Turn N and N+1 had alignment %f, expected cold cache prefill penalty (< 0.05)", rawAlignment)
	}
}

// TestResponsesPrefixStabilizationVolatileExtraction tests standalone and inline volatile extraction.
func TestResponsesPrefixStabilizationVolatileExtraction(t *testing.T) {
	input := "You are an assistant.\n" +
		"Today is 2026-09-05.\n" +
		"Current turn: 42.\n" +
		"Session ID: sess_test_12345.\n" +
		"Always answer truthfully."

	cleaned, extracted := ExtractVolatileMetadata(input)
	if len(extracted) != 3 {
		t.Fatalf("expected 3 extracted items, got %d: %v", len(extracted), extracted)
	}
	if strings.Contains(cleaned, "Today is") {
		t.Errorf("cleaned still has 'Today is': %q", cleaned)
	}
	if strings.Contains(cleaned, "Current turn") {
		t.Errorf("cleaned still has 'Current turn': %q", cleaned)
	}
	if strings.Contains(cleaned, "Session ID") {
		t.Errorf("cleaned still has 'Session ID': %q", cleaned)
	}
	if !strings.Contains(cleaned, "You are an assistant.") || !strings.Contains(cleaned, "Always answer truthfully.") {
		t.Errorf("cleaned lost standing instructions: %q", cleaned)
	}
}

// TestResponsesPrefixStabilizationSummaryExtraction tests <summary> and <compaction_summary> blocks.
func TestResponsesPrefixStabilizationSummaryExtraction(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		wantSum string
	}{
		{
			name:    "standard_summary_tag",
			input:   "Standing rules.\n<summary>\nTurn 1-5 done\n</summary>\nMore rules.",
			wantSum: "<summary>\nTurn 1-5 done\n</summary>",
		},
		{
			name:    "compaction_summary_tag",
			input:   "Standing rules.\n<compaction_summary>\nPrior state summarized\n</compaction_summary>\nMore rules.",
			wantSum: "<compaction_summary>\nPrior state summarized\n</compaction_summary>",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cleaned, sum := ExtractCompactionSummary(tc.input)
			if sum != tc.wantSum {
				t.Errorf("summary = %q, want %q", sum, tc.wantSum)
			}
			if strings.Contains(cleaned, "<summary>") || strings.Contains(cleaned, "<compaction_summary>") {
				t.Errorf("cleaned still has summary tag: %q", cleaned)
			}
			if !strings.Contains(cleaned, "Standing rules.") || !strings.Contains(cleaned, "More rules.") {
				t.Errorf("cleaned lost surrounding instructions: %q", cleaned)
			}
		})
	}
}

// TestResponsesPrefixStabilizationCanonicalToolOrder tests that arbitrary tool orderings
// always sort into the exact same canonical alphabetical order.
func TestResponsesPrefixStabilizationCanonicalToolOrder(t *testing.T) {
	t1 := responsesTool{Name: "zebra"}
	t2 := responsesTool{Name: "apple"}
	t3 := responsesTool{Name: "mango"}

	sorted := CanonicalizeTools([]responsesTool{t1, t2, t3})
	if len(sorted) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(sorted))
	}
	if sorted[0].Name != "apple" || sorted[1].Name != "mango" || sorted[2].Name != "zebra" {
		t.Errorf("tools not canonically sorted: %v", sorted)
	}
}

// TestResponsesPrefixEndToEndGateway verifies that handleResponses applies prefix stabilization
// and records/logs the prefix reuse ratio on incoming requests.
func TestResponsesPrefixEndToEndGateway(t *testing.T) {
	srv := newTestServer(t)
	var logged []string
	srv.logf = func(format string, args ...any) {
		msg := fmt.Sprintf(format, args...)
		logged = append(logged, strings.TrimSpace(strings.TrimRight(msg, "\n")))
	}

	testTools := []responsesTool{
		{Type: "function", Name: "zebra"},
		{Type: "function", Name: "alpha"},
	}
	standing := makeLargeStandingInstructions(2048, testTools)
	body := map[string]any{
		"model":        "test-model",
		"instructions": standing + "\nToday is 2026-09-05. Current turn: 10.",
		"tools": []map[string]any{
			{"type": "function", "name": "zebra"},
			{"type": "function", "name": "alpha"},
		},
		"input": []map[string]any{
			{"role": "user", "content": "Execute test command"},
		},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("json marshal: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(raw))
	w := httptest.NewRecorder()

	srv.handleResponses(w, req)

	// Verify that fak_gateway_compaction_prefix_reuse_ratio was logged
	foundMetric := false
	for _, l := range logged {
		if strings.Contains(l, "fak_gateway_compaction_prefix_reuse_ratio") {
			foundMetric = true
			break
		}
	}
	if !foundMetric {
		t.Errorf("fak_gateway_compaction_prefix_reuse_ratio not found in logs: %v", logged)
	}
}
