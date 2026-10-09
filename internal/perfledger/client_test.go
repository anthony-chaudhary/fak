package perfledger

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestClassifyClientIsClosedVocabulary(t *testing.T) {
	for _, tc := range []struct {
		probe, client, ua string
		want              string
		synthetic         bool
	}{
		{"halo-canary", "", "Go-http-client/1.1", ClientProbe, true},
		{"", "", "fak-router-canary/1", ClientProbe, true},
		{"", "pi", "OpenAI/JS 4.0", ClientPi, false},
		{"", "Some Secret Harness token=abc", "", ClientOther, false},
		{"", "", "opencode/1.0 ai-sdk/5", ClientOpenCode, false},
		{"", "", "claude-cli/2.1.0 (external, cli)", ClientClaudeCode, false},
		{"", "", "codex_cli_rs/0.40", ClientCodex, false},
		{"", "", "OpenAI/JS 4.0", ClientSDK, false},
		{"", "", "curl/8.4.0", ClientHTTPLib, false},
		{"", "", "Mozilla/5.0", ClientOther, false},
		{"", "", "", ClientUnknown, false},
	} {
		got, synthetic := ClassifyClient(tc.probe, tc.client, tc.ua)
		if got != tc.want || synthetic != tc.synthetic {
			t.Errorf("ClassifyClient(%q,%q,%q) = %q/%v, want %q/%v", tc.probe, tc.client, tc.ua, got, synthetic, tc.want, tc.synthetic)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeExcludesColdSyntheticAndFoldsByClient(t *testing.T) {
	canary := Record{Schema: Schema, PromptTokens: 25, CompletionTokens: 21, CacheRegime: "cold", TTFTMS: 120, Client: ClientProbe, Synthetic: true}
	recs := []Record{
		canary, canary, canary,
		{Schema: Schema, PromptTokens: 1000, CachedTokens: 9000, CacheRegime: "frozen", TTFTMS: 900, Client: ClientPi},
		{Schema: Schema, PromptTokens: 5000, CachedTokens: 5000, CacheRegime: "partial", TTFTMS: 5000, Client: ClientPi},
		{Schema: Schema, PromptTokens: 8000, CacheRegime: "cold", TTFTMS: 9000, Client: ClientOpenCode},
	}
	s := Summarize(recs)
	if s.Count != 3 || s.Probes != 3 {
		t.Fatalf("count/probes = %d/%d, want 3/3", s.Count, s.Probes)
	}
	if got := s.ByRegime["cold"].Count; got != 1 {
		t.Fatalf("cold regime count = %d, want 1 (synthetic canaries excluded)", got)
	}
	pi := s.ByClient[ClientPi]
	if pi.Count != 2 || pi.CacheHitShare != 0.7 || pi.ByRegime["frozen"] != 1 || pi.ByRegime["partial"] != 1 {
		t.Fatalf("pi fold = %+v, want 2 turns at H=0.7", pi)
	}
	if _, ok := s.ByClient[ClientProbe]; ok {
		t.Fatalf("by_client carries the synthetic probe client: %+v", s.ByClient)
	}
	line := RenderCompact(Report{Summary: s})
	for _, want := range []string{"probes=3 excluded", "by client: pi=2 H=70%(f1/p1) opencode=1 H=0%(c1)"} {
		if !strings.Contains(line, want) {
			t.Fatalf("compact line missing %q:\n%s", want, line)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestOldRowsWithoutClientStillParseAndOmitByClient(t *testing.T) {
	var r Record
	if err := json.Unmarshal([]byte(`{"schema":"fak.gateway.perf-record.v1","unix_ms":1,"prompt_tokens":4000,"completion_tokens":5,"cached_tokens":0,"e2e_ms":10}`), &r); err != nil {
		t.Fatalf("decode old row: %v", err)
	}
	if r.Client != "" || r.Synthetic {
		t.Fatalf("old row client/synthetic = %q/%v, want empty", r.Client, r.Synthetic)
	}
	if s := Summarize([]Record{r}); s.Count != 1 || s.ByClient != nil {
		t.Fatalf("old window count/by_client = %d/%v, want 1/nil", s.Count, s.ByClient)
	}
	b, err := json.Marshal(Record{Schema: Schema})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if strings.Contains(string(b), `"client"`) || strings.Contains(string(b), `"synthetic"`) {
		t.Fatalf("unset fields serialized: %s", b)
	}
}
