package trajectory

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditCompactionReconstructsAndResetsExplicitContent(t *testing.T) {
	root := t.TempDir()
	lines := []string{
		`{"type":"session_meta","payload":{"id":"compaction-fixture"}}`,
		`{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"é"}]}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"probe","call_id":"one","arguments":"{}"}}`,
		`{"type":"response_item","payload":{"type":"function_call_output","call_id":"one","output":"abc"}}`,
		`{"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":[{"type":"input_text","text":"x"}]}]}}`,
		`{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"🙂"}]}}`,
		`{"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":[{"type":"input_text","text":"yy"}]}]}}`,
		`{"type":"compacted","payload":{"message":"summary alone is incomplete"}}`,
		`{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"append"}]}}`,
		`{"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":"restored"}]}}`,
		`{"type":"compacted","payload":{"replacement_history":[]}}`,
		`{"type":"response_item","payload":{"type":"message","role":"user","content":"z"}}`,
		`{BROKEN}`,
		`{"type":"compacted","payload":{"replacement_history":[]}}`,
		`{"type":"response_item","payload":{"type":"reasoning","encrypted_content":"opaque"}}`,
		`{"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"private"}]}]}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":5}}}}`,
	}
	if err := os.WriteFile(filepath.Join(root, "session.jsonl"), []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceCodex, Root: root, RootLabel: "fixture"}}})
	if err != nil {
		t.Fatal(err)
	}
	events := result.Transcripts[0].CompactionEvents
	if len(events) != 7 {
		t.Fatalf("events = %d, want 7", len(events))
	}
	for i, want := range [][2]int64{{7, 1}, {5, 2}} {
		if events[i].BeforeBytes == nil || *events[i].BeforeBytes != want[0] || events[i].AfterBytes == nil || *events[i].AfterBytes != want[1] || events[i].RetainedRatio == nil || *events[i].RetainedRatio != float64(want[1])/float64(want[0]) {
			t.Fatalf("event %d = %+v, want %v", i, events[i], want)
		}
	}
	if events[2].AfterBytes != nil || events[2].RetainedRatio != nil || events[3].BeforeBytes != nil || events[3].AfterBytes == nil || *events[3].AfterBytes != 8 || events[4].BeforeBytes == nil || *events[4].BeforeBytes != 8 || events[4].AfterBytes == nil || *events[4].AfterBytes != 0 || events[5].BeforeBytes != nil || events[5].AfterBytes == nil || events[6].BeforeBytes != nil || events[6].AfterBytes != nil {
		t.Fatalf("missing/opaque history or replacement reset lost: %+v", events)
	}
	if result.Summary.Tokens != (AuditTokens{InputTokens: 80, CacheReadTokens: 20, OutputTokens: 5}) || len(result.Refusals) != 1 {
		t.Fatalf("usage accounting changed: %+v / %+v", result.Summary.Tokens, result.Refusals)
	}
	var encoded bytes.Buffer
	if err := WriteAuditJSONL(&encoded, result); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(encoded.String(), `"compaction_events"`) || !strings.Contains(encoded.String(), `"missing_or_malformed_replacement_history"`) || strings.Contains(encoded.String(), "summary alone") || strings.Contains(encoded.String(), `"opaque"`) {
		t.Fatal("compaction receipt absent or contains private content")
	}
}
