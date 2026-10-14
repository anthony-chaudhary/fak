package benchcatalog

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestV41EPOwnerManualDiagnosticCatalog(t *testing.T) {
	b, ok := Get("v41-ep-owner")
	if !ok || b.Kind != KindVerb || b.Need != NeedWeights || !b.Manual || b.Level != LevelSmoke || b.Environment != nil || b.Offline() {
		t.Fatalf("diagnostic admission metadata changed: %+v found=%t", b, ok)
	}
	if !strings.Contains(b.Summary, "native execution is unavailable") || b.Run != "fak benchmarks run v41-ep-owner -- --validate --request request.json --rank 0" || b.Doc != "internal/localbench/v41_ep_owner_command.go" {
		t.Fatalf("diagnostic scope changed: %+v", b)
	}
	for _, want := range []string{"--validate", "--supervised-owner", "tools/v41epguard", "package initialization"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("missing boundary %q: %v", want, b.Flags)
		}
	}
	count := 0
	for _, entry := range All() {
		if entry.Name == b.Name {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("catalog contains %d owner entries", count)
	}
	for _, entry := range Offline() {
		if entry.Name == b.Name {
			t.Fatal("manual owner diagnostic entered unattended offline candidates")
		}
	}
}
