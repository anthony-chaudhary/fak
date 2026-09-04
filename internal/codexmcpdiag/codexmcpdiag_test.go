package codexmcpdiag

import "testing"

func TestClassifyCapturedFourServerFalseWarning(t *testing.T) {
	names := []string{"codex_apps", "dos", "fak", "openaiDeveloperDocs"}
	var es []Event
	for _, n := range names {
		es = append(es, Event{Level: "INFO", Target: "codex_core::mcp", Body: "MCP client for `" + n + "` initialized"})
	}
	r := Classify(names, es)
	if r.Verdict != VerdictFalseNegative {
		t.Fatalf("got %s: %#v", r.Verdict, r)
	}
}
func TestClassifyFailureCancellationAndMissing(t *testing.T) {
	cases := []struct{ body, level, want string }{{"fak failed to initialize", "ERROR", VerdictServerFailure}, {"fak startup cancelled during runtime refresh", "WARN", VerdictRuntimeCancellation}, {"unrelated", "INFO", VerdictInsufficient}}
	for _, tc := range cases {
		if got := Classify([]string{"fak"}, []Event{{Level: tc.level, Body: tc.body}}).Verdict; got != tc.want {
			t.Errorf("%q: got %s want %s", tc.body, got, tc.want)
		}
	}
}

func TestClassifyEmptyNames(t *testing.T) {
	r := Classify(nil, nil)
	if r.Verdict != VerdictInsufficient {
		t.Fatalf("expected VerdictInsufficient for empty names, got %s", r.Verdict)
	}
	if len(r.Servers) != 0 {
		t.Fatalf("expected 0 servers, got %d", len(r.Servers))
	}
}
