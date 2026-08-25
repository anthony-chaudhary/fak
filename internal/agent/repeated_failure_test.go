package agent

import "testing"

func TestCanonicalFailureKeyCanonicalizesArgs(t *testing.T) {
	first, failed := canonicalFailureKey("search", `{"query":"x","limit":2}`, `{"error":"timed out"}`)
	if !failed {
		t.Fatal("generic error was not recognized")
	}
	second, failed := canonicalFailureKey("search", ` { "limit": 2, "query": "x" } `, `{"error":"timed out"}`)
	if !failed {
		t.Fatal("generic error was not recognized with reordered args")
	}
	if first != second {
		t.Fatalf("reordered JSON args produced different keys:\n%q\n%q", first, second)
	}
}

func TestCanonicalFailureKeyRecognizesToolReceiptError(t *testing.T) {
	key, failed := canonicalFailureKey("write", `{}`, ToolReceipt{
		Status: ToolResultError,
		Reason: "POLICY_BLOCK",
		Detail: "denied",
	}.JSON())
	if !failed || key == "" {
		t.Fatalf("canonicalFailureKey() = (%q, %t), want nonempty key and failure", key, failed)
	}
}

func TestCanonicalFailureKeyRecognizesGenericError(t *testing.T) {
	key, failed := canonicalFailureKey("read", `{}`, `{"error":"not found"}`)
	if !failed || key == "" {
		t.Fatalf("canonicalFailureKey() = (%q, %t), want nonempty key and failure", key, failed)
	}
}

func TestCanonicalFailureKeyTreatsNonFailuresAsSuccess(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{name: "skipped receipt", content: ToolReceipt{Status: ToolResultSkipped, Reason: "NOT_SENT"}.JSON()},
		{name: "ok receipt", content: `{"status":"ok"}`},
		{name: "plain text", content: "completed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, failed := canonicalFailureKey("tool", `{}`, tt.content)
			if failed || key != "" {
				t.Fatalf("canonicalFailureKey() = (%q, %t), want empty key and success", key, failed)
			}
		})
	}
}

func TestCanonicalFailureKeyDistinguishesErrors(t *testing.T) {
	first, failed := canonicalFailureKey("read", `{}`, `{"error":"not found"}`)
	if !failed {
		t.Fatal("first generic error was not recognized")
	}
	second, failed := canonicalFailureKey("read", `{}`, `{"error":"permission denied"}`)
	if !failed {
		t.Fatal("second generic error was not recognized")
	}
	if first == second {
		t.Fatalf("different errors produced the same key %q", first)
	}
}
