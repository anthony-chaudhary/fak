package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

func TestOwnedMalformedWriteExplainsFieldWithoutMutation(t *testing.T) {
	for _, tc := range []struct{ name, args, want string }{
		{"missing_path", `{"content":"candidate-content-must-not-be-echoed"}`, "file_path"},
		{"unknown_version", `{"file_path":"value.txt","content":"candidate-content-must-not-be-echoed","mode":"overwrite","version":"fv1:invented"}`, `unknown field "version"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, catalog := armLiteralCASFixture(t)
			p := &scriptRecordingPlanner{turns: []*Completion{
				callTurn("observe", codetools.ToolRead, `{"file_path":"value.txt"}`),
				callTurn("invalid-write", codetools.ToolWrite, tc.args),
				{Message: Message{Content: "The malformed call was refused."}},
			}}
			metrics, err := RunArm(context.Background(), p, "Update value.txt using code tools.", true, 5, nil, WithToolCatalog(catalog))
			if err != nil {
				t.Fatal(err)
			}
			if len(p.seen) != 3 {
				t.Fatalf("planner turns = %d, want 3", len(p.seen))
			}
			rc := findReceipt(t, p.seen[2], "invalid-write")
			if rc.Status != ToolResultError || rc.Reason != "MALFORMED" || rc.Disposition != "RETRYABLE" {
				t.Fatalf("invalid write receipt = %+v", rc)
			}
			if !strings.Contains(rc.Detail, tc.want) {
				t.Fatalf("detail %q does not identify %q", rc.Detail, tc.want)
			}
			if strings.Contains(rc.Detail, "candidate-content-must-not-be-echoed") || strings.Contains(rc.Detail, "fv1:invented") {
				t.Fatalf("receipt leaked argument values: %q", rc.Detail)
			}
			if metrics.Denies != 1 {
				t.Fatalf("denies = %d, want 1", metrics.Denies)
			}
			data, err := os.ReadFile(filepath.Join(root, "value.txt"))
			if err != nil || string(data) != "old" {
				t.Fatalf("invalid mutation changed bytes: %q, %v", data, err)
			}
		})
	}
}

func TestOwnedCodingErrorDetailIsBoundedAndEscaped(t *testing.T) {
	detail := "unknown field \"line\\break\n雪\" <diagnostic>"
	result := &abi.Result{Meta: map[string]string{"reason": "MALFORMED", "disposition": "RETRYABLE"}}
	rc := denyToolReceipt(result, abi.Verdict{By: codetools.RungName, Meta: map[string]string{"detail": detail}})
	var decoded ToolReceipt
	if err := json.Unmarshal([]byte(rc.JSON()), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Detail != detail {
		t.Fatalf("detail round trip = %q, want %q", decoded.Detail, detail)
	}
	if strings.Contains(rc.JSON(), "\n") {
		t.Fatalf("raw newline in serialized receipt: %q", rc.JSON())
	}
	long := strings.Repeat("雪", 600)
	bounded := denyToolReceipt(result, abi.Verdict{By: codetools.RungName, Meta: map[string]string{"detail": long}})
	if !utf8.ValidString(bounded.Detail) || utf8.RuneCountInString(bounded.Detail) > 256 || !strings.HasSuffix(bounded.Detail, "…") {
		t.Fatalf("detail is not valid bounded diagnostic: runes=%d suffix=%v", utf8.RuneCountInString(bounded.Detail), strings.HasSuffix(bounded.Detail, "…"))
	}
	if !strings.HasPrefix(bounded.Detail, strings.Repeat("雪", 32)) {
		t.Fatal("diagnostic prefix lost")
	}
	unrelated := denyToolReceipt(result, abi.Verdict{By: "unrelated-policy", Meta: map[string]string{"detail": "private internal context"}})
	if unrelated.Detail != "" {
		t.Fatalf("unrelated verdict detail exposed: %q", unrelated.Detail)
	}
}
