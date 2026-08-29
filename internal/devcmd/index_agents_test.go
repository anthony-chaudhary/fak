package devcmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentsindex"
)

func writeAgentsCLIRepo(t *testing.T) string {
	t.Helper()
	root := writeIndexRepo(t)
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("AGENTS.md", "# root\n\n## Root rules\nroot body\n")
	write("nested/AGENTS.md", "canonical\n")
	write("nested/AGENTS.override.md", "override\n")
	write("nested/FIRST.md", "first fallback\n")
	write("nested/deeper/SECOND.md", "second fallback\n")
	write("nested/deeper/work.go", "package work\n")
	return root
}

func TestIndexAgentsForJSONAndDeterminism(t *testing.T) {
	root := writeAgentsCLIRepo(t)
	args := []string{"agents", "--root", root, "--for", filepath.FromSlash("nested/deeper/work.go"),
		"--fallback", "FIRST.md", "--fallback", "SECOND.md", "--max-bytes", "4096", "--trust", "--json"}
	var first, second, errb bytes.Buffer
	if rc := RunIndex(&first, &errb, args); rc != 0 {
		t.Fatalf("first rc=%d stderr=%s output=%s", rc, errb.String(), first.String())
	}
	errFirst := errb.String()
	errb.Reset()
	if rc := RunIndex(&second, &errb, args); rc != 0 {
		t.Fatalf("second rc=%d stderr=%s output=%s", rc, errb.String(), second.String())
	}
	if first.String() != second.String() || errFirst != errb.String() {
		t.Fatalf("CLI is nondeterministic:\nfirst=%s\nsecond=%s", first.String(), second.String())
	}
	var got agentsindex.EffectiveResult
	if err := json.Unmarshal(first.Bytes(), &got); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, first.String())
	}
	if got.Status != agentsindex.StatusComplete || got.Target != "nested/deeper/work.go" || got.EffectiveSHA256 == "" {
		t.Fatalf("typed result=%+v", got)
	}
	if got.Instructions != "# root\n\n## Root rules\nroot body\noverride\nsecond fallback\n" {
		t.Fatalf("instructions=%q", got.Instructions)
	}
	if len(got.Sources) != 5 || got.Sources[0].Span == nil { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture declares exactly five indexed agent sources and verifies the first source span
		t.Fatalf("missing precedence/span provenance: %+v", got.Sources)
	}
}

func TestIndexAgentsForNonCompleteStillEmitsJSON(t *testing.T) {
	root := writeAgentsCLIRepo(t)
	for _, tc := range []struct {
		name string
		args []string
		want agentsindex.ResolutionStatus
	}{
		{name: "untrusted", args: []string{"--for", "nested", "--json"}, want: agentsindex.StatusUntrusted},
		{name: "truncated", args: []string{"--for", "nested", "--trust", "--max-bytes", "1", "--json"}, want: agentsindex.StatusTruncated},
		{name: "unknown", args: []string{"--for", "docs", "--trust", "--json"}, want: agentsindex.StatusComplete},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := append([]string{"agents", "--root", root}, tc.args...)
			var out, errb bytes.Buffer
			rc := RunIndex(&out, &errb, argv)
			if tc.want == agentsindex.StatusComplete {
				if rc != 0 {
					t.Fatalf("rc=%d stderr=%s", rc, errb.String())
				}
			} else if rc == 0 {
				t.Fatalf("non-complete result returned success: %s", out.String())
			}
			var got agentsindex.EffectiveResult
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatalf("JSON was not emitted: %v, stderr=%s output=%s", err, errb.String(), out.String())
			}
			if got.Status != tc.want {
				t.Fatalf("status=%s want=%s result=%+v", got.Status, tc.want, got)
			}
			if got.Status != agentsindex.StatusComplete && got.Instructions != "" {
				t.Fatalf("non-complete result exposed injection: %+v", got)
			}
		})
	}
}

func TestIndexAgentsForUnknownStillEmitsJSON(t *testing.T) {
	root := writeIndexRepo(t) // deliberately has no instruction file
	var out, errb bytes.Buffer
	if rc := RunIndex(&out, &errb, []string{"agents", "--root", root, "--for", "docs", "--trust", "--json"}); rc == 0 {
		t.Fatalf("unknown result returned success: %s", out.String())
	}
	var got agentsindex.EffectiveResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("unknown JSON missing: %v stderr=%s", err, errb.String())
	}
	if got.Status != agentsindex.StatusUnknown || got.Instructions != "" {
		t.Fatalf("unknown result=%+v", got)
	}
}

func TestIndexAgentsForRejectsLegacySelectors(t *testing.T) {
	root := writeAgentsCLIRepo(t)
	for _, extra := range [][]string{{"query"}, {"--section", "root-rules"}, {"--full"}, {"--write-resident"}} {
		argv := append([]string{"agents", "--root", root, "--for", "nested", "--trust"}, extra...)
		var out, errb bytes.Buffer
		if rc := RunIndex(&out, &errb, argv); rc != 2 {
			t.Fatalf("args=%v rc=%d stderr=%s", extra, rc, errb.String())
		}
		if !strings.Contains(errb.String(), "incompatible") {
			t.Fatalf("args=%v missing incompatibility diagnostic: %s", extra, errb.String())
		}
	}
}

func TestIndexAgentsWithoutForRetainsLegacyBytes(t *testing.T) {
	root := writeAgentsCLIRepo(t)
	doc, err := agentsindex.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	legacyCases := []struct {
		args []string
		want string
	}{
		{args: []string{"agents", "--root", root}, want: doc.RenderTOC()},
		{args: []string{"agents", "--root", root, "--full"}, want: string(doc.Raw)},
	}
	for _, tc := range legacyCases {
		var out, errb bytes.Buffer
		if rc := RunIndex(&out, &errb, tc.args); rc != 0 {
			t.Fatalf("args=%v rc=%d stderr=%s", tc.args, rc, errb.String())
		}
		if out.String() != tc.want {
			t.Fatalf("args=%v legacy bytes changed:\ngot=%q\nwant=%q", tc.args, out.String(), tc.want)
		}
	}
}

func TestIndexAgentsForRelativeAndAbsoluteTargetsMatch(t *testing.T) {
	root := writeAgentsCLIRepo(t)
	absTarget := filepath.Join(root, "nested", "deeper")
	outputs := make([]agentsindex.EffectiveResult, 0, 2)
	for _, target := range []string{filepath.FromSlash("nested/deeper"), absTarget} {
		var out, errb bytes.Buffer
		if rc := RunIndex(&out, &errb, []string{"agents", "--root", root, "--for", target, "--fallback", "SECOND.md", "--trust", "--json"}); rc != 0 {
			t.Fatalf("target=%q rc=%d stderr=%s", target, rc, errb.String())
		}
		var result agentsindex.EffectiveResult
		if err := json.Unmarshal(out.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, result)
	}
	if !reflect.DeepEqual(outputs[0], outputs[1]) {
		t.Fatalf("relative and absolute targets differ:\nrel=%+v\nabs=%+v", outputs[0], outputs[1])
	}
}

func TestIndexAgentsCommittedReceipt(t *testing.T) {
	fixture := filepath.Join("..", "agentsindex", "testdata", "effective")
	receiptBytes, err := os.ReadFile(filepath.Join(fixture, "receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		Schema    string   `json:"schema"`
		Scenarios []string `json:"scenarios"`
		Runs      []struct {
			SHA256 string `json:"sha256"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(receiptBytes, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != "fak-agents-effective-receipt/1" || len(receipt.Runs) != 2 || receipt.Runs[0].SHA256 != receipt.Runs[1].SHA256 {
		t.Fatalf("invalid deterministic receipt: %+v", receipt)
	}
	wantScenarios := []string{"root", "nested", "override", "fallback", "deletion-promotion", "cwd-change", "untrusted", "outside-root", "truncation"}
	if !reflect.DeepEqual(receipt.Scenarios, wantScenarios) {
		t.Fatalf("receipt scenarios=%v want=%v", receipt.Scenarios, wantScenarios)
	}
	root, err := filepath.Abs(filepath.Join(fixture, "repo"))
	if err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	args := []string{"agents", "--root", root, "--for", "nested/deeper/work.go", "--fallback", "FALLBACK.md", "--max-bytes", "4096", "--trust", "--json"}
	if rc := RunIndex(&out, &errb, args); rc != 0 {
		t.Fatalf("receipt replay rc=%d stderr=%s", rc, errb.String())
	}
	h := sha256.Sum256(out.Bytes())
	if got := fmt.Sprintf("%x", h); got != receipt.Runs[0].SHA256 {
		t.Fatalf("receipt drift: got=%s want=%s\n%s", got, receipt.Runs[0].SHA256, out.String())
	}
}
