package issueorchestrator

import "testing"

// TestDecodeIssuesStripsUTF8BOM: on Windows hosts a snapshot written with
// PowerShell Set-Content -Encoding utf8 starts with a UTF-8 BOM, and
// issue-orchestrator refused it with "unable to parse issue payload".
func TestDecodeIssuesStripsUTF8BOM(t *testing.T) {
	t.Parallel()
	body := []byte(`[{"number":7,"title":"fix(x): y","body":"b","labels":[]}]`)
	issues, err := DecodeIssues(append([]byte{0xEF, 0xBB, 0xBF}, body...))
	if err != nil {
		t.Fatalf("BOM-prefixed payload refused: %v", err)
	}
	if len(issues) != 1 || issues[0].Title != "fix(x): y" {
		t.Fatalf("issues = %+v", issues)
	}
}
