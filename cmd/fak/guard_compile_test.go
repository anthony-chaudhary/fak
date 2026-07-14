package main

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type guardCompileDoerFunc func(*http.Request) (*http.Response, error)

func (f guardCompileDoerFunc) Do(r *http.Request) (*http.Response, error) { return f(r) }

func TestRunGuardCompileProposesWithoutApplying(t *testing.T) {
	dir := t.TempDir()
	policyPath := filepath.Join(dir, "policy.json")
	before := []byte(`{"version":"fak-policy/v1","posture":"fail_closed","allow":["Bash"],"arg_rules":[]}`)
	if err := os.WriteFile(policyPath, before, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GUARD_COMPILE_TEST_KEY", "test-key")
	requests := 0
	doer := guardCompileDoerFunc(func(req *http.Request) (*http.Response, error) {
		requests++
		if got := req.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		body, _ := io.ReadAll(req.Body)
		if !bytes.Contains(body, []byte(`"response_format":{"type":"json_object"}`)) {
			t.Errorf("request did not require JSON: %s", body)
		}
		content := `{"deny_regex":"rm\\s+-rf\\s+\\.\\./tools(?:\\s|$)","reason":"POLICY_BLOCK","severity":"block","fix":"use repository scratch"}`
		response := `{"choices":[{"message":{"content":` + quoteJSON(content) + `}}]}`
		return &http.Response{
			StatusCode: http.StatusOK, Status: "200 OK",
			Body: io.NopCloser(strings.NewReader(response)),
		}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runGuardCompileWithDoer(&stdout, &stderr, []string{
		"--transcript", "agent invoked a destructive command",
		"--intent", "block deletion outside the repository",
		"--policy", policyPath,
		"--endpoint", "https://model.invalid/v1/chat/completions",
		"--api-key-env", "GUARD_COMPILE_TEST_KEY",
	}, doer)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if requests != 1 {
		t.Fatalf("HTTP requests = %d, want exactly 1", requests)
	}
	if !strings.Contains(stdout.String(), "proposed; not applied") || !strings.Contains(stdout.String(), "policy was not applied") {
		t.Fatalf("output lacks review-only warning: %s", stdout.String())
	}
	after, err := os.ReadFile(policyPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) {
		t.Fatalf("policy was modified: before=%s after=%s", before, after)
	}
}

func TestGuardCompileExtractorDoesNotRetryHTTPFailure(t *testing.T) {
	requests := 0
	extractor := &guardCompileOpenAIExtractor{
		doer: guardCompileDoerFunc(func(*http.Request) (*http.Response, error) {
			requests++
			return &http.Response{
				StatusCode: http.StatusTooManyRequests, Status: "429 Too Many Requests",
				Body: io.NopCloser(strings.NewReader("rate limited")),
			}, nil
		}),
		endpoint: "https://model.invalid/v1/chat/completions", model: "test", apiKey: "key",
	}
	_, err := extractor.Extract("prompt")
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("error = %v, want loud 429", err)
	}
	if requests != 1 {
		t.Fatalf("HTTP requests = %d, want no retry", requests)
	}
}

func quoteJSON(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '\\', '"':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}
