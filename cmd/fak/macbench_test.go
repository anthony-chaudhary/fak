package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMacBenchJSONDoesNotLeakBearer(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			_, _ = w.Write([]byte(`{"ok":true}`))
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer super-secret-test-key" {
			t.Fatalf("Authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}`))
	}))
	defer ts.Close()

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"decode-longgen",
		"--gateway", ts.URL,
		"--decode-tokens", "8",
		"--gateway-key-file", "",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runMacBench code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"schema": "fak.macbench.result.v1"`) || !strings.Contains(out, "tok/s") {
		t.Fatalf("unexpected output:\n%s", out)
	}
	if strings.Contains(out, "super-secret-test-key") || strings.Contains(stderr.String(), "super-secret-test-key") {
		t.Fatalf("leaked bearer:\nstdout=%s\nstderr=%s", out, stderr.String())
	}
}

func TestParseIntCSVRejectsBadValues(t *testing.T) {
	if _, err := parseIntCSV("128, nope"); err == nil {
		t.Fatal("expected parse error")
	}
}
