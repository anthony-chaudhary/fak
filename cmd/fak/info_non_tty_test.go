package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunInfoNonTTYDefaultsToOneSnapshot(t *testing.T) {
	srv := debugVarsStub(t)
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL})
	if code != 0 {
		t.Fatalf("fak info non-TTY exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, "fak info . ") || strings.Contains(out, "Ctrl-C to stop") {
		t.Fatalf("non-TTY stdout entered the watch loop:\n%s", out)
	}
	if strings.Contains(out, "\x1b[") {
		t.Fatalf("non-TTY stdout contained cursor-control escapes: %q", out)
	}
	if got := strings.Count(strings.TrimRight(out, "\n"), "\n"); got != 0 {
		t.Fatalf("non-TTY stdout rendered more than one snapshot line:\n%s", out)
	}
}

func TestRunInfoNonTTYExplicitWatchOptsBackIn(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{
		"--gateway-url", "http://127.0.0.1:0",
		"--watch", "--interval", "1ms", "--max-idle", "1ms",
	})
	if code != 0 {
		t.Fatalf("fak info --watch non-TTY exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "fak info") || !strings.Contains(out, "every 1ms") {
		t.Fatalf("explicit --watch did not enter the watch renderer:\n%s", out)
	}
	if !strings.Contains(out, "idle backstop") {
		t.Fatalf("bounded explicit watch did not reach its closing line:\n%s", out)
	}
}
