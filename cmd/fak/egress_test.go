package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestEgressCheckBlocksMetadataURL is the headline CLI witness: `fak egress check
// --url <metadata>` runs the REAL kernel floor and reports a blocked destination
// (exit 1, EGRESS_BLOCK) — the deterministic, no-GPU proof that a guarded session on
// a VM refuses the instance-credential SSRF.
func TestEgressCheckBlocksMetadataURL(t *testing.T) {
	var out, errBuf bytes.Buffer
	code := runEgressCheck([]string{"--url", "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}, &out, &errBuf)
	if code != 1 {
		t.Fatalf("metadata url exit=%d, want 1 (blocked); out=%q err=%q", code, out.String(), errBuf.String())
	}
	if !strings.Contains(out.String(), "EGRESS_BLOCK") {
		t.Fatalf("want EGRESS_BLOCK in output, got %q", out.String())
	}
}

// TestEgressCheckAllowsPublicURL proves the negative space at the CLI: a public
// provider URL is not egress-blocked (exit 0).
func TestEgressCheckAllowsPublicURL(t *testing.T) {
	var out, errBuf bytes.Buffer
	code := runEgressCheck([]string{"--url", "https://api.anthropic.com/v1/messages"}, &out, &errBuf)
	if code != 0 {
		t.Fatalf("public url exit=%d, want 0 (allowed); out=%q", code, out.String())
	}
	if strings.Contains(out.String(), "EGRESS_BLOCK") {
		t.Fatalf("public url must not be egress-blocked, got %q", out.String())
	}
}

// TestEgressCheckHostClassifier covers the pure --host classifier path (no tool call).
func TestEgressCheckHostClassifier(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runEgressCheck([]string{"--host", "169.254.169.254"}, &out, &errBuf); code != 1 {
		t.Fatalf("metadata host exit=%d, want 1; out=%q", code, out.String())
	}
	if !strings.Contains(out.String(), "BLOCK") {
		t.Fatalf("want BLOCK for metadata host, got %q", out.String())
	}
	out.Reset()
	if code := runEgressCheck([]string{"--host", "api.openai.com"}, &out, &errBuf); code != 0 {
		t.Fatalf("public host exit=%d, want 0; out=%q", code, out.String())
	}
}

// TestEgressCheckShellCommand proves the CLI reaches the shell-command scan path.
func TestEgressCheckShellCommand(t *testing.T) {
	var out, errBuf bytes.Buffer
	code := runEgressCheck([]string{"--command", "curl -s http://metadata.google.internal/computeMetadata/v1/"}, &out, &errBuf)
	if code != 1 {
		t.Fatalf("metadata curl exit=%d, want 1; out=%q", code, out.String())
	}
}

// TestEgressCheckUsageError pins the usage contract: no destination flag is an exit-2
// usage error, and mutually-exclusive sources are rejected.
func TestEgressCheckUsageError(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runEgressCheck(nil, &out, &errBuf); code != 2 {
		t.Fatalf("no-args exit=%d, want 2", code)
	}
	if code := runEgressCheck([]string{"--url", "http://x", "--command", "curl x"}, &out, &errBuf); code != 2 {
		t.Fatalf("two-source exit=%d, want 2", code)
	}
}
