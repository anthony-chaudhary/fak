package main

import (
	"io"
	"os"
	"strings"
	"testing"
)

// captureStdoutSafe redirects os.Stdout to a pipe and drains it CONCURRENTLY with fn, unlike
// the package's captureStdout (usagelog_record_test.go), which reads only after fn returns.
// This preview's --format json output is a few KB, which on this environment's os.Pipe
// blocks the writer once the (small) pipe buffer fills if nothing is reading yet — a
// pre-existing latent deadlock in the read-after-write pattern, not a bug in the command
// itself. Draining on a background goroutine avoids it regardless of pipe buffer size.
func captureStdoutSafe(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	outCh := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		outCh <- string(b)
	}()

	fn()
	_ = w.Close()
	return <-outCh
}

// TestContextPlanPreviewCLIRendersAllRegionsText is the CLI-facing witness for #1574: `fak
// debug --cmd context-plan-preview` must render the five product-facing regions (pinned,
// recent, deep, elided, query-needed) WITHOUT ingesting or attaching any core image — the
// dry-run report for a long managed-context run before it starts.
func TestContextPlanPreviewCLIRendersAllRegionsText(t *testing.T) {
	out := captureStdoutSafe(t, func() {
		cmdDebugContextPlanPreview([]string{"--intent", "legacy queue migration", "--budget-tokens", "300"})
	})
	for _, want := range []string{"PINNED", "RECENT", "DEEP", "ELIDED", "QUERY-NEEDED", "faithful=true"} {
		if !strings.Contains(out, want) {
			t.Errorf("context-plan-preview text output missing %q:\n%s", want, out)
		}
	}
}

// TestContextPlanPreviewCLIJSON checks the --format json path round-trips the same five
// regions machine-readably, so a scripted caller can gate on them without scraping text.
func TestContextPlanPreviewCLIJSON(t *testing.T) {
	out := captureStdoutSafe(t, func() {
		cmdDebugContextPlanPreview([]string{"--format", "json", "--budget-tokens", "300"})
	})
	for _, want := range []string{`"pinned"`, `"recent"`, `"deep"`, `"elided"`, `"query_needed"`, `"faithful": true`} {
		if !strings.Contains(out, want) {
			t.Errorf("context-plan-preview json output missing %q:\n%s", want, out)
		}
	}
}

// TestContextPlanPreviewCLIMarkdown checks the --format md path renders the shareable
// report form with a heading per region.
func TestContextPlanPreviewCLIMarkdown(t *testing.T) {
	out := captureStdoutSafe(t, func() {
		cmdDebugContextPlanPreview([]string{"--format", "md"})
	})
	for _, want := range []string{"# Context plan preview", "## Pinned", "## Recent", "## Deep", "## Elided", "## Query-needed"} {
		if !strings.Contains(out, want) {
			t.Errorf("context-plan-preview markdown output missing %q:\n%s", want, out)
		}
	}
}

// TestContextPlanPreviewCLIDispatchesFromDebugCmd checks the top-level `fak debug --cmd
// context-plan-preview` dispatch branches out before any cdb core-image ingest/attach is
// attempted (no --session needed, no cdb-image directory touched).
func TestContextPlanPreviewCLIDispatchesFromDebugCmd(t *testing.T) {
	dir := t.TempDir()
	imageDir := dir + "/should-not-be-created"
	out := captureStdoutSafe(t, func() {
		cmdDebug([]string{"--cmd", "context-plan-preview", "--dir", imageDir})
	})
	if !strings.Contains(out, "PINNED") {
		t.Fatalf("expected the preview report, got:\n%s", out)
	}
	if _, err := os.Stat(imageDir); err == nil {
		t.Fatalf("context-plan-preview must not create a core image directory at %s", imageDir)
	}
}
