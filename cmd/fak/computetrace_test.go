package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func TestComputeTraceCaptureAndSummary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.json")
	var out, errOut bytes.Buffer
	if code := runComputeTrace(&out, &errOut, []string{"capture", "--out", path, "--limit", "1", "--run", "r", "--request", "q"}); code != 0 {
		t.Fatalf("capture code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"summary", "--in", path}); code != 0 {
		t.Fatalf("summary code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "schema=fak.compute_trace.v1 events=1 dropped=0") {
		t.Fatalf("summary=%q", out.String())
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/computetrace*.go
// fak-test:runtime fast est=10ms lane=default
func TestComputeTraceSummarizeActivationCapture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "step.json")
	summary := func(a computetrace.Artifact) (int, string) {
		t.Helper()
		writeComputeTraceStepFixture(t, path, a)
		var out, stderr bytes.Buffer
		code := runComputeTrace(&out, &stderr, []string{"summary", "--in", path})
		return code, out.String()
	}
	_, actual := computeTraceStepArtifacts()
	code, text := summary(actual)
	if code != 0 {
		t.Fatalf("valid summary exit=%d", code)
	}
	for _, field := range []string{
		"activation_capture_id=step-current", "kind=incremental_step", "outcome=committed",
		"token=2", "expected_layers=2", "expected_stages=q_latent,kv_latent,attention_projected,moe_sum,residual_streams",
		"declared_stages=5", "capture_validation=valid", "capture_metadata_truncated=false",
	} {
		if !strings.Contains(text, field) {
			t.Fatalf("summary missing %q: %s", field, text)
		}
	}
	t.Run("legacy", func(t *testing.T) {
		_, a := computeTraceStepArtifacts()
		a.ActivationCapture = nil
		code, text := summary(a)
		if code != 0 || !strings.Contains(text, "capture_validation=legacy_numerical") {
			t.Fatal("summary did not label legacy numerical evidence")
		}
	})
	t.Run("aborted", func(t *testing.T) {
		_, a := computeTraceStepArtifacts()
		a.ActivationCapture.Outcome = computetrace.ActivationCaptureAborted
		code, text := summary(a)
		if code == 0 || !strings.Contains(text, "outcome=aborted") || !strings.Contains(text, "capture_validation=invalid") {
			t.Fatal("summary accepted or concealed an aborted capture")
		}
	})
	t.Run("bounded_invalid_metadata", func(t *testing.T) {
		_, a := computeTraceStepArtifacts()
		a.ActivationCapture.ID = strings.Repeat("x", 4096)
		a.ActivationCapture.Kind = strings.Repeat("k", 4096)
		a.ActivationCapture.ExpectedStages = append(a.ActivationCapture.ExpectedStages, strings.Repeat("s", 4096))
		code, text := summary(a)
		if code == 0 || len(text) > 2000 || !strings.Contains(text, "declared_stages=6") || !strings.Contains(text, "capture_metadata_truncated=true") || !strings.Contains(text, "capture_validation=invalid") {
			t.Fatalf("unbounded or misleading metadata summary: exit=%d bytes=%d", code, len(text))
		}
	})
	t.Run("trailing_json_or_garbage", func(t *testing.T) {
		const valid = "{\"schema\":\"fak.compute_trace.v1\",\"events\":[]}"
		for _, suffix := range []string{"{}", "[]", "]", "}", "garbage"} {
			if err := os.WriteFile(path, []byte(valid+suffix), 0o644); err != nil {
				t.Fatal(err)
			}
			var out, stderr bytes.Buffer
			if code := runComputeTrace(&out, &stderr, []string{"summary", "--in", path}); code == 0 || out.Len() != 0 {
				t.Fatalf("summary accepted trailing suffix %q", suffix)
			}
		}
	})
}
