package validate

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
)

// A tip extraction stopped by the committed-tree budget while the validation
// deadline is still live must name that budget's knob: the validation timeout
// was not the binding bound, so pointing an operator at it cannot cure the run.
func TestExtractTipBudgetTimeoutNamesExtractBudgetEnv(t *testing.T) {
	original := extractCommittedTip
	extractCommittedTip = func(string, string) (string, error) {
		return "", &committedtree.BudgetError{Budget: 2 * time.Minute}
	}
	t.Cleanup(func() { extractCommittedTip = original })

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	res := &validateResult{
		Schema: "fak-validate/1", OK: true, TimeoutMS: (10 * time.Minute).Milliseconds(),
		Phases: []validatePhase{}, SkippedPhases: []string{}, Failures: []ciPreflightFailure{},
	}
	recorder := &validateRecorder{ctx: ctx, stderr: io.Discard, res: res}
	var stdout bytes.Buffer
	_, code, stopped := prepareValidateWorkspace(validateWorkspaceRequest{
		stdout: &stdout, stderr: io.Discard, ctx: ctx, result: res, recorder: recorder,
		root: t.TempDir(), tip: "0000000000000000000000000000000000000000", asJSON: true,
	})
	if !stopped || code != 1 {
		t.Fatalf("prepareValidateWorkspace = code %d stopped %v, want a stopped timeout with code 1", code, stopped)
	}
	if !res.TimedOut || res.Reason != "TIMEOUT" {
		t.Fatalf("result timed_out=%v reason=%q, want a typed TIMEOUT", res.TimedOut, res.Reason)
	}
	if len(res.Failures) != 1 || res.Failures[0].Step != "extract_tip" {
		t.Fatalf("failures = %+v, want one extract_tip failure", res.Failures)
	}
	detail := res.Failures[0].Detail
	for _, want := range []string{"FAK_COMMITTEDTREE_EXTRACT_BUDGET", "2m0s", "10m0s"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("extract_tip detail = %q, want it to contain %q", detail, want)
		}
	}
	if strings.Contains(detail, "validation timeout after") {
		t.Fatalf("extract_tip detail = %q, must not blame the validation timeout", detail)
	}
	var emitted struct {
		Failures []ciPreflightFailure `json:"failures"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &emitted); err != nil {
		t.Fatalf("emitted result is not JSON: %v\n%s", err, stdout.String())
	}
	if len(emitted.Failures) != 1 || !strings.Contains(emitted.Failures[0].Detail, "FAK_COMMITTEDTREE_EXTRACT_BUDGET") {
		t.Fatalf("emitted failures = %+v, want the extract budget knob in the operator-visible detail", emitted.Failures)
	}
}

// When the validation deadline itself expires, the validation timeout stays
// the named bound.
func TestExtractTipValidationDeadlineKeepsValidationTimeoutDetail(t *testing.T) {
	original := extractCommittedTip
	release := make(chan struct{})
	extractCommittedTip = func(string, string) (string, error) {
		<-release
		return "", context.Canceled
	}
	t.Cleanup(func() {
		close(release)
		extractCommittedTip = original
	})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	res := &validateResult{
		Schema: "fak-validate/1", OK: true, TimeoutMS: 20,
		Phases: []validatePhase{}, SkippedPhases: []string{}, Failures: []ciPreflightFailure{},
	}
	recorder := &validateRecorder{ctx: ctx, stderr: io.Discard, res: res}
	_, code, stopped := prepareValidateWorkspace(validateWorkspaceRequest{
		stdout: io.Discard, stderr: io.Discard, ctx: ctx, result: res, recorder: recorder,
		root: t.TempDir(), tip: "0000000000000000000000000000000000000000", asJSON: true,
	})
	if !stopped || code != 1 || len(res.Failures) != 1 {
		t.Fatalf("prepareValidateWorkspace = code %d stopped %v failures %+v, want one timeout failure", code, stopped, res.Failures)
	}
	if got := res.Failures[0].Detail; got != "validation timeout after 20ms" {
		t.Fatalf("extract_tip detail = %q, want the validation timeout named", got)
	}
}
