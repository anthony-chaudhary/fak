package adjudicator

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestImmunityAdmitsAttestedWorkerWorktreeRegressionTest covers the #11471 false
// positive: an explicitly assigned worker worktree (host-installed DevEditAttestation
// backed by a live exclusive lease whose tree names the file) required to author a
// fail-before/pass-after regression test must be able to edit that EXISTING test file
// even though the session lane is an implementation lane. The authorship exemption
// only covers brand-new *_test.go files, so without the attestation route the gate
// refuses the exact assignment the orchestrator gave the worker.
func TestImmunityAdmitsAttestedWorkerWorktreeRegressionTest(t *testing.T) {
	wt := t.TempDir()
	// The regression test already exists on disk (fail-before phase): editing it is
	// the tamper shape the gate exists for, UNLESS the target is explicitly assigned.
	if err := os.WriteFile(filepath.Join(wt, "lease_test.go"), []byte("package wt\n"), 0o644); err != nil {
		t.Fatalf("seed regression test: %v", err)
	}
	live := true
	a := New(testImmunityBasePolicy())
	a.SetDevEditAttestation(&DevEditAttestation{
		TraceID: "trace-11471", Worktree: wt, Issue: "11448", Lane: "issue-11448", Holder: "worker-11448", LaneOwnerPID: 4242,
		Paths: []string{"lease_test.go", "internal/wt/impl.go"},
		Verify: func(context.Context, DevEditAttestation) error {
			if !live {
				return errors.New("released")
			}
			return nil
		},
	})

	attestedEdit := func(path, trace string) *abi.ToolCall {
		c := inlineCall("Edit", `{"filePath":"`+filepath.ToSlash(path)+`"}`)
		c.TraceID = trace
		c.Meta = map[string]string{"lane": "implementation"}
		return c
	}

	// The false positive: explicitly assigned regression test in the lease tree.
	if v := a.Adjudicate(context.Background(), attestedEdit("lease_test.go", "trace-11471")); v.Kind != abi.VerdictAllow {
		t.Fatalf("attested worker-worktree regression test edit: got %v/%s, want ALLOW", v.Kind, abi.ReasonName(v.Reason))
	}

	// Fail-closed siblings: the attestation route must not become a test-tamper bypass.
	denied := map[string]*abi.ToolCall{
		"unassigned test file":   attestedEdit("other_test.go", "trace-11471"),
		"wrong trace":            attestedEdit("lease_test.go", "trace-other"),
		"empty trace":            attestedEdit("lease_test.go", ""),
		"absolute outside lease": attestedEdit(filepath.Join(filepath.Dir(wt), "outside", "lease_test.go"), "trace-11471"),
	}
	for name, c := range denied {
		if v := a.Adjudicate(context.Background(), c); v.Kind == abi.VerdictAllow {
			t.Errorf("%s: unexpectedly allowed", name)
		}
	}
	live = false
	if v := a.Adjudicate(context.Background(), attestedEdit("lease_test.go", "trace-11471")); v.Kind == abi.VerdictAllow {
		t.Errorf("released lease: unexpectedly allowed regression test edit")
	}
}
