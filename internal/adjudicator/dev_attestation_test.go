package adjudicator

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func attestedDevCall(path, trace string, meta map[string]string) *abi.ToolCall {
	c := inlineCall("write_file", `{"path":"`+filepath.ToSlash(path)+`"}`)
	c.TraceID = trace
	c.Meta = meta
	return c
}

func TestDevEditAttestationAllowsOnlyExactLiveOwnedPath(t *testing.T) {
	a := New(DevAgentPolicy())
	live := true
	wt := t.TempDir()
	a.SetDevEditAttestation(&DevEditAttestation{
		TraceID: "trace-1", Worktree: wt, Issue: "9850", Lane: "issue-9850", Holder: "worker-9850", LaneOwnerPID: 123,
		Paths: []string{"internal/adjudicator/dev_attestation.go"},
		Verify: func(context.Context, DevEditAttestation) error {
			if !live {
				return errors.New("released")
			}
			return nil
		},
	})
	if v := a.Adjudicate(context.Background(), attestedDevCall("internal/adjudicator/dev_attestation.go", "trace-1", nil)); v.Kind != abi.VerdictAllow {
		t.Fatalf("exact host-attested owned path was not allowed: %+v", v)
	}
	for name, c := range map[string]*abi.ToolCall{
		"missing trace":    attestedDevCall("internal/adjudicator/dev_attestation.go", "", nil),
		"wrong trace":      attestedDevCall("internal/adjudicator/dev_attestation.go", "trace-2", nil),
		"outside path":     attestedDevCall("internal/adjudicator/decide.go", "trace-1", nil),
		"forged meta":      attestedDevCall("internal/adjudicator/decide.go", "", map[string]string{"dev_attestation": "trace-1", "lane": "issue-9850"}),
		"outside worktree": attestedDevCall(filepath.Join(filepath.Dir(wt), "outside", "internal", "adjudicator", "dev_attestation.go"), "trace-1", nil),
	} {
		if v := a.Adjudicate(context.Background(), c); v.Kind == abi.VerdictAllow {
			t.Errorf("%s: unexpectedly allowed", name)
		}
	}
	live = false
	if v := a.Adjudicate(context.Background(), attestedDevCall("internal/adjudicator/dev_attestation.go", "trace-1", nil)); v.Reason != abi.ReasonSelfModify {
		t.Fatalf("released lease: got %v/%s, want SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestDevEditAttestationNeverAllowsRepositoryOrRuntimePolicyReplacement(t *testing.T) {
	a := New(DevAgentPolicy())
	a.SetDevEditAttestation(&DevEditAttestation{
		TraceID: "trace-1", Worktree: t.TempDir(), Issue: "9850", Lane: "issue-9850", Holder: "worker-9850", LaneOwnerPID: 123,
		Paths:      []string{".git/**", "internal/adjudicator/**", "internal/policy/**"},
		PolicyPath: "examples/dev-agent-policy.json",
		Verify:     func(context.Context, DevEditAttestation) error { return nil },
	})
	for _, path := range []string{".git/config", "examples/dev-agent-policy.json"} {
		if v := a.Adjudicate(context.Background(), attestedDevCall(path, "trace-1", nil)); v.Reason != abi.ReasonSelfModify {
			t.Errorf("%s: got %v/%s, want SELF_MODIFY", path, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
