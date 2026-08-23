package adjudicator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestMysteryFreeAdjustmentRefusalsNameRecovery(t *testing.T) {
	policy := mysteryFreePolicy(false)
	monitor := New(policy)
	ctx := context.Background()

	cases := []struct {
		name       string
		tool       string
		wantReason abi.ReasonCode
		recovery   string
	}{
		{name: "explicit policy block", tool: "refund_payment", wantReason: abi.ReasonPolicyBlock, recovery: "deny.refund_payment"},
		{name: "fail-closed unknown tool", tool: "mystery_action", wantReason: abi.ReasonDefaultDeny, recovery: "allow"},
	}

	atlas := readMysteryFreeRecoveryDoc(t, "docs", "fak", "mystery-free-adjustment-atlas.md")
	learningPath := readMysteryFreeRecoveryDoc(t, "LEARNING-PATH.md")
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			verdict := monitor.Adjudicate(ctx, inlineCall(tc.tool, `{}`))
			if verdict.Kind != abi.VerdictDeny || verdict.Reason != tc.wantReason {
				t.Fatalf("verdict = %v/%s, want DENY/%s", verdict.Kind, abi.ReasonName(verdict.Reason), abi.ReasonName(tc.wantReason))
			}
			reason := abi.ReasonName(tc.wantReason)
			for name, doc := range map[string]string{"learning path": learningPath, "adjustment atlas": atlas} {
				if !strings.Contains(doc, reason) || !strings.Contains(doc, tc.recovery) {
					t.Errorf("%s does not join refusal %s to recovery %q", name, reason, tc.recovery)
				}
			}
		})
	}
}

func readMysteryFreeRecoveryDoc(t *testing.T, parts ...string) string {
	t.Helper()
	path := filepath.Join(append([]string{"..", ".."}, parts...)...)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}
