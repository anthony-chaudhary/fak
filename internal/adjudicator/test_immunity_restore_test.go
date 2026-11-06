package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestImmunityAdmitsHeadRestore is the #14209 witness: restoring a test file to
// its committed HEAD content is not test tampering, so TEST_TAMPER_REFUSED must
// not fire; every other restore source still refuses.
// fak-test:runtime fast est=50ms lane=default
func TestImmunityAdmitsHeadRestore(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()
	meta := map[string]string{"lane": "implementation"}

	admitted := []string{
		"git restore --source=HEAD --staged --worktree -- internal/gateway/gateway_test.go",
		"git restore --source HEAD -- internal/gateway/gateway_test.go",
		"git restore -s HEAD internal/gateway/gateway_test.go",
		"git restore --staged --worktree -- internal/gateway/gateway_test.go",
		"git restore -SW -- 'internal/gateway/gateway_test.go' internal/gateway/testdata/x.json",
		"git checkout HEAD -- internal/gateway/gateway_test.go",
		"git checkout -q HEAD -- internal/gateway/gateway_test.go",
	}
	for _, cmd := range admitted {
		t.Run("admit/"+cmd, func(t *testing.T) {
			if !isHeadRestoreCommand(cmd) {
				t.Fatalf("isHeadRestoreCommand(%q) = false, want true", cmd)
			}
			call := inlineCallJSON(t, "Bash", map[string]any{"command": cmd})
			call.Meta = meta
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("got verdict %v (%s), want allow", v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}

	refused := []string{
		"git restore --source=HEAD~1 -- internal/gateway/gateway_test.go",
		"git restore --source=abc1234 -- internal/gateway/gateway_test.go",
		"git restore --source=other-branch -- internal/gateway/gateway_test.go",
		"git restore --source=stash@{0} -- internal/gateway/gateway_test.go",
		"git restore --source=HEAD --source=main -- internal/gateway/gateway_test.go",
		"git restore -- internal/gateway/gateway_test.go",
		"git restore --source=HEAD --patch -- internal/gateway/gateway_test.go",
		"git checkout other-branch -- internal/gateway/gateway_test.go",
		"git checkout -- internal/gateway/gateway_test.go",
		"git checkout HEAD~1 -- internal/gateway/gateway_test.go",
		"git checkout stash@{0} -- internal/gateway/gateway_test.go",
		"git checkout HEAD -- internal/gateway/gateway_test.go && echo x > internal/gateway/gateway_test.go",
		"git checkout HEAD -- internal/gateway/gateway_test.go; git apply p.diff",
		"git -C internal/gateway checkout HEAD -- gateway_test.go",
		"git apply p.diff -- internal/gateway/gateway_test.go",
		"git stash pop -- internal/gateway/gateway_test.go",
	}
	for _, cmd := range refused {
		t.Run("refuse/"+cmd, func(t *testing.T) {
			if isHeadRestoreCommand(cmd) {
				t.Fatalf("isHeadRestoreCommand(%q) = true, want false", cmd)
			}
			call := inlineCallJSON(t, "Bash", map[string]any{"command": cmd})
			call.Meta = meta
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictDeny {
				t.Fatalf("got verdict %v (%s), want deny", v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}

	t.Run("refuse/write tool is never a restore", func(t *testing.T) {
		call := inlineCallJSON(t, "Write", map[string]any{
			"filePath": "internal/gateway/gateway_test.go",
			"command":  "git checkout HEAD -- internal/gateway/gateway_test.go",
		})
		call.Meta = meta
		if v := a.Adjudicate(ctx, call); v.Reason != ReasonTestTamperRefused {
			t.Fatalf("got %s, want TEST_TAMPER_REFUSED", abi.ReasonName(v.Reason))
		}
	})
}
