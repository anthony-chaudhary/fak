package workerworktree

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms
func TestPreparedDisambiguationRecheckHonorsTimeoutRecovery(t *testing.T) {
	oldHas := hasDisambiguationContract
	t.Cleanup(func() { hasDisambiguationContract = oldHas })
	binding := PreparedLandDisambiguation{
		Required: true, RootTreeSHA: strings.Repeat("1", 40), WorktreeTreeSHA: strings.Repeat("2", 40),
		CandidateTreeSHA: strings.Repeat("3", 40), AnalyzerIdentity: preparedDisambiguationAnalyzerIdentity(),
		Verdict: preparedDisambiguationInapplicable,
	}
	receipt := PreparedLandReceipt{TreeSHA: binding.CandidateTreeSHA, CandidatePaths: []string{"internal/example.go"}, Disambiguation: binding}
	git := func(repo string, args []string) (int, string) {
		if strings.Join(args, " ") != "rev-parse --verify HEAD^{tree}" {
			t.Fatalf("unexpected Git operation: %v", args)
		}
		if repo == "/worker" {
			return 0, binding.WorktreeTreeSHA
		}
		return 0, binding.RootTreeSHA
	}

	t.Run("typed deadline preserves cause and refusal", func(t *testing.T) {
		t.Setenv(DisambiguationTimeoutEnv, "300000")
		gitErr := errors.New("git ls-tree: killed")
		hasDisambiguationContract = func(ctx context.Context, _, _ string) (bool, error) {
			deadline, ok := ctx.Deadline()
			if remaining := time.Until(deadline); !ok || remaining <= 4*time.Minute || remaining > 5*time.Minute {
				t.Fatalf("probe deadline remaining=%s present=%t; want explicit five-minute budget", remaining, ok)
			}
			return false, errors.Join(gitErr, context.DeadlineExceeded)
		}
		applicable, err := preparedDisambiguationApplicable("/root", binding)
		wrapped := fmt.Errorf("bind disambiguation applicability: %w", err)
		if applicable || !errors.Is(wrapped, gitErr) || !errors.Is(wrapped, context.DeadlineExceeded) {
			t.Fatalf("probe lost command/deadline identity: applicable=%t err=%v", applicable, wrapped)
		}
		got := validatePreparedLandDisambiguation("/root", "/worker", receipt, git)
		if got.OK || got.Code != LandResultPreparedReprepare || !got.Preserved || got.Applied || got.Committed || got.Disambiguation == nil {
			t.Fatalf("prepared refusal contract changed: %+v", got)
		}
		w := got.Disambiguation
		d := w.Before.Diagnostic
		if d == nil || d.Code != DisambiguationTimeoutCode || !d.Recoverable || d.Witness != "prepared-applicability" || d.Subphase != "contract-probe" || d.TimeoutMS != 300000 {
			t.Fatalf("missing typed probe timeout: %+v", d)
		}
		if w.Timeout.RecoveryMode != disambiguationRecoveryExplicit || w.Timeout.EffectiveTimeoutMS != 300000 || !w.deadlineExceeded() {
			t.Fatalf("wrong timeout authority: %+v", w.Timeout)
		}
		if !strings.Contains(got.Detail, gitErr.Error()) || !strings.Contains(got.RecoveryAction, DisambiguationTimeoutEnv) || !strings.Contains(wrapped.Error(), DisambiguationTimeoutCode) {
			t.Fatalf("missing recovery or original error: %+v; wrapped=%v", got, wrapped)
		}
	})

	t.Run("invalid override retains analyzer absent verdict", func(t *testing.T) {
		t.Setenv(DisambiguationTimeoutEnv, "invalid")
		probes := 0
		hasDisambiguationContract = func(ctx context.Context, _, _ string) (bool, error) {
			probes++
			deadline, ok := ctx.Deadline()
			if remaining := time.Until(deadline); !ok || remaining <= time.Minute || remaining > defaultDisambiguationTimeout {
				t.Fatalf("fallback deadline remaining=%s present=%t", remaining, ok)
			}
			return false, nil
		}
		got := validatePreparedLandDisambiguation("/root", "/worker", receipt, git)
		if !got.OK || probes != 3 || receipt.Disambiguation != binding {
			t.Fatalf("analyzer-absent receipt changed: result=%+v probes=%d", got, probes)
		}
	})

	t.Run("ordinary probe error stays untyped", func(t *testing.T) {
		t.Setenv(DisambiguationTimeoutEnv, "300000")
		gitErr := errors.New("git ls-tree: object missing")
		hasDisambiguationContract = func(context.Context, string, string) (bool, error) { return false, gitErr }
		if _, err := preparedDisambiguationApplicable("/root", binding); err != gitErr {
			t.Fatalf("ordinary probe error changed: %v", err)
		}
		got := validatePreparedLandDisambiguation("/root", "/worker", receipt, git)
		if got.OK || got.Code != LandResultPreparedReprepare || !got.Preserved || got.Disambiguation != nil || got.RecoveryAction != "" || got.Detail != gitErr.Error() {
			t.Fatalf("ordinary failure was relabeled: %+v", got)
		}
	})
}
