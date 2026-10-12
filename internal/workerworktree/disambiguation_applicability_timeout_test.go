package workerworktree

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms
func TestDisambiguationApplicabilityHonorsDeadlineRecovery(t *testing.T) {
	t.Setenv(DisambiguationTimeoutEnv, "300000")
	oldHas, oldRead := hasDisambiguationContract, readDisambiguation
	t.Cleanup(func() {
		hasDisambiguationContract, readDisambiguation = oldHas, oldRead
	})
	readDisambiguation = func(context.Context, string, string, func(string)) DisambiguationWitness {
		t.Error("failed contract probe must not start an oracle witness")
		return DisambiguationWitness{}
	}
	hasDisambiguationContract = func(ctx context.Context, repo, tree string) (bool, error) {
		deadline, ok := ctx.Deadline()
		if remaining := time.Until(deadline); !ok || remaining <= 4*time.Minute || remaining > 5*time.Minute {
			t.Errorf("probe deadline remaining=%s, present=%t; want the explicit five-minute budget", remaining, ok)
		}
		return false, fmt.Errorf("git ls-tree: %w", context.DeadlineExceeded)
	}

	got, ok := verifyApplicableDisambiguation("/root", "/worker", "post-tree")
	if ok || got == nil {
		t.Fatalf("deadline result = (%+v, %t), want a preserved refusal", got, ok)
	}
	if got.Timeout.EffectiveTimeoutMS != 300000 || got.Timeout.RecoveryMode != disambiguationRecoveryExplicit {
		t.Fatalf("probe timeout receipt = %+v, want the explicit bounded override", got.Timeout)
	}
	d := got.Before.Diagnostic
	if d == nil || d.Code != DisambiguationTimeoutCode || !d.Recoverable || d.Witness != "applicability" || d.Subphase != "contract-probe" || d.TimeoutMS != 300000 {
		t.Fatalf("probe diagnostic = %+v, want recoverable typed contract-probe timeout", d)
	}
	if !strings.Contains(d.Recovery, DisambiguationTimeoutEnv) || !strings.Contains(got.Before.Detail, "git ls-tree") || !got.deadlineExceeded() {
		t.Fatalf("probe lost recovery or command evidence: %+v", got.Before)
	}
}
