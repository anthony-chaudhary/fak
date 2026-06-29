package modelroute

import (
	"context"
	"strings"
	"testing"
)

func TestReviewDiffWithScoutRefutes(t *testing.T) {
	var saw Subject
	scout := ClassifierFunc(func(_ context.Context, s Subject) (ScoutLabel, error) {
		saw = s
		return ScoutLabel{Labels: map[string]string{
			"verdict": "refute",
			"reason":  "missing test for the changed behavior",
		}}, nil
	})

	req := ReviewRequest{Model: "cheap-scout", Objective: "ship the loop review rung", Diff: "diff --git a/x b/x\n+broken\n"}
	res, err := ReviewDiffWithScout(context.Background(), scout, req)
	if err != nil {
		t.Fatalf("ReviewDiffWithScout: %v", err)
	}
	if res.Verdict != ReviewRefute {
		t.Fatalf("verdict = %q, want refute", res.Verdict)
	}
	if !strings.Contains(res.Reason, "missing test") {
		t.Fatalf("reason = %q", res.Reason)
	}
	if res.ScoutCalls != 1 {
		t.Fatalf("ScoutCalls = %d, want 1", res.ScoutCalls)
	}
	if saw.Aspect != AspectScout || saw.Tool != ReviewTool {
		t.Fatalf("subject = %+v, want scout/%s", saw, ReviewTool)
	}
	if saw.Labels["objective"] != req.Objective || saw.Labels["diff"] != req.Diff {
		t.Fatalf("subject labels lost objective/diff: %+v", saw.Labels)
	}
	if saw.Labels["diff_sha256"] == "" || res.DiffSHA256 != saw.Labels["diff_sha256"] {
		t.Fatalf("diff digest not carried through: subject=%q result=%q", saw.Labels["diff_sha256"], res.DiffSHA256)
	}
}

func TestReviewDiffWithScoutPassesAliases(t *testing.T) {
	scout := ClassifierFunc(func(context.Context, Subject) (ScoutLabel, error) {
		return ScoutLabel{Labels: map[string]string{"verdict": "approved", "summary": "looks consistent"}}, nil
	})
	res, err := ReviewDiffWithScout(context.Background(), scout, ReviewRequest{Diff: "diff"})
	if err != nil {
		t.Fatalf("ReviewDiffWithScout: %v", err)
	}
	if res.Verdict != ReviewPass {
		t.Fatalf("verdict = %q, want pass", res.Verdict)
	}
}

func TestReviewDiffWithScoutRejectsInvalidVerdict(t *testing.T) {
	scout := ClassifierFunc(func(context.Context, Subject) (ScoutLabel, error) {
		return ScoutLabel{Labels: map[string]string{"verdict": "maybe"}}, nil
	})
	if _, err := ReviewDiffWithScout(context.Background(), scout, ReviewRequest{Diff: "diff"}); err == nil {
		t.Fatal("invalid review verdict should fail loud")
	}
}
