package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

type dummyLoopPlanner struct{}

func (d dummyLoopPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	return &Completion{Message: Message{Content: "done"}}, nil
}

func (d dummyLoopPlanner) Model() string { return "dummy-planner" }

func TestRunArmRetainsCallerPolicySnapshot(t *testing.T) {
	t.Cleanup(func() {
		Configure()
	})

	customPred := adjudicator.ArgPredicate{
		Tool:   "Write",
		Arg:    "file_path",
		Kind:   adjudicator.ArgAllowGlob,
		Glob:   "safe/**",
		Reason: abi.ReasonPolicyBlock,
	}
	customPolicy := adjudicator.Policy{
		Posture:         adjudicator.PostureFailClosed,
		Allow:           map[string]bool{"Write": true, "Bash": true},
		ArgPredicates:   []adjudicator.ArgPredicate{customPred},
		SelfModifyGlobs: []string{"custom/guarded/"},
	}

	ctx := context.Background()
	planner := dummyLoopPlanner{}

	// Run arm with WithPolicySnapshot
	_, err := RunArm(ctx, planner, "dummy task", true, 1, nil, WithPolicySnapshot(customPolicy))
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}

	snap := adjudicator.Default.PolicySnapshot()
	if len(snap.ArgPredicates) != 1 {
		t.Fatalf("expected 1 ArgPredicate, got %d", len(snap.ArgPredicates))
	}
	gotPred := snap.ArgPredicates[0]
	if gotPred.Tool != "Write" || gotPred.Glob != "safe/**" || gotPred.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("unexpected ArgPredicate: %+v", gotPred)
	}
	if len(snap.SelfModifyGlobs) != 1 || snap.SelfModifyGlobs[0] != "custom/guarded/" {
		t.Fatalf("unexpected SelfModifyGlobs: %v", snap.SelfModifyGlobs)
	}

	// Verify predicate enforcement through adjudicator.Default
	safeCall := &abi.ToolCall{
		Tool: "Write",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"safe/report.txt"}`)},
	}
	if v := adjudicator.Default.Adjudicate(ctx, safeCall); v.Kind != abi.VerdictAllow {
		t.Fatalf("safe Write call was not allowed: got %+v", v)
	}

	outOfScopeCall := &abi.ToolCall{
		Tool: "Write",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"unsafe/evil.txt"}`)},
	}
	if v := adjudicator.Default.Adjudicate(ctx, outOfScopeCall); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("out-of-scope Write call got %+v, want VerdictDeny/ReasonPolicyBlock", v)
	}

	// Verify that run without WithPolicySnapshot applies standard Configure() defaults
	_, err = RunArm(ctx, planner, "dummy task", true, 1, nil)
	if err != nil {
		t.Fatalf("RunArm without snapshot failed: %v", err)
	}

	snapDefault := adjudicator.Default.PolicySnapshot()
	if len(snapDefault.ArgPredicates) != 0 {
		t.Fatalf("expected 0 ArgPredicates on default Configure, got %d", len(snapDefault.ArgPredicates))
	}
	if len(snapDefault.SelfModifyGlobs) <= 1 {
		t.Fatalf("expected standard SelfModifyGlobs on default Configure, got %v", snapDefault.SelfModifyGlobs)
	}
}
