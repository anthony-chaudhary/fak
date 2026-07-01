package safecommit

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

type fixedCoreLockWitness struct{ outcome abi.WitnessOutcome }

func (f fixedCoreLockWitness) Resolve(context.Context, *abi.ToolCall, string) abi.WitnessOutcome {
	return f.outcome
}

func hardSelfOpts() Options {
	opts := baseOpts()
	opts.Paths = []string{"internal/corelocks/corelocks.go"}
	opts.Message = "feat(corelocks): tighten hard-self enforcement (#1683) (fak corelocks)"
	return opts
}

func TestCoreLockHardSelfRefusesBeforeStagingWithoutWitness(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["status"] = reply{out: " M internal/corelocks/corelocks.go\n", code: 0}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), hardSelfOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonCoreSelfModify {
		t.Fatalf("reason = %q detail=%q, want %q", res.Reason, res.Detail, ReasonCoreSelfModify)
	}
	for _, want := range []string{"hard-self", "internal/corelocks/corelocks.go", "--core-lock-maintenance-witness"} {
		if !strings.Contains(res.Detail, want) {
			t.Fatalf("detail missing %q:\n%s", want, res.Detail)
		}
	}
	if g.sawSubcommand("add") || g.sawSubcommand("commit") {
		t.Fatalf("hard-self refusal must happen before staging/commit; calls=%v", g.calls)
	}
}

func TestCoreLockOrdinaryLeafIsNotRefused(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonCoreSelfModify {
		t.Fatalf("ordinary leaf must not trip hard-self enforcement: %+v", res)
	}
	if !res.Verified {
		t.Fatalf("ordinary leaf should still commit and verify, got %+v", res)
	}
}

func TestCoreLockMaintenanceWitnessRecordsReadBack(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["status"] = reply{out: " M internal/corelocks/corelocks.go\n", code: 0}
	g.reply["diff-tree"] = reply{out: "internal/corelocks/corelocks.go\n", code: 0}

	opts := hardSelfOpts()
	opts.CoreLockMaintenanceWitness = "ancestor:reviewed-maintenance-sha"
	opts.CoreLockWitnessResolver = fixedCoreLockWitness{outcome: abi.WitnessConfirmed}
	rec, captured := decisionRecorder(t)
	opts.Recorder = rec

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !res.Verified || res.Reason != "" {
		t.Fatalf("maintenance witness should allow a verified commit, got %+v", res)
	}
	if got := res.CoreLockWitness; got != opts.CoreLockMaintenanceWitness {
		t.Fatalf("result witness = %q, want %q", got, opts.CoreLockMaintenanceWitness)
	}
	var maintenance *witness.Decision
	for i := range *captured {
		if (*captured)[i].Op == "corelock-maintenance" {
			maintenance = &(*captured)[i]
			break
		}
	}
	if maintenance == nil {
		t.Fatalf("missing corelock-maintenance decision; captured=%+v", *captured)
	}
	if maintenance.Verdict != witness.VerdictAllow || maintenance.ReasonClass != ReasonCoreSelfModify {
		t.Fatalf("maintenance decision = %+v", *maintenance)
	}
	if maintenance.Witness != opts.CoreLockMaintenanceWitness {
		t.Fatalf("maintenance witness = %q, want %q", maintenance.Witness, opts.CoreLockMaintenanceWitness)
	}
	if len(maintenance.Tree) != 1 || maintenance.Tree[0] != "internal/corelocks/corelocks.go" {
		t.Fatalf("maintenance tree = %+v", maintenance.Tree)
	}
}

func TestCoreLockRefutedMaintenanceWitnessStillRefuses(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["status"] = reply{out: " M internal/corelocks/corelocks.go\n", code: 0}
	opts := hardSelfOpts()
	opts.CoreLockMaintenanceWitness = "ancestor:not-confirmed"
	opts.CoreLockWitnessResolver = fixedCoreLockWitness{outcome: abi.WitnessRefuted}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonCoreSelfModify {
		t.Fatalf("refuted witness should keep hard-self gate closed, got %+v", res)
	}
	if !strings.Contains(res.Detail, "refuted") {
		t.Fatalf("detail should report refuted witness, got %q", res.Detail)
	}
}
