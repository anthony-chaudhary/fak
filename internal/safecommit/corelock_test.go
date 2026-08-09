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

// TestCoreLockMaintenanceRecordsWitnessCorrelation is the end-to-end half of the
// observation rung: a CONFIRMED witness that names a file this commit never touches
// still clears the lock (enforcement is deliberately not shipped here), but the
// mismatch now lands on the append-only maintenance decision instead of being
// invisible. This is the shape of the real record for 525a596cd2, whose witness
// named internal/adjudicator/decide.go while the commit changed three other files.
func TestCoreLockMaintenanceRecordsWitnessCorrelation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		claim   string
		want    string
		wantSub string
	}{
		{
			name:    "witness names an untouched file",
			claim:   "committed:internal/adjudicator/decide.go",
			want:    "uncorrelated",
			wantSub: "internal/adjudicator/decide.go",
		},
		{
			name:    "maintainer names their own changed file",
			claim:   "committed:internal/corelocks/corelocks.go",
			want:    "correlated",
			wantSub: "part of the change",
		},
		{
			name:    "a history-shaped claim cannot be judged either way",
			claim:   "ancestor:reviewed-maintenance-sha",
			want:    "indeterminate",
			wantSub: "before that change is a commit",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := &fakeGit{reply: onTrunkBase()}
			g.reply["status"] = reply{out: " M internal/corelocks/corelocks.go\n", code: 0}
			g.reply["diff-tree"] = reply{out: "internal/corelocks/corelocks.go\n", code: 0}

			opts := hardSelfOpts()
			opts.CoreLockMaintenanceWitness = tc.claim
			opts.CoreLockWitnessResolver = fixedCoreLockWitness{outcome: abi.WitnessConfirmed}
			rec, captured := decisionRecorder(t)
			opts.Recorder = rec

			res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
			if err != nil {
				t.Fatalf("unexpected infra error: %v", err)
			}
			// Observation only: the verdict is unchanged in every case.
			if !res.Verified || res.Reason != "" {
				t.Fatalf("a CONFIRMED witness must still clear the lock, got %+v", res)
			}
			if !strings.HasPrefix(res.CoreLockWitnessCorrelation, tc.want+": ") {
				t.Fatalf("result correlation = %q, want %s", res.CoreLockWitnessCorrelation, tc.want)
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
			if !strings.HasPrefix(maintenance.WitnessCorrelation, tc.want+": ") {
				t.Fatalf("recorded correlation = %q, want %s", maintenance.WitnessCorrelation, tc.want)
			}
			if !strings.Contains(maintenance.WitnessCorrelation, tc.wantSub) {
				t.Fatalf("recorded correlation missing %q:\n%s", tc.wantSub, maintenance.WitnessCorrelation)
			}
		})
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
