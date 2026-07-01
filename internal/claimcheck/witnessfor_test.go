package claimcheck

import (
	"strings"
	"testing"
)

// TestWitnessForClasses pins the closed vocabulary and the fixed precedence
// (shipped > visual > perf > logic-default), and that every plan is runnable: a
// non-empty command + reference, and a skeleton for the classes whose proof is a new
// test. Determinism: the same claim maps to the same plan.
func TestWitnessForClasses(t *testing.T) {
	cases := []struct {
		name         string
		claim        string
		wantClass    string
		wantCommand  string // substring
		wantSkeleton bool
	}{
		{
			name:        "shipped claim maps to the commit witness",
			claim:       "shipped the fix and pushed to main",
			wantClass:   WitnessShipped,
			wantCommand: "dos verify",
		},
		{
			name:         "visual claim maps to a captured-render skeleton",
			claim:        "the TUI pane shows ANSI garbage after redraw",
			wantClass:    WitnessVisual,
			wantCommand:  "go test",
			wantSkeleton: true,
		},
		{
			name:        "perf claim maps to the net-true grader",
			claim:       "throughput is 40% higher at lower p99 latency",
			wantClass:   WitnessPerf,
			wantCommand: "fak claim-check --statement",
		},
		{
			name:         "behavior claim defaults to the fail-before/pass-after repro",
			claim:        "the resolver picks the stale entry when two match",
			wantClass:    WitnessLogic,
			wantCommand:  "go test",
			wantSkeleton: true,
		},
		{
			name:        "shipped outranks perf on a mixed claim",
			claim:       "landed the 3x faster scheduler",
			wantClass:   WitnessShipped,
			wantCommand: "dos verify",
		},
		{
			name:         "visual outranks perf on a mixed claim",
			claim:        "the render is 2x faster and no longer flickers",
			wantClass:    WitnessVisual,
			wantSkeleton: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			plan := WitnessFor(c.claim)
			if plan.Class != c.wantClass {
				t.Fatalf("class = %q, want %q (rationale=%q)", plan.Class, c.wantClass, plan.Rationale)
			}
			if plan.Command == "" || plan.Reference == "" || plan.Rationale == "" {
				t.Fatalf("plan not runnable: %+v", plan)
			}
			if c.wantCommand != "" && !strings.Contains(plan.Command, c.wantCommand) {
				t.Fatalf("command %q missing %q", plan.Command, c.wantCommand)
			}
			if c.wantSkeleton && plan.Skeleton == "" {
				t.Fatalf("class %s plan carries no test skeleton", plan.Class)
			}
			again := WitnessFor(c.claim)
			if again != plan {
				t.Fatalf("WitnessFor is not deterministic:\nfirst  %+v\nsecond %+v", plan, again)
			}
		})
	}
}

// TestWitnessForPerfEmbedsClaim pins that the perf plan's command carries the claim
// verbatim, so the emitted line is runnable without editing the statement back in.
func TestWitnessForPerfEmbedsClaim(t *testing.T) {
	claim := "warm-path tok/s is 3.1x the tuned baseline"
	plan := WitnessFor(claim)
	if plan.Class != WitnessPerf {
		t.Fatalf("class = %q, want perf", plan.Class)
	}
	if !strings.Contains(plan.Command, claim) {
		t.Fatalf("perf command does not embed the claim: %q", plan.Command)
	}
}

// TestRunWitnessFixture is the #2153 witness: one sample claim per class grades to its
// labeled class with a runnable plan — the corpus is the scaffold's own re-derivable
// evidence, the same shape as the grader's RunFixture.
func TestRunWitnessFixture(t *testing.T) {
	cases, passed := RunWitnessFixture()
	if len(cases) == 0 {
		t.Fatal("empty witness fixture corpus")
	}
	if passed != len(cases) {
		for _, c := range cases {
			if !c.OK {
				t.Errorf("fixture %s: claim %q graded %q, want %q", c.Name, c.Claim, c.Got, c.Expect)
			}
		}
		t.Fatalf("witness fixture: %d/%d passed", passed, len(cases))
	}
}
