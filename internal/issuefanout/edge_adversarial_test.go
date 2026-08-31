package issuefanout

// edge_adversarial_test.go is the #2511 qa-edge-sweep: table-driven edge and
// adversarial coverage over every documented input class of the fan-out
// planner and every error path Build can return. The adversarial table is the
// map of what is proven: for ANY input, Build either refuses with a contract
// Refusal or emits candidates that pass the full issuecontract review — the
// planner never hands back a plan that fails its own dispatchability promise.

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// TestBuildEdgeRefusalTable drives every refusal path Build declares, and pins
// the contract on each: the error is a deliberate Refusal (not a bare error),
// the message names the recovery, and no partial plan escapes alongside it.
func TestBuildEdgeRefusalTable(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Input)
		wantMsg string
	}{
		{"zero input", func(in *Input) { *in = Input{} }, "title and leaf are required"},
		{"whitespace title", func(in *Input) { in.Title = " \t " }, "title and leaf are required"},
		{"whitespace leaf", func(in *Input) { in.Leaf = "\n" }, "title and leaf are required"},
		{"whitespace spine ref", func(in *Input) { in.SpineRef = "  " }, "working spine first"},
		{"negative max", func(in *Input) { in.Max = -1 }, "below the fan-out floor"},
		{"max one below floor", func(in *Input) { in.Max = MinFanout - 1 }, "below the fan-out floor"},
		{"unknown area", func(in *Input) { in.Areas = []string{"qa", "bogus"} }, "known:"},
		{"area filter below floor", func(in *Input) { in.Areas = []string{"release"} }, "below the fan-out floor"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := spineInput()
			tc.mutate(&in)
			plan, err := Build(in)
			if err == nil {
				t.Fatalf("Build accepted the input, want a refusal containing %q", tc.wantMsg)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("refusal does not name the recovery: got %q, want substring %q", err, tc.wantMsg)
			}
			if got := ClassifyOutcome(err); got != OutcomeRefused {
				t.Fatalf("refusal classified as %q, want %q (a contract refusal must be a *Refusal)", got, OutcomeRefused)
			}
			if !reflect.DeepEqual(plan, Plan{}) {
				t.Fatalf("a refused Build leaked a partial plan: %+v", plan)
			}
		})
	}
}

// TestBuildEdgeOversizedAndFilterNormalization pins the benign edges: an
// oversized cap is a no-op (never a panic or truncation), area filters
// normalize case/whitespace/duplicates, and a filter of only blank entries
// means "no filter" rather than "no candidates".
func TestBuildEdgeOversizedAndFilterNormalization(t *testing.T) {
	full := mustBuild(t, spineInput())

	oversized := spineInput()
	oversized.Max = math.MaxInt
	if got := mustBuild(t, oversized); !reflect.DeepEqual(got.Candidates, full.Candidates) {
		t.Fatalf("an oversized cap changed the plan: got %d candidates, want %d", len(got.Candidates), len(full.Candidates))
	}

	qa := spineInput()
	qa.Areas = []string{"qa"}
	messy := spineInput()
	messy.Areas = []string{" QA ", "", "qa", "\t"}
	if !reflect.DeepEqual(mustBuild(t, messy).Candidates, mustBuild(t, qa).Candidates) {
		t.Fatal("area filter did not normalize case/whitespace/duplicates")
	}

	blanks := spineInput()
	blanks.Areas = []string{" ", "", "\t"}
	if got := mustBuild(t, blanks); !reflect.DeepEqual(got.Candidates, full.Candidates) {
		t.Fatalf("blank-only area filter must mean no filter: got %d candidates, want %d", len(got.Candidates), len(full.Candidates))
	}
}

// TestBuildAdversarialContractHolds is the planner's core promise under
// hostile input: every plan Build returns is dispatchable the moment it is
// filed. An input that would break the marker-key alphabet, the default
// paths, or the private-boundary screen must be refused — not expanded into
// candidates the issue contract then rejects downstream, after filing.
func TestBuildAdversarialContractHolds(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Input)
		wantBuild bool // true: the input is legitimate and must build
	}{
		{"leaf with a space", func(in *Input) { in.Leaf = "issue fanout" }, false},
		{"leaf with shell metacharacters", func(in *Input) { in.Leaf = "leaf;x|y" }, false},
		{"leaf with template braces", func(in *Input) { in.Leaf = "{leaf}" }, false},
		{"leaf with non-ascii runes", func(in *Input) { in.Leaf = "lëaf" }, false},
		{"oversized leaf", func(in *Input) { in.Leaf = strings.Repeat("a", 200) }, false},
		{"title tripping the private boundary", func(in *Input) { in.Title = "rotate the api key nightly" }, false},
		{"spine ref tripping the private boundary", func(in *Input) { in.SpineRef = "deadbeef (secret key material)" }, false},
		{"nested leaf path stays valid", func(in *Input) { in.Leaf = "sub/pkg" }, true},
		{"placeholder injection stays inert", func(in *Input) { in.Title = "{paths} {spine} {leaf}" }, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := spineInput()
			tc.mutate(&in)
			plan, err := Build(in)
			if err != nil {
				if tc.wantBuild {
					t.Fatalf("legitimate input refused: %v", err)
				}
				if got := ClassifyOutcome(err); got != OutcomeRefused {
					t.Fatalf("hostile input rejected with %q, want %q (a deliberate contract refusal)", got, OutcomeRefused)
				}
				return
			}
			for _, c := range plan.Candidates {
				r := issuepolicy.ReviewCandidate(c, issuepolicy.Options{})
				if r.Dispatchability != issuepolicy.Dispatchable {
					t.Errorf("candidate %s not dispatchable (reasons %v, missing %v): the planner emitted a broken contract instead of refusing",
						c.Key, r.Reasons, r.MissingFields)
				}
			}
		})
	}
}

// TestBuildAdversarialPlaceholderSinglePass pins the replacer's single-pass
// substitution: placeholder tokens smuggled in via input fields stay literal
// in the output instead of being re-expanded into other fields.
func TestBuildAdversarialPlaceholderSinglePass(t *testing.T) {
	in := spineInput()
	in.Title = "{leaf} smuggler"
	plan := mustBuild(t, in)
	if got := plan.Candidates[0].Title; !strings.Contains(got, "{leaf} smuggler") {
		t.Fatalf("smuggled placeholder was re-expanded: %q", got)
	}
}

// TestClassifyOutcomeEdgeWrapped proves outcome classification survives error
// wrapping: a CLI or wave layer that wraps Build's refusal must still fold it
// as a refusal, and a wrapped plain error must still fold as an error.
func TestClassifyOutcomeEdgeWrapped(t *testing.T) {
	bad := spineInput()
	bad.SpineRef = " "
	_, refusal := Build(bad)
	if refusal == nil {
		t.Fatal("setup: empty spine_ref must refuse")
	}
	if got := ClassifyOutcome(fmt.Errorf("cli: %w", refusal)); got != OutcomeRefused {
		t.Fatalf("wrapped refusal classified as %q, want %q", got, OutcomeRefused)
	}
	if got := ClassifyOutcome(fmt.Errorf("outer: %w", fmt.Errorf("inner: %w", refusal))); got != OutcomeRefused {
		t.Fatalf("doubly-wrapped refusal classified as %q, want %q", got, OutcomeRefused)
	}
	if got := ClassifyOutcome(fmt.Errorf("cli: %w", errors.New("boom"))); got != OutcomeError {
		t.Fatalf("wrapped plain error classified as %q, want %q", got, OutcomeError)
	}
}

// TestAdoptionEdgeHostileMarkerKeys pins the credit/orphan/ignore boundary on
// malformed marker keys, so a corrupt gh row can shift a count but never
// crash or silently double-book the adoption meter.
func TestAdoptionEdgeHostileMarkerKeys(t *testing.T) {
	rep := Adoption([]string{"a"}, []string{
		"fanout-",        // bare prefix: matches no leaf -> orphan
		"fanout-a",       // missing the trailing dash -> orphan, not a credit
		"  fanout-a-x  ", // whitespace-wrapped -> trimmed and credited
		"fanout-a-",      // empty slug -> still credited (prefix holds)
		"FANOUT-a-y",     // wrong case -> not a fan-out key at all: ignored
	})
	if len(rep.Leaves) != 1 || rep.Leaves[0].FanoutFiled != 2 {
		t.Fatalf("credit boundary wrong: %+v (want leaf a credited exactly 2)", rep.Leaves)
	}
	if !reflect.DeepEqual(rep.OrphanMarkers, []string{"fanout-", "fanout-a"}) {
		t.Fatalf("orphans: got %v, want [fanout- fanout-a] (the wrong-case key is ignored, not orphaned)", rep.OrphanMarkers)
	}
}

// mustBuild fails the test on any refusal — for rows where Build must accept.
func mustBuild(t *testing.T, in Input) Plan {
	t.Helper()
	plan, err := Build(in)
	if err != nil {
		t.Fatalf("Build(%+v): %v", in, err)
	}
	return plan
}

// TestFileLiveEdgeRefusalTable maps every strict live refusal class through
// the real post-filing review seam. Each refusal must happen before the first
// external write and must not leak a partial result.
func TestFileLiveEdgeRefusalTable(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Plan, *LiveOptions)
		wantMsg string
	}{
		{
			name: "nil runner",
			mutate: func(_ *Plan, opt *LiveOptions) {
				opt.Runner = nil
			},
			wantMsg: "needs a gh Runner",
		},
		{
			name: "missing parent issue",
			mutate: func(plan *Plan, _ *LiveOptions) {
				plan.Input.ParentIssue = 0
			},
			wantMsg: "requires --parent-issue and --parent-baseline-points",
		},
		{
			name: "missing parent baseline",
			mutate: func(plan *Plan, _ *LiveOptions) {
				plan.Input.ParentBaseline = 0
			},
			wantMsg: "requires --parent-issue and --parent-baseline-points",
		},
		{
			name: "malformed final candidate",
			mutate: func(plan *Plan, _ *LiveOptions) {
				plan.Candidates[len(plan.Candidates)-1].RequiredModelTier = ""
			},
			wantMsg: "fails the strict post-filing issue contract",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan := liveTestPlan(t)
			calls := 0
			opt := LiveOptions{Runner: func([]string) (string, string, bool) {
				calls++
				return "https://github.com/o/r/issues/1", "", true
			}}
			tc.mutate(&plan, &opt)
			got, err := FileLive(plan, nil, opt)
			if err == nil {
				t.Fatalf("FileLive accepted input, want refusal containing %q", tc.wantMsg)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("refusal = %q, want substring %q", err, tc.wantMsg)
			}
			if gotOutcome := ClassifyOutcome(err); gotOutcome != OutcomeRefused {
				t.Fatalf("outcome = %q, want %q", gotOutcome, OutcomeRefused)
			}
			if calls != 0 {
				t.Fatalf("runner calls = %d, want zero before whole-batch validation", calls)
			}
			if !reflect.DeepEqual(got, LiveResult{}) {
				t.Fatalf("refusal leaked a partial live result: %+v", got)
			}
		})
	}
}

// TestFileLiveAdversarialRunnerOutput pins both non-refusal live error paths:
// hostile stderr and malformed success stdout become failed rows, while the
// remaining candidates are still attempted and no fabricated issue is filed.
func TestFileLiveAdversarialRunnerOutput(t *testing.T) {
	tests := []struct {
		name       string
		stdout     string
		stderr     string
		ok         bool
		wantReason string
	}{
		{"hostile stderr", "", "  gh: denied\nwith control-like text --force  ", false, "gh: denied\nwith control-like text --force"},
		{"malformed success output", "created issue without a URL", "", true, "gh issue create exited 0 but printed no issue URL"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan := liveTestPlan(t)
			calls := 0
			res, err := FileLive(plan, nil, LiveOptions{Runner: func([]string) (string, string, bool) {
				calls++
				return tc.stdout, tc.stderr, tc.ok
			}})
			if err != nil {
				t.Fatalf("FileLive returned a contract refusal for a runner result: %v", err)
			}
			if calls != len(plan.Candidates) {
				t.Fatalf("runner calls = %d, want %d", calls, len(plan.Candidates))
			}
			if res.Failed != len(plan.Candidates) || res.Filed != 0 || res.Skipped != 0 {
				t.Fatalf("result = %+v, want every candidate failed and none filed/skipped", res)
			}
			for _, row := range res.Rows {
				if row.Action != "failed" || row.Number != nil || row.URL != "" || row.Reason != tc.wantReason {
					t.Fatalf("row = %+v, want failed row with reason %q and no fabricated issue", row, tc.wantReason)
				}
			}
		})
	}
}
