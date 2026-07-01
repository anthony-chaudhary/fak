package releasestatus

import (
	"strings"
	"testing"
)

// TestFoldSchemaAndShape proves the fold emits the python SCHEMA id and the full
// record shape (every sub-record the python emits has a value), so a consumer of
// the native fold sees the same envelope.
func TestFoldSchemaAndShape(t *testing.T) {
	s := Fold(Facts{Root: "/r", HeadSHA: "deadbeef", Branch: "main"})
	if s.Schema != "fleet-release-status/1" {
		t.Fatalf("schema = %q, want fleet-release-status/1", s.Schema)
	}
	if s.Root != "/r" || s.HeadSHA != "deadbeef" || s.Branch != "main" {
		t.Fatalf("head fields not threaded: %+v", s)
	}
	// dirty/cadence/stable are always present (never nil maps); next_action + loop
	// fields always populated.
	if s.NextAction.Kind == "" {
		t.Fatal("next_action.kind empty")
	}
	if s.Stable.Evidence.Rows == nil || s.Stable.Evidence.Failures == nil {
		t.Fatal("stable evidence slices must be non-nil for stable JSON parity")
	}
}

// TestDirtyRelevanceClassification ports release_status.py's
// is_release_relevant_dirty_path split: a VERSION edit is release-relevant, a random
// source file is not, and the counts/lists partition correctly.
func TestDirtyRelevanceClassification(t *testing.T) {
	d := foldDirty([]DirtyEntry{
		{Path: "VERSION", Untracked: false},
		{Path: ".github/workflows/ci.yml", Untracked: false},
		{Path: "internal/foo/bar.go", Untracked: false},
		{Path: "newfile.txt", Untracked: true},
		{Path: "tools/release_cut.py", Untracked: true},
	})
	if d.Clean {
		t.Fatal("not clean with 5 entries")
	}
	if d.ModifiedCount != 3 || d.UntrackedCount != 2 {
		t.Fatalf("modified=%d untracked=%d, want 3/2", d.ModifiedCount, d.UntrackedCount)
	}
	// VERSION, the workflow, and the tools/release_ path are release-relevant.
	if d.ReleaseRelevantCount != 3 {
		t.Fatalf("release_relevant_count=%d, want 3 (%v)", d.ReleaseRelevantCount, d.ReleaseRelevant)
	}
	if d.UnrelatedCount != 2 {
		t.Fatalf("unrelated_count=%d, want 2 (%v)", d.UnrelatedCount, d.Unrelated)
	}
}

// TestNextActionReleaseHotTree ports the decision==release + dirty-tree branch
// (kind cut_release_hot_tree), and the clean-tree branch (cut_release).
func TestNextActionReleaseBranches(t *testing.T) {
	clean := foldNextAction(
		Decision{Decision: "release", NextVersion: "v1.2.4"},
		StableSummary{},
		DirtySummary{Clean: true},
		CIDiagnosis{},
	)
	if clean.Kind != "cut_release" {
		t.Fatalf("clean tree release kind=%q, want cut_release", clean.Kind)
	}
	if !strings.Contains(clean.Detail, "v1.2.4") {
		t.Fatalf("cut_release detail missing version: %q", clean.Detail)
	}
	hot := foldNextAction(
		Decision{Decision: "release", NextVersion: "v1.2.4"},
		StableSummary{},
		DirtySummary{Clean: false, ModifiedCount: 2, UntrackedCount: 1},
		CIDiagnosis{},
	)
	if hot.Kind != "cut_release_hot_tree" {
		t.Fatalf("dirty tree release kind=%q, want cut_release_hot_tree", hot.Kind)
	}
	if !strings.Contains(hot.Detail, "fak release ship --execute") {
		t.Fatalf("hot tree detail missing ship verb: %q", hot.Detail)
	}
}

// TestNextActionBlockerPriority ports the blocker → kind routing, including the
// dirty-release-relevant gate that precedes CI blockers, and the CI billing branch.
func TestNextActionBlockerPriority(t *testing.T) {
	cases := []struct {
		name     string
		decision Decision
		dirty    DirtySummary
		ci       CIDiagnosis
		want     string
	}{
		{
			name:     "release-relevant dirty gates before CI",
			decision: Decision{Decision: "hold", Blockers: []string{"CI_BASE_RED"}},
			dirty:    DirtySummary{Clean: false, ReleaseRelevantCount: 1},
			want:     "clean_worktree",
		},
		{
			name:     "ci billing classification",
			decision: Decision{Decision: "hold", Blockers: []string{"CI_BASE_RED"}},
			dirty:    DirtySummary{Clean: true},
			ci:       CIDiagnosis{Action: "fix_ci_billing", Detail: "billing"},
			want:     "fix_ci_billing",
		},
		{
			name:     "ci red without billing",
			decision: Decision{Decision: "hold", Blockers: []string{"CI_BASE_RED"}},
			dirty:    DirtySummary{Clean: true},
			want:     "fix_ci",
		},
		{
			name:     "version drift",
			decision: Decision{Decision: "hold", Blockers: []string{"VERSION_DRIFT"}},
			dirty:    DirtySummary{Clean: true},
			want:     "fix_version_topology",
		},
		{
			name:     "unparseable workflow",
			decision: Decision{Decision: "hold", Blockers: []string{"WORKFLOW_UNPARSEABLE"}},
			dirty:    DirtySummary{Clean: true},
			want:     "fix_workflow",
		},
		{
			name:     "generic blocker holds",
			decision: Decision{Decision: "hold", Reason: "soak", Blockers: []string{"SOMETHING_ELSE"}},
			dirty:    DirtySummary{Clean: true},
			want:     "hold",
		},
		{
			name:     "nothing pending waits",
			decision: Decision{Decision: "hold", Reason: "nothing"},
			dirty:    DirtySummary{Clean: true},
			want:     "wait",
		},
	}
	for _, tc := range cases {
		got := foldNextAction(tc.decision, StableSummary{Evidence: StableEvidence{OK: true}}, tc.dirty, tc.ci)
		if got.Kind != tc.want {
			t.Errorf("%s: kind=%q, want %q", tc.name, got.Kind, tc.want)
		}
	}
}

// TestLoopStatusGating proves an ACTION kind reds the loop bit and a non-action kind
// (wait) reads OK — the ACTION_NEXT_ACTIONS membership rule.
func TestLoopStatusGating(t *testing.T) {
	ok, verdict, _ := loopStatus(NextAction{Kind: "wait", Detail: "nothing"})
	if !ok || verdict != "OK" {
		t.Fatalf("wait should be OK, got ok=%v verdict=%q", ok, verdict)
	}
	ok, verdict, _ = loopStatus(NextAction{Kind: "cut_release", Detail: "cut"})
	if ok || verdict != "ACTION" {
		t.Fatalf("cut_release should be ACTION, got ok=%v verdict=%q", ok, verdict)
	}
}

// TestStableEvidenceFold ports stable_evidence_status(): a missing evidence file, a
// candidate_sha mismatch, and a clean row.
func TestStableEvidenceFold(t *testing.T) {
	stable := []StableTag{
		{Tag: "stable/2026-06-stable", SHA: "aaaa1111bbbb", EvidenceFound: false},
		{
			Tag:           "stable/2026-05-stable",
			SHA:           "cccc2222dddd",
			EvidenceFound: true,
			Evidence: map[string]string{
				"candidate_sha":      "9999wrong",
				"underlying_version": "v1.0.0",
				"codename":           "2026-05-stable",
			},
			UnderlyingExists: true,
			UnderlyingTagSHA: "cccc2222dddd",
		},
		{
			Tag:           "stable/2026-04-stable",
			SHA:           "eeee3333ffff",
			EvidenceFound: true,
			Evidence: map[string]string{
				"candidate_sha":      "eeee3333ffff",
				"underlying_version": "v0.9.0",
				"codename":           "2026-04-stable",
			},
			UnderlyingExists: true,
			UnderlyingTagSHA: "eeee3333ffff",
		},
	}
	ev := foldStableEvidence(stable)
	if ev.Checked != 3 {
		t.Fatalf("checked=%d, want 3", ev.Checked)
	}
	if ev.OK {
		t.Fatal("evidence should not be OK: a missing file + a sha mismatch")
	}
	// Row 0: missing file. Row 1: candidate_sha mismatch. Row 2: clean.
	if ev.Rows[0].OK || ev.Rows[1].OK || !ev.Rows[2].OK {
		t.Fatalf("row OK flags = %v/%v/%v, want false/false/true", ev.Rows[0].OK, ev.Rows[1].OK, ev.Rows[2].OK)
	}
	foundMismatch := false
	for _, f := range ev.Failures {
		if f.Tag == "stable/2026-05-stable" && strings.Contains(f.Detail, "does not match") {
			foundMismatch = true
		}
	}
	if !foundMismatch {
		t.Fatalf("expected a candidate_sha mismatch failure; failures=%+v", ev.Failures)
	}
}

// TestStableCandidateStates ports stable_candidate_status(): no candidate, soaking,
// ready, and already-promoted.
func TestStableCandidateStates(t *testing.T) {
	// no rolling tag -> no_candidate
	c := foldStableCandidate(Facts{StableWindowDays: 3})
	if c.State != "no_candidate" {
		t.Fatalf("empty rolling state=%q, want no_candidate", c.State)
	}

	base := Facts{
		RollingTags:       []string{"v1.0.0", "v1.1.0"},
		CandidateSHA:      "abcd1234",
		StableWindowDays:  3,
		CandidateAgeDays:  1.5,
		CandidateAgeKnown: true,
		SuggestedCodename: "2026-06-stable",
	}
	c = foldStableCandidate(base)
	if c.State != "soaking" || c.Ready {
		t.Fatalf("young candidate state=%q ready=%v, want soaking/false", c.State, c.Ready)
	}
	if c.RemainingDays != 1.5 {
		t.Fatalf("remaining_days=%v, want 1.5", c.RemainingDays)
	}

	old := base
	old.CandidateAgeDays = 5
	c = foldStableCandidate(old)
	if c.State != "ready" || !c.Ready {
		t.Fatalf("soaked candidate state=%q ready=%v, want ready/true", c.State, c.Ready)
	}

	promoted := base
	promoted.Stable = []StableTag{{Tag: "stable/2026-06-stable", SHA: "abcd1234"}}
	c = foldStableCandidate(promoted)
	if c.State != "already_promoted" {
		t.Fatalf("promoted candidate state=%q, want already_promoted", c.State)
	}
}

// TestCadenceFold ports cadence_status() over injected yaml text, including the
// absent-file case and the posture booleans.
func TestCadenceFold(t *testing.T) {
	absent := foldCadence(CadenceText{Present: false})
	if absent.Present || absent.Path != ".github/workflows/release-cadence.yml" {
		t.Fatalf("absent cadence = %+v", absent)
	}
	yaml := `on:
  schedule:
    - cron: "0 */2 * * *"
  workflow_dispatch:
    inputs:
      dry_run:
        default: "true"
concurrency:
  group: release-cadence
  cancel-in-progress: false
jobs:
  release:
    if: ${{ github.event.inputs.dry_run == false }}
    steps:
      - run: python tools/release_tag.py --require-ci --wait-ci
      - run: python tools/release_publish.py --execute
`
	c := foldCadence(CadenceText{Present: true, Text: yaml})
	if !c.Schedule || !c.ManualDispatch || !c.DryRunFirst || !c.SingleWriter || !c.TagAfterGreen || !c.CheckedGitHubRelease {
		t.Fatalf("cadence posture not all true: %+v", c)
	}
}

// TestStableLagAndRecommendation ports stable_summary()'s lag count over rolling
// tags newer than the latest stable's VERSION.
func TestStableLagAndRecommendation(t *testing.T) {
	f := Facts{
		RollingTags:      []string{"v1.0.0", "v1.1.0", "v1.2.0"},
		StableWindowDays: 3,
		Stable: []StableTag{
			{Tag: "stable/2026-06-stable", SHA: "abc", Version: "v1.0.0"},
		},
		CandidateSHA:      "zzz",
		CandidateAgeKnown: true,
		CandidateAgeDays:  10,
		SuggestedCodename: "2026-06-stable-2",
	}
	s := foldStable(f)
	if s.StableLag == nil || *s.StableLag != 2 {
		t.Fatalf("stable_lag = %v, want 2 (v1.1.0, v1.2.0 newer than v1.0.0)", s.StableLag)
	}
	if s.LatestStable == nil || s.LatestStable.Tag != "stable/2026-06-stable" {
		t.Fatalf("latest_stable not threaded: %+v", s.LatestStable)
	}
}

func TestBranchRegimeFold(t *testing.T) {
	cases := []struct {
		name          string
		facts         BranchRegimeFacts
		wantDrift     string
		wantBlocked   bool
		wantBlocker   string
		wantCandidate string
	}{
		{
			name:        "no drift",
			facts:       BranchRegimeFacts{DevelopmentBranch: "dev", DevelopmentHead: "aaa111", ReleaseBranch: "main", ReleaseHead: "aaa111", LatestTag: "v1.2.3", DevelopmentCI: "success"},
			wantDrift:   "no_drift",
			wantBlocked: false,
		},
		{
			name:          "development ahead is a promotion candidate",
			facts:         BranchRegimeFacts{DevelopmentBranch: "dev", DevelopmentHead: "bbb222", ReleaseBranch: "main", ReleaseHead: "aaa111", LatestTag: "v1.2.3", DevelopmentAhead: 4, DevelopmentCI: "success"},
			wantDrift:     "development_ahead",
			wantBlocked:   false,
			wantCandidate: "bbb222",
		},
		{
			name:        "release ahead blocks promotion",
			facts:       BranchRegimeFacts{DevelopmentBranch: "dev", DevelopmentHead: "bbb222", ReleaseBranch: "main", ReleaseHead: "ccc333", DevelopmentAhead: 2, ReleaseAhead: 1, DevelopmentCI: "success"},
			wantDrift:   "diverged",
			wantBlocked: true,
			wantBlocker: "RELEASE_AHEAD",
		},
		{
			name:        "red development ci blocks promotion",
			facts:       BranchRegimeFacts{DevelopmentBranch: "dev", DevelopmentHead: "bbb222", ReleaseBranch: "main", ReleaseHead: "aaa111", DevelopmentAhead: 3, DevelopmentCI: "failure"},
			wantDrift:   "development_ahead",
			wantBlocked: true,
			wantBlocker: "DEVELOPMENT_CI_RED",
		},
		{
			name:        "release lock blocks promotion",
			facts:       BranchRegimeFacts{DevelopmentBranch: "dev", DevelopmentHead: "bbb222", ReleaseBranch: "main", ReleaseHead: "aaa111", DevelopmentAhead: 3, DevelopmentCI: "success", ReleaseLockHeld: true},
			wantDrift:   "development_ahead",
			wantBlocked: true,
			wantBlocker: "RELEASE_LOCK_HELD",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := FoldBranchRegime(tc.facts)
			if got.Drift != tc.wantDrift || got.PromotionBlocked != tc.wantBlocked {
				t.Fatalf("drift/blocked = %q/%v, want %q/%v (%+v)", got.Drift, got.PromotionBlocked, tc.wantDrift, tc.wantBlocked, got)
			}
			if tc.wantCandidate != "" && got.PromotionCandidate != tc.wantCandidate {
				t.Fatalf("promotion candidate = %q, want %q", got.PromotionCandidate, tc.wantCandidate)
			}
			if tc.wantBlocker != "" && !contains(got.PromotionBlockers, tc.wantBlocker) {
				t.Fatalf("promotion blockers = %v, want %s", got.PromotionBlockers, tc.wantBlocker)
			}
			if got.LatestTag != tc.facts.LatestTag {
				t.Fatalf("latest tag = %q, want %q", got.LatestTag, tc.facts.LatestTag)
			}
			if got.NextAction == "" {
				t.Fatalf("next action empty: %+v", got)
			}
		})
	}
}

// TestRenderHumanShape proves Render emits the python render_human() head lines.
func TestRenderHumanShape(t *testing.T) {
	s := Fold(Facts{
		Decision:        Decision{Decision: "hold", Reason: "soak window not met"},
		LastTag:         "v1.2.3",
		CommitsSinceTag: 4,
	})
	out := Render(s)
	if !strings.HasPrefix(out, "release-status: HOLD - soak window not met") {
		t.Fatalf("render head line wrong:\n%s", out)
	}
	if !strings.Contains(out, "last tag: v1.2.3") || !strings.Contains(out, "commits since tag: 4") {
		t.Fatalf("render missing tag/commit lines:\n%s", out)
	}
	if !strings.Contains(out, "stable: none") {
		t.Fatalf("render missing stable line:\n%s", out)
	}
}
