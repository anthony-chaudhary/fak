package conceptbench

import (
	"testing"
)

// repoRoot is the repo root relative to internal/conceptbench — the same base
// scenario_refusal_test uses to load the repo's own dos.toml, so the ship-stamp
// lint reads the LIVE lane taxonomy (a real membership check, not a recording).
const repoRoot = "../.."

// diffWitnessedAudit is a dos_verify + dos_commit_audit response recorded from a
// live referee for a commit whose diff backs its subject: shipped, verdict OK,
// witness diff-witnessed. Binding it to the fixture keeps the grade reproducible
// offline while the verdict still names the exact referee (WitnessSource).
func diffWitnessedAudit() RecordedReferee {
	return RecordedReferee{
		VerifyResp:      VerifyResult{Shipped: true, Raw: `{"shipped":true,"ref":"HEAD"}`},
		CommitAuditResp: CommitAuditResult{Verdict: "OK", Witness: "diff-witnessed", Raw: `{"verdict":"OK","witness":"diff-witnessed"}`},
	}
}

// TestStampTasksAreEngineered pins the #2733 scope: >=2 tasks, one a clean
// single-file change and one spanning two leaves, each carrying the ground-truth
// touched paths and the leaf they imply — and the two-leaf task's paths really
// do span two distinct lanes (else it is not a two-leaf task).
func TestStampTasksAreEngineered(t *testing.T) {
	tasks := StampTasks()
	if len(tasks) < 2 {
		t.Fatalf("StampTasks() = %d tasks, want >=2 per the #2733 scope", len(tasks))
	}
	var sawSingle, sawSpan bool
	lint := RuleStampLint{Root: repoRoot}
	for _, task := range tasks {
		if task.Prompt == "" {
			t.Errorf("task %s has no prompt — a task must name the change to make", task.Name)
		}
		if len(task.TouchedPaths) == 0 {
			t.Errorf("task %s has no touched paths — the over-staging check needs a ground-truth set", task.Name)
		}
		// The lint over the task's own touched paths must resolve a lane, and the
		// task's ExpectLeaf must be an acceptable stamp for them — so the fixtures
		// stay legal as the lane taxonomy evolves.
		ln := lint.LintStamp("feat("+task.ExpectLeaf+"): touch it (fak "+task.ExpectLeaf+")", task.TouchedPaths)
		if len(ln.PathLanes) == 0 {
			t.Errorf("task %s: no lane inferred for touched paths %v", task.Name, task.TouchedPaths)
		}
		if !ln.LeafMatches {
			t.Errorf("task %s: ExpectLeaf %q does not match touched paths %v (lanes %v)", task.Name, task.ExpectLeaf, task.TouchedPaths, ln.PathLanes)
		}
		switch len(task.TouchedPaths) {
		case 1:
			sawSingle = true
		default:
			sawSpan = true
			if len(ln.PathLanes) < 2 {
				t.Errorf("task %s is meant to span two leaves but its paths resolve to %v", task.Name, ln.PathLanes)
			}
		}
	}
	if !sawSingle || !sawSpan {
		t.Fatalf("want both a single-file and a spans-two-leaves task; single=%v span=%v", sawSingle, sawSpan)
	}
}

// TestGradeStampPassAndFailureClasses is the #2733 acceptance witness: a correct
// fixture commit passes, and a subject-only / wrong-leaf / off-trunk fixture each
// fails with the RIGHT recorded class (plus the over-staging and absent-trailer
// classes the scope names). Every episode grades through the real ship-stamp
// lint (RuleStampLint -> hooks.LintCommitMessage) and a real dos_verify +
// dos_commit_audit reading.
func TestGradeStampPassAndFailureClasses(t *testing.T) {
	lint := RuleStampLint{Root: repoRoot}
	single := StampTasks()[0] // touches internal/conceptbench/report.go
	witnessed := diffWitnessedAudit()

	cases := []struct {
		name    string
		task    StampTask
		commit  StampCommit
		ref     CommitReferee
		outcome StampOutcome
		pass    bool
	}{
		{
			// The fak way: verb-led subject, correct (fak conceptbench) trailer,
			// staged by explicit path, on main, diff-witnessed.
			name:    "correct_commit_passes",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733) (fak conceptbench)", StagedPaths: []string{"internal/conceptbench/report.go"}, Branch: "main", Ref: "HEAD"},
			ref:     witnessed,
			outcome: StampPass,
			pass:    true,
		},
		{
			// subject-only: a valid stamp on main, but dos_commit_audit reads
			// CLAIM_UNWITNESSED — the diff does not back the subject's claim.
			name:    "subject_only_fails_claim_unwitnessed",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733) (fak conceptbench)", StagedPaths: []string{"internal/conceptbench/report.go"}, Branch: "main", Ref: "HEAD"},
			ref:     RecordedReferee{VerifyResp: VerifyResult{Shipped: true}, CommitAuditResp: CommitAuditResult{Verdict: "CLAIM_UNWITNESSED", ClaimUnwitnessed: true, Raw: `{"verdict":"CLAIM_UNWITNESSED","witness":"subject-only"}`}},
			outcome: StampUnwitnessed,
			pass:    false,
		},
		{
			// wrong leaf: a stamp parses but names gateway, a lane the touched
			// internal/conceptbench path does not live in.
			name:    "wrong_leaf_fails",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733) (fak gateway)", StagedPaths: []string{"internal/conceptbench/report.go"}, Branch: "main", Ref: "HEAD"},
			ref:     witnessed,
			outcome: StampWrongLeaf,
			pass:    false,
		},
		{
			// off-trunk: everything else clean, but the commit landed on a
			// feature branch (the OFF_TRUNK guard refuses this).
			name:    "off_trunk_fails",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733) (fak conceptbench)", StagedPaths: []string{"internal/conceptbench/report.go"}, Branch: "fix/quick-2733", Ref: "HEAD"},
			ref:     witnessed,
			outcome: StampOffTrunk,
			pass:    false,
		},
		{
			// absent trailer: no (fak <leaf>) ship-stamp parses at all.
			name:    "absent_trailer_fails",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733)", StagedPaths: []string{"internal/conceptbench/report.go"}, Branch: "main", Ref: "HEAD"},
			ref:     witnessed,
			outcome: StampAbsentTrailer,
			pass:    false,
		},
		{
			// over-staging: a git add -A swept a sibling's in-flight file
			// (internal/gateway/mcp.go) into a commit whose task touched only
			// internal/conceptbench/report.go.
			name:    "over_staging_fails",
			task:    single,
			commit:  StampCommit{Subject: "feat(conceptbench): add a leaderboard column (#2733) (fak conceptbench)", StagedPaths: []string{"internal/conceptbench/report.go", "internal/gateway/mcp.go"}, Branch: "main", Ref: "HEAD"},
			ref:     witnessed,
			outcome: StampOverStaging,
			pass:    false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			row := GradeStamp(tc.task, tc.commit, lint, tc.ref)
			if row.Outcome != tc.outcome {
				t.Errorf("outcome = %s, want %s (evidence: %s)", row.Outcome, tc.outcome, row.Evidence)
			}
			if row.Pass != tc.pass {
				t.Errorf("pass = %v, want %v (evidence: %s)", row.Pass, tc.pass, row.Evidence)
			}
			// The row must record the SPECIFIC failure class, not just pass/fail.
			if tc.pass {
				if row.FailureClass != "" {
					t.Errorf("a pass must record no failure class, got %q", row.FailureClass)
				}
				if row.Score != 1 {
					t.Errorf("score = %v, want 1 for a pass", row.Score)
				}
			} else {
				if row.FailureClass != string(tc.outcome) {
					t.Errorf("failure_class = %q, want %q — the row must localize the failure", row.FailureClass, tc.outcome)
				}
				if row.Score != 0 {
					t.Errorf("score = %v, want 0 for a fail", row.Score)
				}
			}
			if row.WitnessSource != WitnessDosVerify+"+"+WitnessDosCommitAudit {
				t.Errorf("witness_source = %q, want the dos_verify+dos_commit_audit referee", row.WitnessSource)
			}
			if row.Evidence == "" {
				t.Error("empty evidence — the referees' reading must be auditable")
			}
		})
	}
}

// TestRuleStampLintIsRealHooksTwin proves the leaf-match rung adjudicates
// through the real path-aware ship-stamp grammar, not a canned answer: the same
// internal/conceptbench path that MATCHES a (fak conceptbench) stamp does NOT
// match a (fak gateway) stamp, and an unstamped subject parses no leaf.
func TestRuleStampLintIsRealHooksTwin(t *testing.T) {
	lint := RuleStampLint{Root: repoRoot}
	paths := []string{"internal/conceptbench/report.go"}

	ok := lint.LintStamp("feat(conceptbench): add a column (fak conceptbench)", paths)
	if !ok.StampParses || ok.Leaf != "conceptbench" || !ok.LeafMatches {
		t.Errorf("correct stamp: parses=%v leaf=%q matches=%v, want a matching conceptbench leaf (%s)", ok.StampParses, ok.Leaf, ok.LeafMatches, ok.Raw)
	}
	bad := lint.LintStamp("feat(conceptbench): add a column (fak gateway)", paths)
	if !bad.StampParses || bad.LeafMatches {
		t.Errorf("wrong leaf: parses=%v matches=%v, want parses=true matches=false (%s)", bad.StampParses, bad.LeafMatches, bad.Raw)
	}
	none := lint.LintStamp("feat(conceptbench): add a column", paths)
	if none.StampParses {
		t.Errorf("unstamped subject: parses=%v, want false (%s)", none.StampParses, none.Raw)
	}
}

// TestGradeStampTwoLeafTask exercises the spans-two-leaves task: a commit that
// picks EITHER touched leaf as its primary ship-stamp passes the leaf rung
// (membership in the touched lanes), while a stamp naming a third, untouched
// lane fails wrong_leaf — the "pick the right leaf or split" behaviour.
func TestGradeStampTwoLeafTask(t *testing.T) {
	lint := RuleStampLint{Root: repoRoot}
	span := StampTasks()[1] // touches internal/conceptbench/grade.go + internal/hooks/commitstamp.go
	witnessed := diffWitnessedAudit()
	staged := span.TouchedPaths

	for _, leaf := range []string{"conceptbench", "hooks"} {
		commit := StampCommit{Subject: "feat(" + leaf + "): span two leaves (#2733) (fak " + leaf + ")", StagedPaths: staged, Branch: "main", Ref: "HEAD"}
		if row := GradeStamp(span, commit, lint, witnessed); !row.Pass {
			t.Errorf("primary leaf %q on a two-leaf commit should pass, got %s (evidence: %s)", leaf, row.Outcome, row.Evidence)
		}
	}
	wrong := StampCommit{Subject: "feat(gateway): span two leaves (#2733) (fak gateway)", StagedPaths: staged, Branch: "main", Ref: "HEAD"}
	if row := GradeStamp(span, wrong, lint, witnessed); row.Outcome != StampWrongLeaf {
		t.Errorf("a third, untouched leaf should fail wrong_leaf, got %s (evidence: %s)", row.Outcome, row.Evidence)
	}
}
