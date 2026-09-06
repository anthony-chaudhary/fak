package safecommit

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

var (
	sinkResult    Result
	sinkScore     Result
	sinkVelocity  CommitVelocity
	sinkEvidence  Result
	sinkPreflight PathPreflightReport
	sinkBool      bool
	sinkString    string
	sinkStrings   []string
	sinkInt       int
	sinkMap       map[string]bool
)

// BenchmarkCommitWith_CleanCommit measures end-to-end execution of the safecommit pipeline
// on a clean, verified commit under simulated git and lock environments.
func BenchmarkCommitWith_CleanCommit(b *testing.B) {
	ctx := context.Background()
	opts := baseOpts()
	opts.Window = NewWindow(100)
	g := &fakeGit{reply: onTrunkBase()}
	lock := okLock(nil)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.calls = g.calls[:0]
		res, err := CommitWith(ctx, g.run, lock, opts)
		if err != nil || !res.Verified {
			b.Fatalf("unexpected commit failure: err=%v, res=%+v", err, res)
		}
		sinkResult = res
	}
}

// BenchmarkCommitWith_Refusals measures early pre-commit refusal paths that reject unsafe or
// conflicting commits before invoking destructive operations.
func BenchmarkCommitWith_Refusals(b *testing.B) {
	ctx := context.Background()

	b.Run("LockBusy", func(b *testing.B) {
		opts := baseOpts()
		g := &fakeGit{reply: onTrunkBase()}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.calls = g.calls[:0]
			res, _ := CommitWith(ctx, g.run, busyLock, opts)
			sinkResult = res
		}
	})

	b.Run("WindowFull", func(b *testing.B) {
		opts := baseOpts()
		opts.Window = NewWindow(1)
		rel, ok := opts.Window.TryAcquire()
		if !ok {
			b.Fatal("failed to acquire initial window slot")
		}
		defer rel(Result{})
		g := &fakeGit{reply: onTrunkBase()}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.calls = g.calls[:0]
			res, _ := CommitWith(ctx, g.run, okLock(nil), opts)
			sinkResult = res
		}
	})

	b.Run("NothingStaged", func(b *testing.B) {
		opts := baseOpts()
		g := &fakeGit{reply: onTrunkBase()}
		g.reply["status"] = reply{out: "", code: 0}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.calls = g.calls[:0]
			res, _ := CommitWith(ctx, g.run, okLock(nil), opts)
			sinkResult = res
		}
	})

	b.Run("PreStagedOverlap", func(b *testing.B) {
		opts := baseOpts()
		g := &fakeGit{reply: onTrunkBase()}
		g.reply["status"] = reply{out: "M  internal/foo/bar.go\n", code: 0}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.calls = g.calls[:0]
			res, _ := CommitWith(ctx, g.run, okLock(nil), opts)
			sinkResult = res
		}
	})

	b.Run("CoreSelfModify", func(b *testing.B) {
		opts := hardSelfOpts()
		g := &fakeGit{reply: onTrunkBase()}
		g.reply["status"] = reply{out: " M internal/corelocks/corelocks.go\n", code: 0}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.calls = g.calls[:0]
			res, _ := CommitWith(ctx, g.run, okLock(nil), opts)
			sinkResult = res
		}
	})
}

// BenchmarkScoreResult measures deterministic outcome quality scoring, continuous ratio calculation,
// and legacy grading over distinct Result states.
func BenchmarkScoreResult(b *testing.B) {
	cases := []struct {
		name string
		res  Result
	}{
		{
			name: "VerifiedDelivery",
			res: Result{
				Committed: true,
				Verified:  true,
				Evidence: &CommitEvidence{
					CompletionClass: CompletionVerifiedDelivery,
					Recorded:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					DiffWitnessed:   EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Compiled:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Tested:          EvidenceAxis{Outcome: EvidencePassed, Required: true},
				},
			},
		},
		{
			name: "VerifiedWithBuildCheckNote",
			res: Result{
				Committed: true,
				Verified:  true,
				BuildCheck: &BuildCheckResult{
					Outcome:    BuildCheckSkippedTimeout,
					Compiled:   false,
					FailedOpen: true,
				},
				Evidence: &CommitEvidence{
					CompletionClass: CompletionVerifiedDelivery,
					Recorded:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					DiffWitnessed:   EvidenceAxis{Outcome: EvidencePassed, Required: true},
				},
			},
		},
		{
			name: "RecordOnly",
			res: Result{
				Committed: true,
				Verified:  true,
				Evidence: &CommitEvidence{
					CompletionClass: CompletionRecordOnly,
					Recorded:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					DiffWitnessed:   EvidenceAxis{Outcome: EvidencePassed, Required: true},
				},
			},
		},
		{
			name: "UnverifiedCommitted",
			res: Result{
				Committed: true,
				Verified:  false,
			},
		},
		{
			name: "ContentionRefusal",
			res: Result{
				Reason: ReasonLockBusy,
			},
		},
		{
			name: "HookRefused",
			res: Result{
				Reason: ReasonHookRefused,
			},
		},
		{
			name: "PathspecRace",
			res: Result{
				Committed:  true,
				Reason:     ReasonPathspecRace,
				RacedExtra: []string{"internal/extra.go"},
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			input := tc.res
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkScore = ScoreResult(input)
			}
		})
	}
}

// BenchmarkScoreCommitVelocity measures ship-speed grading against local and push wall-clock budgets.
func BenchmarkScoreCommitVelocity(b *testing.B) {
	budgets := DefaultVelocityBudgets
	localDur := 2 * time.Second
	pushDur := 5 * time.Second

	cases := []struct {
		name string
		res  Result
	}{
		{
			name: "LocalAndPushQualified",
			res: Result{
				Committed: true,
				Verified:  true,
				Pushed:    true,
				Evidence: &CommitEvidence{
					CompletionClass: CompletionVerifiedDelivery,
					Recorded:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					DiffWitnessed:   EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Compiled:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Tested:          EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Pushed:          EvidenceAxis{Outcome: EvidencePassed, Required: true},
				},
			},
		},
		{
			name: "LocalQualifiedOnly",
			res: Result{
				Committed: true,
				Verified:  true,
				Pushed:    false,
				Evidence: &CommitEvidence{
					CompletionClass: CompletionVerifiedDelivery,
					Recorded:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					DiffWitnessed:   EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Compiled:        EvidenceAxis{Outcome: EvidencePassed, Required: true},
					Tested:          EvidenceAxis{Outcome: EvidencePassed, Required: true},
				},
			},
		},
		{
			name: "UnqualifiedRetainedTiming",
			res: Result{
				Committed: false,
				Verified:  false,
				Reason:    ReasonLockBusy,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			input := tc.res
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkVelocity = ScoreCommitVelocity(input, localDur, pushDur, budgets)
			}
		})
	}
}

// BenchmarkFinalizeEvidence measures aggregation and verification of completion contracts.
func BenchmarkFinalizeEvidence(b *testing.B) {
	cases := []struct {
		name     string
		res      Result
		contract EvidenceContract
	}{
		{
			name: "VerifiedDelivery_Passed",
			res: Result{
				Committed: true,
				Verified:  true,
				BuildCheck: &BuildCheckResult{
					Outcome:         BuildCheckPassed,
					Compiled:        true,
					CompileEvidence: EvidencePassed,
					TestEvidence:    EvidencePassed,
				},
			},
			contract: EvidenceContract{
				CompletionClass: CompletionVerifiedDelivery,
			},
		},
		{
			name: "VerifiedDelivery_MissingTest",
			res: Result{
				Committed: true,
				Verified:  true,
				BuildCheck: &BuildCheckResult{
					Outcome:         BuildCheckPassed,
					Compiled:        true,
					CompileEvidence: EvidencePassed,
					TestEvidence:    EvidenceFailed,
				},
			},
			contract: EvidenceContract{
				CompletionClass: CompletionVerifiedDelivery,
			},
		},
		{
			name: "RecordOnly",
			res: Result{
				Committed: true,
				Verified:  true,
			},
			contract: EvidenceContract{
				CompletionClass: CompletionRecordOnly,
			},
		},
		{
			name: "StrictClosure",
			res: Result{
				Committed: true,
				Verified:  true,
				Pushed:    true,
				BuildCheck: &BuildCheckResult{
					Outcome:         BuildCheckPassed,
					Compiled:        true,
					CompileEvidence: EvidencePassed,
					TestEvidence:    EvidencePassed,
				},
			},
			contract: EvidenceContract{
				CompletionClass: CompletionVerifiedDelivery,
				RequirePush:     true,
				RequireClosure:  true,
				ClosureBound:    true,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			input := tc.res
			contract := tc.contract
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkEvidence = FinalizeEvidence(input, contract)
			}
		})
	}
}

// BenchmarkClassifyPaths measures path-preflight inspection of index and worktree state.
func BenchmarkClassifyPaths(b *testing.B) {
	ctx := context.Background()

	b.Run("SingleTracked", func(b *testing.B) {
		paths := []string{"internal/safecommit/safecommit.go"}
		runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
			if len(args) > 0 && args[0] == "rev-parse" {
				return ".git", 0, nil
			}
			if len(args) > 2 && args[2] == "--cached" {
				return "internal/safecommit/safecommit.go\x00", 0, nil
			}
			return "", 0, nil
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep, _ := ClassifyPaths(ctx, runner, "/repo", paths)
			sinkPreflight = rep
		}
	})

	b.Run("MultiTracked_10", func(b *testing.B) {
		paths := make([]string, 10)
		for i := 0; i < 10; i++ {
			paths[i] = fmt.Sprintf("internal/pkg/file_%d.go", i)
		}
		runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
			if len(args) > 0 && args[0] == "rev-parse" {
				return ".git", 0, nil
			}
			if len(args) > 2 && args[2] == "--cached" {
				return "some/file.go\x00", 0, nil
			}
			return "", 0, nil
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep, _ := ClassifyPaths(ctx, runner, "/repo", paths)
			sinkPreflight = rep
		}
	})

	b.Run("MixedStates", func(b *testing.B) {
		paths := []string{"tracked.go", "untracked.go", "unmatched.go"}
		runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
			if len(args) > 0 && args[0] == "rev-parse" {
				return ".git", 0, nil
			}
			last := args[len(args)-1]
			if len(args) > 2 && args[2] == "--cached" {
				if last == "tracked.go" {
					return "tracked.go\x00", 0, nil
				}
				return "", 0, nil
			}
			if len(args) > 2 && args[2] == "--others" {
				if last == "untracked.go" {
					return "untracked.go\x00", 0, nil
				}
				return "", 0, nil
			}
			return "", 0, nil
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep, _ := ClassifyPaths(ctx, runner, "/repo", paths)
			sinkPreflight = rep
		}
	})
}

// BenchmarkAdaptiveWindow measures concurrency admission and AIMD modulation.
func BenchmarkAdaptiveWindow(b *testing.B) {
	b.Run("Sequential_AcquireRelease", func(b *testing.B) {
		w := NewWindow(10)
		resSuccess := Result{Committed: true, Verified: true}
		resFailure := Result{Reason: ReasonHookRefused}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			release, ok := w.TryAcquire()
			if ok && release != nil {
				if i%10 == 0 {
					release(resFailure)
				} else {
					release(resSuccess)
				}
			}
		}
	})

	b.Run("Parallel_Contention", func(b *testing.B) {
		w := NewWindow(4)
		resSuccess := Result{Committed: true, Verified: true}
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				release, ok := w.TryAcquire()
				if ok && release != nil {
					release(resSuccess)
				}
			}
		})
	})
}

// BenchmarkGitLockContention measures classification of git error streams against transient and permanent locks.
func BenchmarkGitLockContention(b *testing.B) {
	cases := []struct {
		name string
		out  string
	}{
		{
			name: "TransientIndexLock",
			out:  "fatal: Unable to create 'C:/work/fak/.git/index.lock': File exists.\nAnother git process seems to be running in this repository.",
		},
		{
			name: "PermanentCorruptRef",
			out:  "error: cannot lock ref 'refs/heads/main': unable to resolve reference 'refs/heads/main': reference broken",
		},
		{
			name: "CleanSuccessOrUnrelated",
			out:  "error: pathspec 'foo' did not match any file(s) known to git",
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			out := tc.out
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBool = isGitLockContention(out)
			}
		})
	}
}

// BenchmarkPreStagedPathOverlap measures parsing and intersection of porcelain status entries.
func BenchmarkPreStagedPathOverlap(b *testing.B) {
	paths := []string{"internal/safecommit/safecommit.go", "internal/safecommit/window.go"}

	cleanStatus := " M other/file.go\n?? untracked.txt\n"

	var sb10 strings.Builder
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&sb10, "M  internal/pkg%d/file.go\n", i)
	}
	status10 := sb10.String()

	var sb100 strings.Builder
	for i := 0; i < 100; i++ {
		if i == 50 {
			sb100.WriteString("M  internal/safecommit/safecommit.go\n")
		} else {
			fmt.Fprintf(&sb100, " M internal/other%d/file.go\n", i)
		}
	}
	status100 := sb100.String()

	b.Run("CleanStatus", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			detail, fired := preStagedPathOverlapFromStatus(cleanStatus, paths)
			sinkString = detail
			sinkBool = fired
		}
	})

	b.Run("Status_10Entries", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			detail, fired := preStagedPathOverlapFromStatus(status10, paths)
			sinkString = detail
			sinkBool = fired
		}
	})

	b.Run("Status_100Entries", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			detail, fired := preStagedPathOverlapFromStatus(status100, paths)
			sinkString = detail
			sinkBool = fired
		}
	})
}

// BenchmarkStaleBase_DroppedPeerRun measures contiguous dropped run detection over unified diffs.
func BenchmarkStaleBase_DroppedPeerRun(b *testing.B) {
	smallDiff := "--- a/file.go\n+++ b/file.go\n@@ -1,3 +1,4 @@\n context\n+new line 1\n+new line 2\n context\n"
	peerAddedSmall := normalizedAddedLines(smallDiff)
	wtPresentSmall := map[string]bool{"new line 1": true, "new line 2": true}

	var sbAdded strings.Builder
	var sbRemoved strings.Builder
	sbRemoved.WriteString("--- a/file.go\n+++ b/file.go\n@@ -1,50 +1,0 @@\n")
	for i := 0; i < 50; i++ {
		line := fmt.Sprintf("func PeerAddedFunction%d() { return }", i)
		fmt.Fprintf(&sbAdded, "+%s\n", line)
		fmt.Fprintf(&sbRemoved, "-%s\n", line)
	}
	peerAdded50 := normalizedAddedLines(sbAdded.String())
	diffRemoved50 := sbRemoved.String()

	b.Run("NoDrop_SmallDiff", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkInt = droppedPeerRun(smallDiff, peerAddedSmall, wtPresentSmall, staleBaseMinRun)
		}
	})

	b.Run("DroppedRun_50Lines", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkInt = droppedPeerRun(diffRemoved50, peerAdded50, nil, staleBaseMinRun)
		}
	})

	b.Run("NormalizedAddedLines", func(b *testing.B) {
		diff := sbAdded.String()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkMap = normalizedAddedLines(diff)
		}
	})
}

// BenchmarkDecideBuildCheck measures evaluation of prospective tree build outcomes.
func BenchmarkDecideBuildCheck(b *testing.B) {
	cases := []struct {
		name         string
		outcome      BuildCheckOutcome
		detail       string
		allowTimeout bool
	}{
		{
			name:         "Passed",
			outcome:      BuildCheckPassed,
			detail:       "ok",
			allowTimeout: false,
		},
		{
			name:         "FailedRed",
			outcome:      BuildCheckFailed,
			detail:       "syntax error on line 42",
			allowTimeout: false,
		},
		{
			name:         "Timeout_Disallowed",
			outcome:      BuildCheckSkippedTimeout,
			detail:       "deadline exceeded",
			allowTimeout: false,
		},
		{
			name:         "Timeout_Allowed",
			outcome:      BuildCheckSkippedTimeout,
			detail:       "deadline exceeded",
			allowTimeout: true,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			outcome := tc.outcome
			detail := tc.detail
			allowTimeout := tc.allowTimeout
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, admit, reason := DecideBuildCheck(outcome, detail, allowTimeout)
				sinkBool = admit
				sinkString = reason
				_ = res
			}
		})
	}
}

// BenchmarkPathUtilities measures path normalization, argument synthesis, and message processing.
func BenchmarkPathUtilities(b *testing.B) {
	paths := []string{
		"internal/safecommit/safecommit.go",
		"internal/safecommit/window.go",
		"cmd/fak/main.go",
	}

	b.Run("NormalizePaths", func(b *testing.B) {
		input := []string{"./internal/safecommit/safecommit.go", "internal\\safecommit\\window.go", "cmd/fak/main.go"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			norm, ok := normalizePaths(input)
			sinkStrings = norm
			sinkBool = ok
		}
	})

	b.Run("BuildCommitArgs", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			args := buildCommitArgs(true, "/tmp/msg.txt", paths)
			sinkStrings = args
		}
	})

	b.Run("RacedExtra", func(b *testing.B) {
		diffTree := "internal/safecommit/safecommit.go\ninternal/safecommit/window.go\ninternal/peer/raced.go\n"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			extra := racedExtra(diffTree, paths)
			sinkStrings = extra
		}
	})

	b.Run("ComparableCommitMessage", func(b *testing.B) {
		msg := "fix(safecommit): add benchmark functions\n\nDetailed body explanation.\n\nSigned-off-by: Developer <dev@example.com>\n"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkString = comparableCommitMessage(msg)
		}
	})
}

// BenchmarkAutoIndex_InsertEntries measures markdown note index entry placement.
func BenchmarkAutoIndex_InsertEntries(b *testing.B) {
	index := `# INDEX

## Architecture

Some architectural notes.

## Notes & research (` + "`docs/notes/`" + `)

Dated working notes.

- [Existing Note 1](docs/notes/2026-06-01-note-1.md) -- note 1.
- [Existing Note 2](docs/notes/2026-05-01-note-2.md) -- note 2.

## Next Steps
`
	entries := []noteIndexEntry{
		{
			Path:  "docs/notes/2026-07-01-new-feature.md",
			Base:  "2026-07-01-new-feature.md",
			Date:  "2026-07-01",
			Title: "New Feature",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = insertNoteIndexEntries(index, entries)
	}
}

// BenchmarkLockProbe_ClassifyRemoveErr measures OS error classification for broken lock reaping.
func BenchmarkLockProbe_ClassifyRemoveErr(b *testing.B) {
	cases := []struct {
		name string
		err  error
	}{
		{"NotExist", os.ErrNotExist},
		{"Permission", os.ErrPermission},
		{"Nil", nil},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			err := tc.err
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkString = classifyRemoveErr(err)
			}
		})
	}
}
