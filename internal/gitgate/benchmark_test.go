package gitgate

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

var (
	benchLawSink         string
	benchDeniedSink      bool
	benchVerdictSink     abi.Verdict
	benchFindingSink     CollectiveFinding
	benchSweepSink       SweepGuardFinding
	benchPathSink        string
	benchBoolSink        bool
	benchSegsSink        [][]string
	benchStringsSink     []string
	benchMaintResultSink MaintResult
)

func BenchmarkClassify(b *testing.B) {
	g := New()
	cases := []struct {
		name string
		cmd  string
	}{
		{"HazardForcePush", "git push --force origin main"},
		{"HazardAmend", "git commit --amend --no-edit"},
		{"HazardAddAll", "git add -A"},
		{"HazardInteractiveRebase", "git rebase -i HEAD~3"},
		{"HazardUnscopedStash", "git stash"},
		{"HazardResetHard", "git reset --hard HEAD~1"},
		{"HazardPushDeleteRefspec", "git push origin :dead-branch"},
		{"HazardSubshellBashC", `bash -c "git push -f origin main"`},
		{"HazardCmdSubst", "echo $(git commit --amend)"},
		{"SafeStatus", "git status"},
		{"SafeCommitScoped", `git commit -m "fix: safe" -- internal/gitgate/gitgate.go`},
		{"SafeNonGit", "cargo build --release"},
		{"SafeEchoMentionsHazard", "echo git push --force"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchLawSink, benchDeniedSink = g.Classify(tc.cmd)
			}
		})
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	g := New()
	ctx := context.Background()

	hazardCall := &abi.ToolCall{
		Tool: "bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"git push --force origin main"}`)},
	}
	safeCall := &abi.ToolCall{
		Tool: "bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"git status"}`)},
	}
	nonShellCall := &abi.ToolCall{
		Tool: "read_file",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"internal/gitgate/gitgate.go"}`)},
	}

	collectivePlanBytes, err := json.Marshal(CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{
				ID:     "writer-1",
				Leases: []string{"internal/gitgate"},
				Paths:  []string{"internal/gitgate/gitgate.go"},
			},
		},
		CommitPaths: []string{"internal/gitgate/gitgate.go"},
	})
	if err != nil {
		b.Fatalf("failed to marshal collective commit plan: %v", err)
	}
	collectiveCall := &abi.ToolCall{
		Tool: ToolCollectiveCommit,
		Args: abi.Ref{Kind: abi.RefInline, Inline: collectivePlanBytes},
	}

	sweepPlanBytes, err := json.Marshal(SweepGuardPlan{
		Op:   "git commit -- internal/gitgate/gitgate.go",
		Self: "session-1",
		Live: []string{"session-1"},
		Targets: []wipattr.Attribution{
			{
				File:  "internal/gitgate/gitgate.go",
				Edit:  []string{"+ modified line"},
				State: wipattr.AttrOwned,
				Owner: "session-1",
			},
		},
	})
	if err != nil {
		b.Fatalf("failed to marshal sweep guard plan: %v", err)
	}
	sweepCall := &abi.ToolCall{
		Tool: ToolSweepGuard,
		Args: abi.Ref{Kind: abi.RefInline, Inline: sweepPlanBytes},
	}

	b.Run("DenyHazard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = g.Adjudicate(ctx, hazardCall)
		}
	})

	b.Run("DeferSafe", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = g.Adjudicate(ctx, safeCall)
		}
	})

	b.Run("DeferNonShell", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = g.Adjudicate(ctx, nonShellCall)
		}
	})

	b.Run("AllowCollectiveCommit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = g.Adjudicate(ctx, collectiveCall)
		}
	})

	b.Run("AllowSweepGuard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = g.Adjudicate(ctx, sweepCall)
		}
	})
}

func BenchmarkCheckCollectiveCommit(b *testing.B) {
	validPlan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{
				ID:     "writer-1",
				Leases: []string{"internal/gitgate", "cmd/fak"},
				Paths:  []string{"internal/gitgate/gitgate.go", "cmd/fak/main.go"},
			},
			{
				ID:     "writer-2",
				Leases: []string{"internal/engine"},
				Paths:  []string{"internal/engine/engine.go"},
			},
			{
				ID:     "writer-3",
				Leases: []string{"internal/policy"},
				Paths:  []string{"internal/policy/policy.go"},
			},
		},
		CommitPaths: []string{
			"internal/gitgate/gitgate.go",
			"cmd/fak/main.go",
			"internal/engine/engine.go",
			"internal/policy/policy.go",
		},
	}

	conflictPlan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{
				ID:     "writer-1",
				Leases: []string{"internal/gitgate"},
				Paths:  []string{"internal/gitgate/gitgate.go"},
			},
			{
				ID:     "writer-2",
				Leases: []string{"internal/gitgate/subpkg"},
				Paths:  []string{"internal/gitgate/subpkg/sub.go"},
			},
		},
		CommitPaths: []string{"internal/gitgate/gitgate.go"},
	}

	outsidePlan := CollectiveCommitPlan{
		Writers: []CollectiveWriter{
			{
				ID:     "writer-1",
				Leases: []string{"internal/gitgate"},
				Paths:  []string{"internal/engine/engine.go"},
			},
		},
		CommitPaths: []string{"internal/engine/engine.go"},
	}

	b.Run("DisjointWriters", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFindingSink = CheckCollectiveCommit(validPlan)
		}
	})

	b.Run("ConflictOverlapping", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFindingSink = CheckCollectiveCommit(conflictPlan)
		}
	})

	b.Run("OutsideLease", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFindingSink = CheckCollectiveCommit(outsidePlan)
		}
	})
}

func BenchmarkCheckSweepGuard(b *testing.B) {
	self := "session-alpha"
	peer := "session-beta"

	cleanPlan := SweepGuardPlan{
		Op:   "git commit -- internal/gitgate/gitgate.go",
		Self: self,
		Live: []string{self, peer},
		Targets: []wipattr.Attribution{
			{
				File:  "internal/gitgate/gitgate.go",
				Edit:  []string{"+ new line"},
				State: wipattr.AttrOwned,
				Owner: self,
			},
		},
	}

	peerPlan := SweepGuardPlan{
		Op:   "git commit -- internal/peer/peer.go",
		Self: self,
		Live: []string{self, peer},
		Targets: []wipattr.Attribution{
			{
				File:  "internal/peer/peer.go",
				Edit:  []string{"+ peer line"},
				State: wipattr.AttrOwned,
				Owner: peer,
			},
		},
	}

	orphanPlan := SweepGuardPlan{
		Op:   "git commit -- internal/orphan/orphan.go",
		Self: self,
		Live: []string{self, peer},
		Targets: []wipattr.Attribution{
			{
				File:  "internal/orphan/orphan.go",
				Edit:  []string{"+ orphan line"},
				State: wipattr.AttrOrphan,
			},
		},
	}

	b.Run("ClearOwn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSweepSink = CheckSweepGuard(cleanPlan)
		}
	})

	b.Run("AdvisePeer", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSweepSink = CheckSweepGuard(peerPlan)
		}
	})

	b.Run("RefuseOrphan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSweepSink = CheckSweepGuard(orphanPlan)
		}
	})
}

func BenchmarkCleanRepoPath(b *testing.B) {
	paths := []string{
		"internal/gitgate/gitgate.go",
		`internal\gitgate\..\gitgate\gitgate.go`,
		"cmd/fak/main.go",
		"docs/plans/2026-09-05.md",
		"invalid/.././../../escape",
		"",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := paths[i%len(paths)]
		benchPathSink, benchBoolSink = CleanRepoPath(p)
	}
}

func BenchmarkTreeContains(b *testing.B) {
	pairs := [][2]string{
		{"internal/gitgate", "internal/gitgate/gitgate.go"},
		{"internal/gitgate", "internal/engine/engine.go"},
		{"cmd/fak", "cmd/fak"},
		{"internal", "internal/abi/types.go"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pair := pairs[i%len(pairs)]
		benchBoolSink = TreeContains(pair[0], pair[1])
	}
}

func BenchmarkCoveredByAnyTree(b *testing.B) {
	trees := []string{
		"internal/gitgate",
		"internal/abi",
		"cmd/fak",
		"docs",
	}
	paths := []string{
		"internal/gitgate/gitgate.go",
		"internal/engine/engine.go",
		"cmd/fak/main.go",
		"unknown/path/file.txt",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := paths[i%len(paths)]
		benchBoolSink = CoveredByAnyTree(p, trees)
	}
}

func BenchmarkTokenizeSegments(b *testing.B) {
	commands := []string{
		"git status",
		"git commit -m 'commit message with \"quotes\" and flags --force' -- internal/gitgate/gitgate.go",
		"echo hello; git fetch origin main && git merge origin/main || echo failed",
		`bash -c "git push -f origin main" > /dev/null 2>&1`,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := commands[i%len(commands)]
		benchSegsSink = tokenizeSegments(cmd)
	}
}

func BenchmarkUnwrapShellSources(b *testing.B) {
	commands := []string{
		"git status",
		`bash -c "sh -c 'git push --force origin main'"`,
		"echo $(git commit --amend) && `git push -f`",
		"cat << 'EOF'\ngit push --force\nEOF\ngit status",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := commands[i%len(commands)]
		benchStringsSink = unwrapShellSources(cmd)
	}
}

func BenchmarkRunMaint(b *testing.B) {
	ctx := context.Background()
	tmpDir := b.TempDir()
	gitDir := filepath.Join(tmpDir, ".git")
	if err := os.MkdirAll(filepath.Join(gitDir, "objects"), 0o755); err != nil {
		b.Fatal(err)
	}

	runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
		if len(args) >= 3 && args[0] == "config" && args[1] == "--get" {
			switch args[2] {
			case "gc.auto":
				return "0\n", 0, nil
			case "maintenance.auto":
				return "false\n", 0, nil
			case "core.fsmonitor":
				return "\n", 1, nil // unset -> safe
			case "core.untrackedcache":
				return "true\n", 0, nil
			}
		}
		if len(args) >= 1 && args[0] == "count-objects" {
			return "count: 5\nsize: 10\nin-pack: 100\npacks: 2\nsize-pack: 500\nprune-packable: 0\ngarbage: 0\nsize-garbage: 0\n", 0, nil
		}
		return "", 0, nil
	}

	optsApply := MaintOptions{
		RepoRoot:     tmpDir,
		GitCommonDir: gitDir,
		Apply:        true,
	}
	optsDry := MaintOptions{
		RepoRoot:     tmpDir,
		GitCommonDir: gitDir,
		Apply:        false,
	}

	b.Run("DryRun", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchMaintResultSink = RunMaint(ctx, runner, optsDry)
		}
	})

	b.Run("ApplySafe", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchMaintResultSink = RunMaint(ctx, runner, optsApply)
		}
	})
}
