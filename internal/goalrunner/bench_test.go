package goalrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// BenchmarkRankHostTractability measures host string parsing and tractability ranking.
func BenchmarkRankHostTractability(b *testing.B) {
	hosts := []string{
		"full",
		"",
		"partial: GPU node",
		"blocked: CUDA node",
		"unknown: cluster",
		"PARTIAL: vLLM",
		"BLOCKED: L4",
		"  FULL  ",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, h := range hosts {
			_ = RankHostTractability(h)
		}
	}
}

// BenchmarkOrderContracts measures tractability-aware stable sorting of fleet contracts.
func BenchmarkOrderContracts(b *testing.B) {
	contracts := make([]GoalContract, 60)
	for j := 0; j < len(contracts); j++ {
		host := "full"
		if j%3 == 1 {
			host = "partial: GPU node"
		} else if j%3 == 2 {
			host = "blocked: CUDA node"
		}
		contracts[j] = GoalContract{
			N:        (j * 17) % len(contracts),
			Lane:     "goalrunner",
			Pointer:  fmt.Sprintf("issue-%d.md", j),
			Host:     host,
			Priority: j % 5,
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ordered := OrderContracts(contracts)
		if len(ordered) != len(contracts) {
			b.Fatal("unexpected contract count")
		}
	}
}

// BenchmarkFormatGoalPrompt measures prompt validation, prefix normalization, and length enforcement.
func BenchmarkFormatGoalPrompt(b *testing.B) {
	cases := []string{
		"solve issue 42",
		"/goal solve issue 42",
		strings.Repeat("medium prompt content ", 25),
		strings.Repeat("near max cap ", 250),
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, c := range cases {
			s, err := FormatGoalPrompt(c)
			if err != nil || len(s) == 0 {
				b.Fatal("unexpected prompt error")
			}
		}
	}
}

// BenchmarkScrubParentAuthEnv measures parent gateway auth token filtering from process environments.
func BenchmarkScrubParentAuthEnv(b *testing.B) {
	baseEnv := []string{
		"ANTHROPIC_API_KEY=sk-ant-test-key-12345",
		"ANTHROPIC_BASE_URL=http://localhost:8080",
		"anthropic_custom_header=token",
		"CLAUDE_CODE_SESSION_ID=sess_98765",
		"CLAUDE_CODE_CHILD_SESSION=child_54321",
		"CLAUDE_CONFIG_DIR=/path/to/claude-config",
		"CLAUDE_CODE_OAUTH_TOKEN=oauth_valid_token",
		"PATH=/usr/bin:/bin:/usr/local/bin",
		"USER=testuser",
		"HOME=/home/testuser",
		"SHELL=/bin/bash",
		"LANG=en_US.UTF-8",
		"TERM=xterm-256color",
		"GOROOT=/usr/local/go",
		"GOPATH=/home/testuser/go",
	}

	testEnv := make([]string, 0, len(baseEnv)*4)
	for j := 0; j < 4; j++ {
		for _, e := range baseEnv {
			testEnv = append(testEnv, fmt.Sprintf("%s_%d", e, j))
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		scrubbed := ScrubParentAuthEnv(testEnv)
		if len(scrubbed) == 0 {
			b.Fatal("unexpected empty scrubbed env")
		}
	}
}

// BenchmarkSerializePlanJSON measures JSON marshaling of full fleet execution plans.
func BenchmarkSerializePlanJSON(b *testing.B) {
	contracts := make([]GoalContract, 25)
	for j := 0; j < len(contracts); j++ {
		contracts[j] = GoalContract{
			N:        j + 1,
			Lane:     "goalrunner",
			Pointer:  fmt.Sprintf("contracts/issue-%d.md", j+1),
			Host:     "full",
			Priority: j % 4,
		}
	}
	plan := &FleetPlan{
		Name:             "bench-fleet",
		Workspace:        "/fak/workspace",
		RunRoot:          "/fak/workspace/.goal-runs/run-bench",
		Contracts:        contracts,
		PerWorkerTimeout: 45 * time.Minute,
		RollupPath:       "/fak/workspace/.goal-runs/run-bench/rollup.md",
		StatusPath:       "/fak/workspace/.goal-runs/run-bench/STATUS.txt",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		str, err := SerializePlanJSON(plan)
		if err != nil || len(str) == 0 {
			b.Fatal("failed to serialize plan")
		}
	}
}

// BenchmarkRenderRollup measures Markdown execution rollup report synthesis.
func BenchmarkRenderRollup(b *testing.B) {
	contracts := make([]GoalContract, 20)
	results := make([]*WitnessResult, 20)
	for j := 0; j < 20; j++ {
		contracts[j] = GoalContract{
			N:        j + 1,
			Lane:     "goalrunner",
			Pointer:  fmt.Sprintf("contracts/issue-%d.md", j+1),
			Host:     "full",
			Priority: j % 3,
		}
		results[j] = &WitnessResult{
			Issue:           j + 1,
			Outcome:         "met (witnessed ship)",
			ShippedSHA:      fmt.Sprintf("commit-%08d", j),
			ShippedVerdict:  "OK",
			ShippedWitness:  "diff-witnessed",
			BestSeenSHA:     fmt.Sprintf("commit-%08d", j),
			BestSeenVerdict: "OK",
			IssueState:      "closed",
			WorkerLog:       fmt.Sprintf(".goal-runs/run-%d.log", j+1),
			PID:             10000 + j,
			TimedOut:        false,
		}
	}
	plan := &FleetPlan{
		Name:      "bench-fleet",
		Workspace: "/fak/workspace",
		RunRoot:   "/fak/workspace/.goal-runs/run-bench",
		Contracts: contracts,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		md := RenderRollup(plan, results)
		if len(md) == 0 {
			b.Fatal("unexpected empty rollup")
		}
	}
}

// BenchmarkSweepDeadPidBreadcrumbs measures breadcrumb directory scanning and dead PID pruning.
func BenchmarkSweepDeadPidBreadcrumbs(b *testing.B) {
	tempDir := b.TempDir()
	logDir := filepath.Join(tempDir, ".goal-runs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		for j := 0; j < 5; j++ {
			pidFile := filepath.Join(logDir, fmt.Sprintf("bench-worker-%d.pid", j))
			_ = os.WriteFile(pidFile, []byte(fmt.Sprintf("%d\n", 99990000+j)), 0644)
		}
		b.StartTimer()

		swept, err := SweepDeadPidBreadcrumbs(tempDir)
		if err != nil {
			b.Fatalf("sweep failed: %v", err)
		}
		if len(swept) == 0 {
			b.Fatal("expected swept dead pids")
		}
	}
}
