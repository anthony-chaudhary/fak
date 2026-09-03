package perfscout

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAnalyzeRepoQwenFlash(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	raw := GitHubRawRepo{
		FullName:        "testuser/qwen-flash-spark",
		Description:     "Qwen3.8-Flash-Next on 1x DGX Spark GB10 with vLLM NVFP4: measured 47.5 tok/s code decode, MTP speculative decoding",
		URL:             "https://github.com/testuser/qwen-flash-spark",
		StargazersCount: 4,
		UpdatedAt:       "2026-09-03T10:00:00Z",
		PushedAt:        "2026-09-03T10:00:00Z",
		CreatedAt:       "2026-08-20T00:00:00Z",
		Language:        "Python",
	}

	scored := AnalyzeRepo(raw, now, 500)
	if scored.ModelFamily != FamilyQwenFlash {
		t.Fatalf("expected FamilyQwenFlash, got %v", scored.ModelFamily)
	}
	if scored.TargetModel != "Qwen3.8-Flash-Next" {
		t.Errorf("expected TargetModel Qwen3.8-Flash-Next, got %s", scored.TargetModel)
	}
	if !strings.Contains(scored.HardwareTarget, "DGX Spark") {
		t.Errorf("expected DGX Spark in hardware, got %s", scored.HardwareTarget)
	}
	if !strings.Contains(scored.ServingEngine, "vLLM") {
		t.Errorf("expected vLLM in engine, got %s", scored.ServingEngine)
	}
	if !strings.Contains(scored.Quantization, "NVFP4") {
		t.Errorf("expected NVFP4 in quant, got %s", scored.Quantization)
	}
	if !strings.Contains(scored.PerformanceProof, "47.5 tok/s") {
		t.Errorf("expected 47.5 tok/s in proof, got %s", scored.PerformanceProof)
	}
	if scored.EvidenceGrade != GradeMeasured {
		t.Errorf("expected GradeMeasured, got %v", scored.EvidenceGrade)
	}
	if scored.PerformanceScore < 70 {
		t.Errorf("expected high performance score, got %d", scored.PerformanceScore)
	}
	if !scored.UnpopularIndie {
		t.Errorf("expected UnpopularIndie to be true with 4 stars")
	}
}

func TestAnalyzeRepoGLMFlash(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	raw := GitHubRawRepo{
		FullName:        "testlab/ferrite",
		Description:     "native GLM-5.3-Flash inference engine in Rust: PDAF disaggregation, composable TP/CP/DCP axis algebra, exact MHC, WYF chunkwise GatedDeltaNet, sm_100a CUDA kernels. CPU golden-standard verified (74 tests).",
		URL:             "https://github.com/testlab/ferrite",
		StargazersCount: 1,
		UpdatedAt:       "2026-09-03T11:00:00Z",
		PushedAt:        "2026-09-03T11:00:00Z",
		CreatedAt:       "2026-08-25T00:00:00Z",
		Language:        "Rust",
	}

	scored := AnalyzeRepo(raw, now, 500)
	if scored.ModelFamily != FamilyGLMFlash {
		t.Fatalf("expected FamilyGLMFlash, got %v", scored.ModelFamily)
	}
	if scored.TargetModel != "GLM-5.3-Flash" {
		t.Errorf("expected TargetModel GLM-5.3-Flash, got %s", scored.TargetModel)
	}
	if !strings.Contains(scored.ServingEngine, "ferrite (Rust)") {
		t.Errorf("expected ferrite (Rust) in engine, got %s", scored.ServingEngine)
	}
	if scored.EvidenceGrade != GradeKernel {
		t.Errorf("expected GradeKernel, got %v", scored.EvidenceGrade)
	}
}

func TestAnalyzeRepoDualFamily(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	raw := GitHubRawRepo{
		FullName:        "clusterlab/bench-dgx",
		Description:     "Benchmark & tuning lab on dual DGX Spark: Qwen3.8-Flash-Next, DeepSeek-V4-Flash, GLM-5.3-Flash under vLLM TP2",
		URL:             "https://github.com/clusterlab/bench-dgx",
		StargazersCount: 12,
		UpdatedAt:       "2026-09-03T09:00:00Z",
		PushedAt:        "2026-09-03T09:00:00Z",
		CreatedAt:       "2026-08-28T00:00:00Z",
		Language:        "Python",
	}

	scored := AnalyzeRepo(raw, now, 500)
	if scored.ModelFamily != FamilyDual {
		t.Fatalf("expected FamilyDual, got %v", scored.ModelFamily)
	}
	if scored.EvidenceGrade != GradeEvaluation {
		t.Errorf("expected GradeEvaluation, got %v", scored.EvidenceGrade)
	}
}

func TestRunFixturePipeline(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	sample := []GitHubRawRepo{
		{
			FullName:        "userA/qwen38-rdna4",
			Description:     "Shape-specialized ROCm inference for Qwen3.8 Flash Next on RDNA4, measured 55 tok/s",
			URL:             "https://github.com/userA/qwen38-rdna4",
			StargazersCount: 15,
			UpdatedAt:       "2026-09-03T08:00:00Z",
			PushedAt:        "2026-09-03T08:00:00Z",
			CreatedAt:       "2026-08-10T00:00:00Z",
			Language:        "C++",
		},
		{
			FullName:        "userB/glm53-spark",
			Description:     "GLM-5.3-Flash at 64 tok/s on one DGX Spark — TP1 EXL3 2.05 + DFlash2 K7",
			URL:             "https://github.com/userB/glm53-spark",
			StargazersCount: 8,
			UpdatedAt:       "2026-09-03T07:00:00Z",
			PushedAt:        "2026-09-03T07:00:00Z",
			CreatedAt:       "2026-08-15T00:00:00Z",
			Language:        "Python",
		},
		{
			FullName:        "userC/mega-framework",
			Description:     "General framework supporting Qwen3.8 and GLM-5.3",
			URL:             "https://github.com/userC/mega-framework",
			StargazersCount: 50000, // popular, should be filtered out by unpopular indie filter
			UpdatedAt:       "2026-09-03T07:00:00Z",
			PushedAt:        "2026-09-03T07:00:00Z",
			CreatedAt:       "2020-01-01T00:00:00Z",
			Language:        "Python",
		},
		{
			FullName:        "userD/unrelated-repo",
			Description:     "Just another web app with react and node",
			URL:             "https://github.com/userD/unrelated-repo",
			StargazersCount: 2,
			UpdatedAt:       "2026-09-03T07:00:00Z",
			PushedAt:        "2026-09-03T07:00:00Z",
			CreatedAt:       "2026-08-01T00:00:00Z",
			Language:        "TypeScript",
		},
	}

	tmpDir := t.TempDir()
	fixturePath := filepath.Join(tmpDir, "fixture.json")
	data, err := json.Marshal(sample)
	if err != nil {
		t.Fatalf("marshal sample: %v", err)
	}
	if err := os.WriteFile(fixturePath, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	report, err := Run(SearchOptions{
		FixturePath: fixturePath,
		MaxStars:    500,
		MinScore:    20,
		MaxAgeDays:  30,
		CohortCount: 2,
		Now:         now,
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if report.TotalFetched != 4 {
		t.Errorf("expected 4 fetched, got %d", report.TotalFetched)
	}
	if report.RetainedCount != 2 {
		t.Errorf("expected 2 retained (filtered mega and unrelated), got %d", report.RetainedCount)
	}
	if len(report.Cohorts) != 2 {
		t.Errorf("expected 2 cohorts, got %d", len(report.Cohorts))
	}

	md := RenderMarkdown(report)
	if !strings.Contains(md, "userA/qwen38-rdna4") || !strings.Contains(md, "userB/glm53-spark") {
		t.Errorf("expected markdown to contain both retained repos, got:\n%s", md)
	}

	prompt := RenderSubagentPrompt(1, report.Cohorts[1])
	if !strings.Contains(prompt, "Cohort 1") {
		t.Errorf("expected subagent prompt to contain Cohort 1, got:\n%s", prompt)
	}
}
