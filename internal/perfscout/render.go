package perfscout

import (
	"encoding/json"
	"fmt"
	"strings"
)

// RenderMarkdown renders the complete inventory report into a structured Markdown document.
func RenderMarkdown(report InventoryReport) string {
	var b strings.Builder

	b.WriteString("# Next-Gen OSS Performance Scout: Qwen 3.8 & GLM 5.3 Flash\n\n")
	b.WriteString(fmt.Sprintf("- **Generated At**: %s\n", report.GeneratedAt.Format("2006-01-02 15:04:05 UTC")))
	b.WriteString(fmt.Sprintf("- **Total Candidate Repos Scored**: %d\n", report.TotalScored))
	b.WriteString(fmt.Sprintf("- **Retained Performance Repos**: %d\n", report.RetainedCount))
	b.WriteString(fmt.Sprintf("- **Qwen 3.8 Flash Repos**: %d\n", report.QwenCount))
	b.WriteString(fmt.Sprintf("- **GLM 5.3 Flash Repos**: %d\n", report.GLMCount))
	b.WriteString(fmt.Sprintf("- **Cross-Model Dual Repos**: %d\n\n", report.DualCount))

	b.WriteString("## Performance Inventory Table\n\n")
	b.WriteString("| Repo | Stars | Model | Engine | Hardware | Quant | Proof / tok/s | Grade | Score |\n")
	b.WriteString("|---|---|---|---|---|---|---|---|---|\n")

	for _, r := range report.Repositories {
		shortName := r.FullName
		if len(shortName) > 28 {
			shortName = shortName[:25] + "..."
		}
		repoLink := fmt.Sprintf("[%s](%s)", shortName, r.URL)

		proof := r.PerformanceProof
		if proof == "" {
			proof = "Recipe"
		}
		if len(proof) > 22 {
			proof = proof[:19] + "..."
		}

		b.WriteString(fmt.Sprintf("| %s | %d | %s | %s | %s | %s | %s | %s | %d |\n",
			repoLink,
			r.StargazersCount,
			r.TargetModel,
			r.ServingEngine,
			r.HardwareTarget,
			r.Quantization,
			proof,
			r.EvidenceGrade,
			r.PerformanceScore,
		))
	}

	b.WriteString("\n## Subagent Cohort Breakdown\n\n")
	for cohortID := 1; cohortID <= len(report.Cohorts); cohortID++ {
		repos := report.Cohorts[cohortID]
		b.WriteString(fmt.Sprintf("### Cohort %d (%d Repositories)\n\n", cohortID, len(repos)))
		for _, r := range repos {
			b.WriteString(fmt.Sprintf("- **[%s](%s)** (`%s`): %s\n", r.FullName, r.URL, r.TargetModel, r.Description))
			b.WriteString(fmt.Sprintf("  - *Hardware*: %s | *Engine*: %s | *Quant*: %s | *Proof*: %s\n",
				r.HardwareTarget, r.ServingEngine, r.Quantization, r.PerformanceProof))
			if len(r.SpecialMechanisms) > 0 {
				b.WriteString(fmt.Sprintf("  - *Mechanisms*: %s\n", strings.Join(r.SpecialMechanisms, ", ")))
			}
		}
		b.WriteString("\n")
	}

	return b.String()
}

// RenderJSON serializes the inventory report to indented JSON bytes.
func RenderJSON(report InventoryReport) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

// RenderSubagentPrompt generates the exact task prompt for an autonomous subagent inspecting a cohort.
func RenderSubagentPrompt(cohortID int, repos []ScoredRepo) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Deeply inventory Cohort %d (%d OSS performance repos running Qwen 3.8 Flash or GLM 5.3 Flash).\n\n", cohortID, len(repos)))
	b.WriteString("Your goal is to inspect the code, scripts, configurations, and benchmark evidence in each repository, verify the concrete performance claims, and produce a durable, evidence-backed inventory record.\n\n")
	b.WriteString("Repositories to inspect:\n")
	for _, r := range repos {
		b.WriteString(fmt.Sprintf("1. %s (%s)\n", r.FullName, r.URL))
		b.WriteString(fmt.Sprintf("   - Target: %s on %s\n", r.TargetModel, r.HardwareTarget))
		b.WriteString(fmt.Sprintf("   - Claimed Engine/Proof: %s | %s\n", r.ServingEngine, r.PerformanceProof))
	}
	b.WriteString("\nFor EACH repository, investigate and extract:\n")
	b.WriteString("1. Repository Overview: Exact model weight & variant tested, repo license, author's goal.\n")
	b.WriteString("2. Hardware & Setup: Exact GPU/accelerator topology, memory footprint, host environment.\n")
	b.WriteString("3. Serving Stack & Quantization: Engine (vLLM, SGLang, llama.cpp, custom), exact quantization format (NVFP4, EXL3, AWQ, FP8).\n")
	b.WriteString("4. Measured Performance Metrics: Measured decode tok/s, prefill tok/s, TTFT, concurrency scaling, context length.\n")
	b.WriteString("5. Architectural Innovations & Patches: Custom kernels, MTP speculative decoding heads, SSD/NVMe streaming, ROCm/Vulkan/CUDA graph tweaks.\n")
	b.WriteString("6. Reproducibility & Traps: Are configs/scripts runnable? What pitfalls or bugs did the author encounter?\n")
	b.WriteString("7. Relevance to fak: What can the fak kernel borrow or learn from this repository?\n\n")
	b.WriteString("Return a structured Markdown report with concrete quotes, file paths, and empirical evidence.")
	return b.String()
}
