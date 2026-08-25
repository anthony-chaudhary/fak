package devindex

import (
	"fmt"
	"strings"
)

const DiscoveryBenchmarkSchema = "fak-devindex-discovery-benchmark/1"

type DiscoveryQuestion struct {
	ID       string   `json:"id"`
	Category string   `json:"category"`
	Query    string   `json:"query"`
	Owners   []string `json:"owners"`
	TopK     int      `json:"top_k"`
}

type DiscoveryCaseResult struct {
	DiscoveryQuestion
	Success       bool     `json:"success"`
	Rank          int      `json:"rank,omitempty"`
	ResultCount   int      `json:"result_count"`
	RenderedBytes int      `json:"rendered_bytes"`
	Approximate   bool     `json:"approximate,omitempty"`
	TopPaths      []string `json:"top_paths,omitempty"`
}

type DiscoveryBenchmarkReport struct {
	Schema        string                `json:"schema"`
	Source        string                `json:"source"`
	Coverage      string                `json:"coverage"`
	Questions     int                   `json:"questions"`
	Successes     int                   `json:"successes"`
	TopKRate      float64               `json:"top_k_rate"`
	TotalResults  int                   `json:"total_results"`
	RenderedBytes int                   `json:"rendered_bytes"`
	Cases         []DiscoveryCaseResult `json:"cases"`
}

func DefaultDiscoveryQuestions() []DiscoveryQuestion {
	return []DiscoveryQuestion{
		{"shared-tree-validation", "ownership", "shared tree validation", []string{"docs/dev-tooling.md"}, 5},
		{"commit-explicit-paths", "ownership", "commit explicit paths", []string{"CONTRIBUTING.md", "docs/dev-tooling.md"}, 5},
		{"native-inference", "concept", "native inference llama fallback", []string{"docs/native-inference-goal.md"}, 5},
		{"fleet-compute", "runtime", "gpu fleet compute nodes", []string{"docs/fleet-compute-nodes.md"}, 5},
		{"private-comms", "runtime", "private comms gpu bridge", []string{"docs/private-comms-channel.md"}, 5},
		{"spine-first", "concept", "spine first fanout", []string{"docs/spine-first-defaults.md"}, 5},
		{"scratch-lifecycle", "ownership", "scratch allocation generated output", []string{"docs/generated-output-defaults.md"}, 5},
		{"ci-contract", "ownership", "ci workflow contract migration", []string{"docs/ci/ci-spec-change-migration.md"}, 5},
		{"trajectory-observability", "runtime", "trajectory corpus similarity", []string{"docs/observability/trajectory.md"}, 5},
		{"opencode-integration", "cli", "opencode integration", []string{"docs/integrations/opencode.md"}, 5},
		{"cli-reference", "cli", "every cli verb reference", []string{"docs/cli-reference.md"}, 5},
		{"policy-floor", "concept", "default deny capability policy", []string{"POLICY.md"}, 5},
		{"claims-status", "history", "real simulated stub claims", []string{"CLAIMS.md", "STATUS.md"}, 5},
		{"benchmark-authority", "history", "benchmark authority numbers", []string{"BENCHMARK-AUTHORITY.md"}, 5},
		{"rollback", "runtime", "rollback downgrade pin", []string{"docs/ROLLBACK.md"}, 5},
		{"release-channel", "history", "release channel contract", []string{"docs/releases-channel.md"}, 5},
		{"problem-frame", "concept", "problems we solve p1 p4", []string{"docs/problems-we-solve.md"}, 5},
		{"supported-features", "concept", "supported features status", []string{"docs/supported/features.md"}, 5},
		{"repro-packet", "docs", "sixty second proof offline", []string{"docs/repro-packet.md"}, 5},
		{"contributing", "docs", "human contribution guide", []string{"CONTRIBUTING.md"}, 5},
		{"architecture", "docs", "kernel architecture subsystems", []string{"ARCHITECTURE.md"}, 5},
		{"extending", "docs", "extend kernel new leaf", []string{"EXTENDING.md"}, 5},
		{"learning-path", "docs", "learning path course", []string{"LEARNING-PATH.md"}, 5},
		{"wiki", "docs", "wiki freshness structure verify", []string{"docs/wiki.md", "internal/wiki/doc.go"}, 5},
	}
}

func (c *Catalog) RunDiscoveryBenchmark(questions []DiscoveryQuestion) DiscoveryBenchmarkReport {
	report := DiscoveryBenchmarkReport{Schema: DiscoveryBenchmarkSchema, Source: "INDEX.md curated lexical docs", Coverage: "curated_docs_only", Questions: len(questions)}
	for _, q := range questions {
		hits := c.SearchDocs(q.Query)
		upto := min(q.TopK, len(hits))
		result := DiscoveryCaseResult{DiscoveryQuestion: q, ResultCount: upto}
		report.TotalResults += upto
		for i, hit := range hits[:upto] {
			result.TopPaths = append(result.TopPaths, hit.Path)
			result.RenderedBytes += len(hit.Path) + 1 + len(hit.Title) + 1
			if hit.Approx {
				result.Approximate = true
			}
			if result.Rank == 0 && pathOwnedBy(hit.Path, q.Owners) {
				result.Rank = i + 1
				result.Success = true
			}
		}
		if result.Success {
			report.Successes++
		}
		report.RenderedBytes += result.RenderedBytes
		report.Cases = append(report.Cases, result)
	}
	if report.Questions > 0 {
		report.TopKRate = float64(report.Successes) / float64(report.Questions)
	}
	return report
}

func pathOwnedBy(path string, owners []string) bool {
	p := strings.ToLower(strings.ReplaceAll(path, "\\", "/"))
	for _, owner := range owners {
		o := strings.ToLower(strings.ReplaceAll(owner, "\\", "/"))
		if p == o {
			return true
		}
	}
	return false
}

func (r DiscoveryBenchmarkReport) Summary() string {
	return fmt.Sprintf("%d/%d top-k (%.1f%%), %d results, %d rendered bytes, coverage=%s", r.Successes, r.Questions, 100*r.TopKRate, r.TotalResults, r.RenderedBytes, r.Coverage)
}
