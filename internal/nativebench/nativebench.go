// Package nativebench owns the benchmark obligations for fak-native capabilities.
// It records comparison arms, not results: a result becomes claimable only when a
// reproducible witness is attached to every required arm.
package nativebench

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type AlternativeClass string

const (
	TunedBaseline         AlternativeClass = "tuned_baseline"
	NextBest              AlternativeClass = "next_best"
	FirstClassIntegration AlternativeClass = "first_class_integration"
)

type Alternative struct {
	Name        string           `json:"name"`
	Class       AlternativeClass `json:"class"`
	Integration string           `json:"integration,omitempty"`
	Source      string           `json:"source"`
}

type Contract struct {
	Capability   string        `json:"capability"`
	NativePath   string        `json:"native_path"`
	Workload     string        `json:"workload"`
	Metrics      []string      `json:"metrics"`
	Alternatives []Alternative `json:"alternatives"`
	Witness      string        `json:"witness,omitempty"`
	Integrations []string      `json:"integrations,omitempty"`
}

type Finding struct {
	Capability string `json:"capability"`
	Reason     string `json:"reason"`
}

type Coverage struct {
	NativeLeaves      int      `json:"native_leaves"`
	CoveredLeaves     int      `json:"covered_leaves"`
	MissingLeaves     []string `json:"missing_leaves,omitempty"`
	OrphanContracts   []string `json:"orphan_contracts,omitempty"`
	DiscoveryComplete bool     `json:"discovery_complete"`
}

type Report struct {
	Contracts []Contract `json:"contracts"`
	Coverage  Coverage   `json:"coverage"`
	Findings  []Finding  `json:"findings"`
	Complete  bool       `json:"complete"`
}

var contracts = []Contract{
	{
		Capability: "tool_filtering",
		NativePath: "internal/gateway/mcp_defer.go",
		Workload:   "same tool catalog, requests, model, provider cache state, and correctness grader across every arm",
		Metrics:    []string{"task_success", "tool_recall", "input_tokens", "ttft_ms", "total_cost"},
		Alternatives: []Alternative{
			{Name: "all tool schemas (tuned no-filter baseline)", Class: TunedBaseline, Source: "in-repository A/B arm"},
			{Name: "retrieval-based tool selection (ToolRAG class)", Class: NextBest, Source: "https://arxiv.org/abs/2403.06011"},
		},
	},
	{
		Capability: "context_compression",
		NativePath: "internal/ctxmmu",
		Workload:   "same long-horizon transcripts, model, context budget, and end-task grader across every arm",
		Metrics:    []string{"task_success", "retained_fact_recall", "input_tokens", "latency_ms", "total_cost"},
		Alternatives: []Alternative{
			{Name: "full history with provider caching (tuned no-compression baseline)", Class: TunedBaseline, Source: "in-repository A/B arm"},
			{Name: "LongLLMLingua", Class: NextBest, Source: "https://arxiv.org/abs/2310.06839"},
			{Name: "fak + LLMLingua-2 compressor", Class: FirstClassIntegration, Integration: "headroom/lingua", Source: "https://github.com/anthony-chaudhary/fak/issues/3204"},
		},
		Integrations: []string{"headroom/lingua"},
	},
}

func All() []Contract {
	out := append([]Contract(nil), contracts...)
	sort.Slice(out, func(i, j int) bool { return out[i].Capability < out[j].Capability })
	return out
}

func Validate(cs []Contract) []Finding {
	var findings []Finding
	seen := map[string]bool{}
	for _, c := range cs {
		if c.Capability == "" || seen[c.Capability] {
			findings = append(findings, Finding{c.Capability, "capability must be non-empty and unique"})
		}
		seen[c.Capability] = true
		if c.NativePath == "" {
			findings = append(findings, Finding{c.Capability, "native implementation path is required"})
		}
		if c.Workload == "" {
			findings = append(findings, Finding{c.Capability, "shared workload contract is required"})
		}
		if len(c.Metrics) == 0 {
			findings = append(findings, Finding{c.Capability, "quality, cost, and latency metrics are required"})
		}
		classes := map[AlternativeClass]int{}
		integrations := map[string]int{}
		for _, a := range c.Alternatives {
			classes[a.Class]++
			if a.Name == "" || a.Source == "" {
				findings = append(findings, Finding{c.Capability, "every alternative needs a name and source"})
			}
			if a.Class == FirstClassIntegration {
				if a.Integration == "" {
					findings = append(findings, Finding{c.Capability, "first-class integration alternative needs its integration id"})
				}
				integrations[a.Integration]++
			}
		}
		if classes[TunedBaseline] == 0 {
			findings = append(findings, Finding{c.Capability, "tuned baseline arm is required"})
		}
		if classes[NextBest]+classes[FirstClassIntegration] == 0 {
			findings = append(findings, Finding{c.Capability, "next-best alternative arm is required"})
		}
		for id, n := range integrations {
			if n > 1 {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("integration %q appears more than once", id)})
			}
		}
		declaredIntegrations := make(map[string]bool, len(c.Integrations))
		for _, id := range c.Integrations {
			if id == "" || declaredIntegrations[id] {
				findings = append(findings, Finding{c.Capability, "integration inventory entries must be non-empty and unique"})
			}
			declaredIntegrations[id] = true
			if integrations[id] == 0 {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("equivalent first-class integration %q has no comparison arm", id)})
			}
		}
		for id := range integrations {
			if !declaredIntegrations[id] {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("integration arm %q is absent from the equivalent-integration inventory", id)})
			}
		}
		if c.Witness == "" {
			findings = append(findings, Finding{c.Capability, "benchmark witness is missing"})
		}
	}
	return findings
}

func DiscoverNativeLeaves(root string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(root, "internal"))
	if err != nil {
		return nil, fmt.Errorf("discover native leaves: %w", err)
	}
	var leaves []string
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == "nativebench" {
			continue
		}
		files, err := os.ReadDir(filepath.Join(root, "internal", entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("discover native leaf %s: %w", entry.Name(), err)
		}
		hasCode := false
		for _, file := range files {
			name := file.Name()
			if !file.IsDir() && filepath.Ext(name) == ".go" && !strings.HasSuffix(name, "_test.go") {
				hasCode = true
				break
			}
		}
		if hasCode {
			leaves = append(leaves, entry.Name())
		}
	}
	sort.Strings(leaves)
	return leaves, nil
}

func AuditRoot(root string) Report {
	cs := All()
	fs := Validate(cs)
	leaves, err := DiscoverNativeLeaves(root)
	coverage := Coverage{}
	if err != nil {
		fs = append(fs, Finding{Reason: err.Error()})
	} else {
		coverage.DiscoveryComplete = true
		coverage.NativeLeaves = len(leaves)
		byLeaf := make(map[string]bool, len(cs))
		for _, c := range cs {
			nativePath := filepath.Clean(c.NativePath)
			if filepath.Ext(nativePath) == ".go" {
				nativePath = filepath.Dir(nativePath)
			}
			leaf := filepath.Base(nativePath)
			byLeaf[leaf] = true
		}
		leafSet := make(map[string]bool, len(leaves))
		for _, leaf := range leaves {
			leafSet[leaf] = true
			if byLeaf[leaf] {
				coverage.CoveredLeaves++
			} else {
				coverage.MissingLeaves = append(coverage.MissingLeaves, leaf)
			}
		}
		for leaf := range byLeaf {
			if !leafSet[leaf] {
				coverage.OrphanContracts = append(coverage.OrphanContracts, leaf)
			}
		}
		sort.Strings(coverage.OrphanContracts)
		if len(coverage.MissingLeaves) > 0 {
			fs = append(fs, Finding{Reason: fmt.Sprintf("%d native leaves have no comparison contract", len(coverage.MissingLeaves))})
		}
		if len(coverage.OrphanContracts) > 0 {
			fs = append(fs, Finding{Reason: fmt.Sprintf("%d contracts do not map to a native leaf", len(coverage.OrphanContracts))})
		}
	}
	return Report{Contracts: cs, Coverage: coverage, Findings: fs, Complete: len(fs) == 0}
}

func Audit() Report {
	root, err := os.Getwd()
	if err != nil {
		return Report{Contracts: All(), Findings: []Finding{{Reason: err.Error()}}}
	}
	return AuditRoot(root)
}
