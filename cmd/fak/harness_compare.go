package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
)

// HarnessComparisonEntry represents a single external harness compared against the fak native harness.
type HarnessComparisonEntry struct {
	Key                  string                        `json:"key"`
	Name                 string                        `json:"name"`
	Role                 string                        `json:"role"`
	IntegrationMode      string                        `json:"integration_mode"`
	FieldBorrowingPoints []string                      `json:"field_borrowing_points"`
	ArchitecturalSeams   []HarnessArchitecturalFeature `json:"architectural_seams"`
	UpstreamAdaptation   []HarnessAdaptationItem       `json:"upstream_adaptation"`
	NBABenchmarkTargets  HarnessNBATargets             `json:"nba_benchmark_targets"`
}

// HarnessArchitecturalFeature details one architectural dimension contrasting native vs external harness.
type HarnessArchitecturalFeature struct {
	Dimension      string `json:"dimension"`
	NativeFeature  string `json:"native_feature"`
	ExternalStatus string `json:"external_status"`
	Advantage      string `json:"advantage"`
}

// HarnessAdaptationItem details how a capability adapts (or not) back to external harnesses.
type HarnessAdaptationItem struct {
	Feature   string `json:"feature"`
	Status    string `json:"status"` // "adaptable" or "native_only"
	Mechanism string `json:"mechanism"`
}

// HarnessNBATargets defines the empirical next-best alternative benchmarking targets.
type HarnessNBATargets struct {
	Workload            string  `json:"workload"`
	TurnEfficiencyRatio float64 `json:"turn_efficiency_ratio_target"`
	TokenEconomyRatio   float64 `json:"token_economy_ratio_target"`
	MemoryFootprintMB   int     `json:"memory_footprint_mb_target"`
	SecurityContainment string  `json:"security_containment"`
}

// HarnessComparisonReport is the top-level machine and human report for harness comparisons.
type HarnessComparisonReport struct {
	Schema      string                   `json:"schema"`
	Premise     string                   `json:"premise"`
	Strategy    string                   `json:"strategy"`
	Comparisons []HarnessComparisonEntry `json:"comparisons"`
}

var canonicalHarnessComparisons = []HarnessComparisonEntry{
	{
		Key:             "opencode",
		Name:            "OpenCode",
		Role:            "Tuned Open-Source Baseline / Fast Terminal Loop",
		IntegrationMode: "Tier-1 Supported via OpenAI Chat Completions wire and MCP endpoint (fak serve, fak manage)",
		FieldBorrowingPoints: []string{
			"Lightweight terminal UX and low-ceremony CLI execution loop",
			"Broad multi-provider configuration model without proprietary lock-in",
			"Transparent tool definitions with lowercase conventions and camelCase arguments",
		},
		ArchitecturalSeams: []HarnessArchitecturalFeature{
			{
				Dimension:      "Turn Loop Control",
				NativeFeature:  "In-kernel RunArm dispatch loop",
				ExternalStatus: "External Node/TypeScript execution runtime",
				Advantage:      "Native loop owns turn boundaries, eliminating inter-process RPC and subprocess churn",
			},
			{
				Dimension:      "Tool Caching",
				NativeFeature:  "In-kernel vDSO FastPath (sub-µs read hits)",
				ExternalStatus: "Direct subprocess spawns on every tool invocation",
				Advantage:      "Repeated Grep/Glob and unchanged file reads resolve without engine round-trips",
			},
			{
				Dimension:      "Write Barrier",
				NativeFeature:  "Proactive pre-consumption barWrite",
				ExternalStatus: "Reactive (mutations land on disk unchecked)",
				Advantage:      "Squashed speculative reads strictly bar follow-on mutations before disk writes occur",
			},
			{
				Dimension:      "Grammar Repair",
				NativeFeature:  "In-syscall abi.VerdictTransform",
				ExternalStatus: "Round-trip error return to model",
				Advantage:      "Argument aliases and parameter typos repaired in-syscall without consuming model turns",
			},
			{
				Dimension:      "Multi-Worker Memory",
				NativeFeature:  "<25 MiB/seat (co-hosted single-process)",
				ExternalStatus: "~400-600 MiB per process",
				Advantage:      "Runs 20+ headless workers concurrently without host memory exhaustion",
			},
			{
				Dimension:      "State Persistence",
				NativeFeature:  "Zero-daemon append-only WAL (PendingTurn)",
				ExternalStatus: "Local JSON files or in-memory session",
				Advantage:      "Crash-consistent turn resumption without daemon dependencies",
			},
			{
				Dimension:      "Capability Gate",
				NativeFeature:  "In-kernel default-deny capability monitor",
				ExternalStatus: "Client-side prompt steering or unconfined shell",
				Advantage:      "Deterministic security floor the model cannot talk past",
			},
		},
		UpstreamAdaptation: []HarnessAdaptationItem{
			{Feature: "Session budgets & pacing", Status: "adaptable", Mechanism: "Gateway preflight and MCP middleware"},
			{Feature: "Policy capability floor", Status: "adaptable", Mechanism: "fak serve proxy and fak preflight"},
			{Feature: "Context shedding & byte splicing", Status: "adaptable", Mechanism: "Gateway reverse proxy (OpenAI Chat wire)"},
			{Feature: "Trajectory & audit logging", Status: "adaptable", Mechanism: "Gateway hash-chained decision journal"},
			{Feature: "In-kernel vDSO tool cache", Status: "native_only", Mechanism: "Requires in-syscall interception before engine dispatch"},
			{Feature: "Proactive write barrier", Status: "native_only", Mechanism: "Requires turn speculation and epoch control"},
			{Feature: "In-syscall grammar repair", Status: "native_only", Mechanism: "External runtime controls deserialization"},
			{Feature: "Single-process worker co-hosting", Status: "native_only", Mechanism: "External harness requires separate OS processes"},
		},
		NBABenchmarkTargets: HarnessNBATargets{
			Workload:            "multi-turn coding agent benchmark with read/write/edit/bash tools across repo tasks",
			TurnEfficiencyRatio: 0.80,
			TokenEconomyRatio:   0.70,
			MemoryFootprintMB:   50,
			SecurityContainment: "100% default-deny capability floor (0 unadjudicated mutations)",
		},
	},
	{
		Key:             "codex",
		Name:            "OpenAI Codex",
		Role:            "Proprietary Frontier Coding CLI & Responses Protocol",
		IntegrationMode: "Tier-1 Supported via OpenAI Responses wire, fak codex launcher, and MCP server bridge",
		FieldBorrowingPoints: []string{
			"Deep reasoning integration and high-effort reasoning presets",
			"Structured task and plan management via update_plan idioms",
			"Clean responses-wire lifecycle and streaming item events",
		},
		ArchitecturalSeams: []HarnessArchitecturalFeature{
			{
				Dimension:      "Turn Loop Control",
				NativeFeature:  "In-kernel RunArm dispatch loop",
				ExternalStatus: "Closed client binary with remote OpenAI backend",
				Advantage:      "Local kernel ownership enables local vDSO interception and offline deterministic simulation",
			},
			{
				Dimension:      "Tool Caching",
				NativeFeature:  "In-kernel vDSO FastPath (sub-µs read hits)",
				ExternalStatus: "Remote API round-trip per turn",
				Advantage:      "Sub-microsecond local cache hits avoid costly model round-trips",
			},
			{
				Dimension:      "Write Barrier",
				NativeFeature:  "Proactive pre-consumption barWrite",
				ExternalStatus: "Tool execution within sandboxed container",
				Advantage:      "Prevents invalid writes before disk commit, reducing sandbox rollbacks",
			},
			{
				Dimension:      "Grammar Repair",
				NativeFeature:  "In-syscall abi.VerdictTransform",
				ExternalStatus: "Model retry turns for parameter mismatches",
				Advantage:      "Repairs snake_case / functions.* namespace aliases in-syscall",
			},
			{
				Dimension:      "Multi-Worker Memory",
				NativeFeature:  "<25 MiB/seat (co-hosted single-process)",
				ExternalStatus: "~600-800 MiB per process",
				Advantage:      "Eliminates duplicate Codex runtime processes across worker waves",
			},
			{
				Dimension:      "State Persistence",
				NativeFeature:  "Zero-daemon append-only WAL (PendingTurn)",
				ExternalStatus: "Remote server-side session or local SQLite",
				Advantage:      "Zero-dependency crash consistency",
			},
			{
				Dimension:      "Capability Gate",
				NativeFeature:  "In-kernel default-deny capability monitor",
				ExternalStatus: "Sandbox-only containment",
				Advantage:      "Enforces fine-grained semantic path/tool permissions",
			},
		},
		UpstreamAdaptation: []HarnessAdaptationItem{
			{Feature: "Tool dialect mapping", Status: "adaptable", Mechanism: "HarnessProfile in fak codex / fak manage"},
			{Feature: "MCP server integration", Status: "adaptable", Mechanism: "fak serve --stdio bridge"},
			{Feature: "Token telemetry extraction", Status: "adaptable", Mechanism: "internal/vcacheextract JSONL parser"},
			{Feature: "In-kernel vDSO tool cache", Status: "native_only", Mechanism: "Requires in-syscall interception before engine dispatch"},
			{Feature: "Proactive write barrier", Status: "native_only", Mechanism: "Requires turn speculation and epoch control"},
			{Feature: "Single-process worker co-hosting", Status: "native_only", Mechanism: "External harness requires separate OS processes"},
		},
		NBABenchmarkTargets: HarnessNBATargets{
			Workload:            "multi-turn coding agent benchmark with read/write/edit/bash tools across repo tasks",
			TurnEfficiencyRatio: 0.85,
			TokenEconomyRatio:   0.75,
			MemoryFootprintMB:   50,
			SecurityContainment: "100% default-deny capability floor (0 unadjudicated mutations)",
		},
	},
	{
		Key:             "cursor",
		Name:            "Cursor",
		Role:            "IDE-Native Workspace Agent & Editor Subsystem",
		IntegrationMode: "Tier-1 Supported via Custom Base URL (OpenAI Chat wire) and MCP Server (fak serve --stdio)",
		FieldBorrowingPoints: []string{
			"Fluid diff generation, inline review, and atomic file application",
			"Rich workspace contextual indexing and background semantic search",
			"Operator interruptibility and seamless steering mid-generation",
		},
		ArchitecturalSeams: []HarnessArchitecturalFeature{
			{
				Dimension:      "Turn Loop Control",
				NativeFeature:  "In-kernel RunArm dispatch loop",
				ExternalStatus: "IDE-embedded extension loop",
				Advantage:      "Headless-first architecture optimized for automated fleets and CI/CD rather than desktop GUI",
			},
			{
				Dimension:      "Tool Caching",
				NativeFeature:  "In-kernel vDSO FastPath (sub-µs read hits)",
				ExternalStatus: "Ad-hoc editor buffer cache",
				Advantage:      "System-level cache invalidation tied to git and file modification versions",
			},
			{
				Dimension:      "Write Barrier",
				NativeFeature:  "Proactive pre-consumption barWrite",
				ExternalStatus: "Editor tab preview with manual reject",
				Advantage:      "Automated headless protection against destructive hallucinated edits",
			},
			{
				Dimension:      "Grammar Repair",
				NativeFeature:  "In-syscall abi.VerdictTransform",
				ExternalStatus: "Model turn retry",
				Advantage:      "Zero-turn alias normalization",
			},
			{
				Dimension:      "Multi-Worker Memory",
				NativeFeature:  "<25 MiB/seat (co-hosted single-process)",
				ExternalStatus: "1-2 GiB (Electron IDE runtime)",
				Advantage:      "Enables 50+ headless parallel workers without desktop UI overhead",
			},
			{
				Dimension:      "State Persistence",
				NativeFeature:  "Zero-daemon append-only WAL (PendingTurn)",
				ExternalStatus: "IDE workspace state / SQLite",
				Advantage:      "Portable, diff-friendly session journals",
			},
			{
				Dimension:      "Capability Gate",
				NativeFeature:  "In-kernel default-deny capability monitor",
				ExternalStatus: "User confirmation prompts",
				Advantage:      "Headless-safe policy adjudication without operator stalls",
			},
		},
		UpstreamAdaptation: []HarnessAdaptationItem{
			{Feature: "Workspace MCP tools", Status: "adaptable", Mechanism: "fak serve --stdio (fak_fak_adjudicate, fak_fak_syscall)"},
			{Feature: "Reverse proxy routing", Status: "adaptable", Mechanism: "fak serve custom base URL"},
			{Feature: "In-kernel vDSO tool cache", Status: "native_only", Mechanism: "Requires in-syscall interception before engine dispatch"},
			{Feature: "Proactive write barrier", Status: "native_only", Mechanism: "Requires turn speculation and epoch control"},
			{Feature: "Headless fleet co-hosting", Status: "native_only", Mechanism: "External harness requires Electron runtime"},
		},
		NBABenchmarkTargets: HarnessNBATargets{
			Workload:            "multi-turn coding agent benchmark with read/write/edit/bash tools across repo tasks",
			TurnEfficiencyRatio: 0.85,
			TokenEconomyRatio:   0.75,
			MemoryFootprintMB:   50,
			SecurityContainment: "100% default-deny capability floor (0 unadjudicated mutations)",
		},
	},
	{
		Key:             "claude",
		Name:            "Claude Code",
		Role:            "Subagent Messages Architecture & CLI Agent",
		IntegrationMode: "Tier-1 Supported via Anthropic Messages wire, OAuth credential discovery, and fak manage hooks",
		FieldBorrowingPoints: []string{
			"Bounded task decomposition primitives (Task / TodoWrite)",
			"Hierarchical markdown context conventions (CLAUDE.md / AGENTS.md)",
			"PreCompact hooks and Stop hook auto-continuation patterns",
			"Fine-grained token and provider cache telemetry transparency",
		},
		ArchitecturalSeams: []HarnessArchitecturalFeature{
			{
				Dimension:      "Turn Loop Control",
				NativeFeature:  "In-kernel RunArm dispatch loop",
				ExternalStatus: "External Node.js CLI process",
				Advantage:      "Native loop owns turn boundaries, eliminating inter-process RPC and Node runtime tax",
			},
			{
				Dimension:      "Tool Caching",
				NativeFeature:  "In-kernel vDSO FastPath (sub-µs read hits)",
				ExternalStatus: "Full bash/tool execution on each call",
				Advantage:      "Sub-microsecond hits on Grep/Glob/Read bypass model turns",
			},
			{
				Dimension:      "Write Barrier",
				NativeFeature:  "Proactive pre-consumption barWrite",
				ExternalStatus: "Reactive tool failure return",
				Advantage:      "Halts invalid speculative writes before touching the filesystem",
			},
			{
				Dimension:      "Grammar Repair",
				NativeFeature:  "In-syscall abi.VerdictTransform",
				ExternalStatus: "Model apologizes and retries malformed tool calls",
				Advantage:      "Argument aliases repaired without consuming model turns",
			},
			{
				Dimension:      "Multi-Worker Memory",
				NativeFeature:  "<25 MiB/seat (co-hosted single-process)",
				ExternalStatus: "~600 MiB per seat (Node.js runtime)",
				Advantage:      "Cuts multi-worker fleet memory from 12.9 GiB to <500 MiB",
			},
			{
				Dimension:      "State Persistence",
				NativeFeature:  "Zero-daemon append-only WAL (PendingTurn)",
				ExternalStatus: "Local JSONL session files",
				Advantage:      "Atomic crash consistency without orphan locks",
			},
			{
				Dimension:      "Capability Gate",
				NativeFeature:  "In-kernel default-deny capability monitor",
				ExternalStatus: "Built-in permission prompts",
				Advantage:      "Headless-safe policy enforcement without interactive prompt stalls",
			},
		},
		UpstreamAdaptation: []HarnessAdaptationItem{
			{Feature: "PreCompact auto-compaction", Status: "adaptable", Mechanism: "RepointSettingsFile in fak manage"},
			{Feature: "Stop hook continuation", Status: "adaptable", Mechanism: "fak manage hooks and auto-continue"},
			{Feature: "OAuth token discovery", Status: "adaptable", Mechanism: "IdentityClaude in internal/harnessprofile"},
			{Feature: "In-kernel vDSO tool cache", Status: "native_only", Mechanism: "Requires in-syscall interception before engine dispatch"},
			{Feature: "Proactive write barrier", Status: "native_only", Mechanism: "Requires turn speculation and epoch control"},
			{Feature: "In-syscall grammar repair", Status: "native_only", Mechanism: "External runtime controls deserialization"},
			{Feature: "Single-process worker co-hosting", Status: "native_only", Mechanism: "External harness requires separate OS processes"},
		},
		NBABenchmarkTargets: HarnessNBATargets{
			Workload:            "multi-turn coding agent benchmark with read/write/edit/bash tools across repo tasks",
			TurnEfficiencyRatio: 0.80,
			TokenEconomyRatio:   0.70,
			MemoryFootprintMB:   50,
			SecurityContainment: "100% default-deny capability floor (0 unadjudicated mutations)",
		},
	},
}

func runHarnessCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseline := fs.String("baseline", "all", "target harness to compare: opencode, codex, cursor, claude, or all")
	view := fs.String("view", "cli", "output format: cli or json")
	dimensions := fs.String("dimensions", "all", "filter dimensions: architecture, learning, adaptation, metrics, or all")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	selectedKey := strings.ToLower(strings.TrimSpace(*baseline))
	var filtered []HarnessComparisonEntry
	for _, entry := range canonicalHarnessComparisons {
		if selectedKey == "all" || selectedKey == "" || selectedKey == entry.Key {
			filtered = append(filtered, entry)
		}
	}

	if len(filtered) == 0 {
		fmt.Fprintf(stderr, "fak harness compare: unknown baseline %q (supported: opencode, codex, cursor, claude, all)\n", *baseline)
		return 2
	}

	report := HarnessComparisonReport{
		Schema:      "fak.harness.comparison/v1",
		Premise:     "low-ego market realism; always learning; tier-1 external support up to proxy seam; mainstream work prioritizes native kernel advantages",
		Strategy:    "evaluate native fak harness against tuned next-best alternatives (NBAs); pioneer features in native kernel and adapt outward where possible",
		Comparisons: filtered,
	}

	if *view == "json" {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak harness compare: %v\n", err)
			return 1
		}
		return 0
	}

	printHarnessCompareCLI(stdout, report, *dimensions)
	return 0
}

func printHarnessCompareCLI(w io.Writer, report HarnessComparisonReport, dimFilter string) {
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintln(w, "FAK NATIVE HARNESS vs NEXT-BEST ALTERNATIVE (NBA) COMPARISON")
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "Premise:  %s\n", report.Premise)
	fmt.Fprintf(w, "Strategy: %s\n\n", report.Strategy)

	showArch := dimFilter == "all" || dimFilter == "architecture"
	showLearning := dimFilter == "all" || dimFilter == "learning"
	showAdapt := dimFilter == "all" || dimFilter == "adaptation"
	showMetrics := dimFilter == "all" || dimFilter == "metrics"

	for _, entry := range report.Comparisons {
		fmt.Fprintf(w, "--------------------------------------------------------------------------------\n")
		fmt.Fprintf(w, "TARGET: %s (%s)\n", entry.Name, entry.Role)
		fmt.Fprintf(w, "INTEGRATION MODE: %s\n", entry.IntegrationMode)
		fmt.Fprintf(w, "--------------------------------------------------------------------------------\n")

		if showLearning {
			fmt.Fprintln(w, "[LOW-EGO FIELD-BORROWING & LEARNING POINTS]")
			for _, pt := range entry.FieldBorrowingPoints {
				fmt.Fprintf(w, "  * %s\n", pt)
			}
			fmt.Fprintln(w)
		}

		if showArch {
			fmt.Fprintln(w, "[ARCHITECTURAL SEAMS & KERNEL ADVANTAGES]")
			fmt.Fprintf(w, "  %-22s | %-32s | %-28s\n", "Dimension", "fak Native Harness", entry.Name)
			fmt.Fprintf(w, "  %-22s-+-%-32s-+-%-28s\n", "----------------------", "--------------------------------", "----------------------------")
			for _, feat := range entry.ArchitecturalSeams {
				fmt.Fprintf(w, "  %-22s | %-32s | %-28s\n", feat.Dimension, feat.NativeFeature, feat.ExternalStatus)
				fmt.Fprintf(w, "    -> Win: %s\n", feat.Advantage)
			}
			fmt.Fprintln(w)
		}

		if showAdapt {
			fmt.Fprintln(w, "[UPSTREAM ADAPTATION STATUS]")
			for _, adapt := range entry.UpstreamAdaptation {
				badge := "[ADAPTABLE]  "
				if adapt.Status == "native_only" {
					badge = "[NATIVE-ONLY]"
				}
				fmt.Fprintf(w, "  %s %-32s: %s\n", badge, adapt.Feature, adapt.Mechanism)
			}
			fmt.Fprintln(w)
		}

		if showMetrics {
			fmt.Fprintln(w, "[NBA BENCHMARKING TARGETS]")
			fmt.Fprintf(w, "  * Workload:                 %s\n", entry.NBABenchmarkTargets.Workload)
			fmt.Fprintf(w, "  * Turn Efficiency Target:   fak / baseline <= %.2f (lower is better)\n", entry.NBABenchmarkTargets.TurnEfficiencyRatio)
			fmt.Fprintf(w, "  * Token Economy Target:     fak / baseline <= %.2f (lower is better)\n", entry.NBABenchmarkTargets.TokenEconomyRatio)
			fmt.Fprintf(w, "  * Memory Footprint Target:  < %d MiB/seat\n", entry.NBABenchmarkTargets.MemoryFootprintMB)
			fmt.Fprintf(w, "  * Security Floor Target:    %s\n", entry.NBABenchmarkTargets.SecurityContainment)
			fmt.Fprintln(w)
		}
	}
}
