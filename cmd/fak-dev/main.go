package main

import (
	"fmt"
	"io"
	"os"
	"runtime/debug"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/devcmd"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		writeHelp(stdout)
		return 0
	}
	switch argv[0] {
	case "version", "--version":
		return runVersion(stdout, argv[1:])
	case "index", "devindex":
		return devcmd.RunIndex(stdout, stderr, argv[1:])
	case "wiki":
		return devcmd.RunWiki(stdout, stderr, argv[1:])
	case "orient":
		return devcmd.RunOrient(stdout, stderr, argv[1:])
	case "backend":
		return devcmd.RunBackend(stdout, stderr, argv[1:])
	case "boundary":
		return devcmd.RunBoundary(stdout, stderr, argv[1:])
	case "ci-preflight":
		return devcmd.RunCIPreflight(stdout, stderr, argv[1:])
	case "checkpoint":
		return devcmd.RunCheckpoint(stdout, stderr, argv[1:])
	case "build":
		return devcmd.RunBuild(stdout, stderr, argv[1:])
	case "buildcheck":
		return devcmd.RunBuildCheck(stdout, stderr, argv[1:])
	case "blast":
		return devcmd.RunBlast(stdout, stderr, argv[1:])
	case "amd-gpu-facts":
		return devcmd.RunAMDGPUFacts(stdout, stderr, argv[1:])
	case "amd-setup":
		return devcmd.RunAMDSetup(stdout, stderr, argv[1:])
	case "commit-subject-coverage":
		return devcmd.RunCommitSubjectCoverage(stdout, stderr, argv[1:])
	case "readme-visual-audit":
		return devcmd.RunReadmeVisualAudit(stdout, stderr, argv[1:])
	case "plan-audit":
		return devcmd.RunPlanAudit(stdout, stderr, argv[1:])
	case "codex-memory":
		return devcmd.RunCodexMemory(stdout, stderr, argv[1:])
	case "sessiondiag":
		return devcmd.RunSessionDiag(stdout, stderr, argv[1:], nil)
	case "codex-tool-errors":
		return devcmd.RunCodexToolErrors(stdout, stderr, argv[1:])
	case "codex-hook-census":
		return devcmd.RunCodexHookCensus(stdout, stderr, argv[1:])
	case "codex-plugin-sync":
		return devcmd.RunCodexPluginSync(stdout, stderr, argv[1:])
	case "codex-hook-profile":
		return devcmd.RunCodexHookProfile(stdout, stderr, argv[1:])
	case "codex-stop-acceptance":
		return devcmd.RunCodexStopAcceptance(stdout, stderr, argv[1:])
	case "codex-hook-gate":
		return devcmd.RunCodexHookRecurrence(stdout, stderr, argv[1:])
	case "refactor-verify":
		return devcmd.RunRefactorVerify(stdout, stderr, argv[1:])
	case "tool-coverage-audit":
		return devcmd.RunToolCoverageAudit(stdout, stderr, argv[1:])
	case "workflow-audit":
		return devcmd.RunWorkflowAudit(stdout, stderr, argv[1:])
	case "fleetcap":
		return devcmd.RunFleetcap(stdout, stderr, argv[1:])
	case "catchup":
		return devcmd.RunCatchUpScore(stdout, stderr, argv[1:])
	case "whats-changed":
		return devcmd.RunWhatsChanged(stdout, stderr, argv[1:])
	case "windows-setup":
		return runWindowsSetup(stdout, stderr, argv[1:])
	case "feature":
		return devcmd.RunFeature(stdout, stderr, argv[1:])
	case "capabilities":
		return devcmd.RunCapabilities(stdout, stderr, argv[1:])
	case "issue":
		if len(argv) > 1 && argv[1] == "inventory" {
			return devcmd.RunIssueInventory(stdout, stderr, argv[2:])
		}
		return devcmd.RunIssue(stdout, stderr, argv[1:])
	case "issue-contract-repair":
		return devcmd.RunIssueContractRepair(stdout, stderr, argv[1:])
	case "project":
		if len(argv) > 1 && argv[1] == "completion" {
			return devcmd.RunProjectCompletion(stdout, stderr, argv[2:])
		}
		fmt.Fprintln(stderr, "usage: fak-dev project completion [flags]")
		return 2
	case "study-monitor":
		return devcmd.RunStudyMonitor(stdout, stderr, argv[1:])
	case "study-inventory":
		return devcmd.RunStudyInventory(stdout, stderr, argv[1:])
	case "study-forge":
		return devcmd.RunStudyForge(stdout, stderr, argv[1:])
	case "study-classify":
		return devcmd.RunStudyClassify(stdout, stderr, argv[1:])
	case "study-link":
		return devcmd.RunStudyLink(stdout, stderr, argv[1:])
	case "study-priority":
		return devcmd.RunStudyPriority(stdout, stderr, argv[1:])
	case "study-tickets":
		return devcmd.RunStudyTickets(stdout, stderr, argv[1:])
	case "study-adjacency":
		return devcmd.RunStudyAdjacency(stdout, stderr, argv[1:])
	case "idea-scout":
		return devcmd.RunIdeaScout(stdout, stderr, argv[1:])
	case "borrow-provenance":
		return devcmd.RunBorrowProvenance(stdout, stderr, argv[1:])
	case "perfscout":
		return devcmd.RunPerfScout(stdout, stderr, argv[1:])
	case "customization-index":
		return devcmd.RunCustomizationIndex(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak-dev: unknown command %q\n", argv[0])
		fmt.Fprintln(stderr, "run 'fak-dev help' for repository-development commands")
		return 2
	}
}

func writeHelp(w io.Writer) {
	fmt.Fprintln(w, "fak-dev — repository-development tooling for fak maintainers")
	fmt.Fprintln(w, "usage: fak-dev <command> [args...]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "commands:")
	fmt.Fprintln(w, "  index [docs] <query> [--json]         search the curated documentation map (docs is the default)")
	fmt.Fprintln(w, "  index ownership [--json|--write-manifest] [--root PATH]  audit or generate runtime/dev ownership")
	fmt.Fprintln(w, "  index policy [--json] [--root PATH]     enforce dev ownership at the repository boundary")
	fmt.Fprintln(w, "  wiki <structure|verify|fresh|score>    audit the repository documentation wiki")
	fmt.Fprintln(w, "  orient [env] --paths GLOB             show repository conventions and live ownership")
	fmt.Fprintln(w, "  backend scaffold NAME --lane LANE     generate a repository compute backend")
	fmt.Fprintln(w, "  boundary [--json] [--workspace DIR]   lint repository demo/package boundaries")
	fmt.Fprintln(w, "  ci-preflight [--ref R] [flags]        check committed repository tip in isolation")
	fmt.Fprintln(w, "  checkpoint [flags]                    append a developer progress milestone")
	fmt.Fprintln(w, "  build [--profile P] [--out F] [--receipt F] [--json]  build fak and record phase timings")
	fmt.Fprintln(w, "  buildcheck [--vet] [packages...]       compile-check while masking peer WIP")
	fmt.Fprintln(w, "  blast estimate PATH [flags]           estimate dependency blast radius")
	fmt.Fprintln(w, "  amd-gpu-facts [flags]                 inspect AMD GPU development-host counters")
	fmt.Fprintln(w, "  amd-setup [--apply] [--json] [flags]  diagnose/configure AMD GPU governor and TTM limits")
	fmt.Fprintln(w, "  commit-subject-coverage [flags]       audit repository commit subject grammar")
	fmt.Fprintln(w, "  readme-visual-audit [flags]           audit repository README visual health")
	fmt.Fprintln(w, "  plan-audit [flags]                   audit repository plan-document drift")
	fmt.Fprintln(w, "  codex-memory doctor [flags]          inspect Codex memory posture")
	fmt.Fprintln(w, "  codex-hook-profile [flags]           inspect effective Codex hook topology and readiness")
	fmt.Fprintln(w, "  sessiondiag [flags]                  diagnose abrupt Codex/fak sessions")
	fmt.Fprintln(w, "  refactor-verify [flags]              verify declaration-preserving code motion")
	fmt.Fprintln(w, "  tool-coverage-audit [flags]          audit load-bearing tool test coverage")
	fmt.Fprintln(w, "  workflow-audit [flags]               audit repository workflow refs and docs")
	fmt.Fprintln(w, "  fleetcap [flags]                      plan fleet capacity (Little's law, no live worker)")
	fmt.Fprintln(w, "  catchup [flags]                       measure repository development catch-up debt")
	fmt.Fprintln(w, "  whats-changed --paths P [flags]       report peer commits under repository paths")
	fmt.Fprintln(w, "  windows-setup [--apply] [--json]      allow native tests + fleet spine; suppress firewall prompts (one UAC prompt)")
	fmt.Fprintln(w, "  feature query <intent> [flags]        query repository and live capability cards")
	fmt.Fprintln(w, "  capabilities [intent] [flags]         query product + repository capabilities by outcome")
	fmt.Fprintln(w, "  issue <subcommand> [flags]             manage repository issue contracts and filing")
	fmt.Fprintln(w, "  issue-contract-repair [flags]          repair issue candidates into contract shape")
	fmt.Fprintln(w, "  project completion [flags]             audit project completion from issue evidence")
	fmt.Fprintln(w, "  study-monitor [flags]                  inspect the recurring study source registry")
	fmt.Fprintln(w, "  study-inventory [flags]                map local checkouts for deep study passes")
	fmt.Fprintln(w, "  study-forge <capture|validate> [flags] capture forge-history evidence")
	fmt.Fprintln(w, "  study-classify <command> [flags]       classify and validate study mechanisms")
	fmt.Fprintln(w, "  study-link <build|validate> [flags]    join study evidence to witnessed fak work")
	fmt.Fprintln(w, "  study-priority <build|validate> [flags] prioritize uncovered study joins")
	fmt.Fprintln(w, "  study-tickets <build|validate> [flags] construct audited study ticket closure")
	fmt.Fprintln(w, "  study-adjacency <validate|render> [flags] audit related-runtime adjacency")
	fmt.Fprintln(w, "  idea-scout [flags]                     plan deduplicated research issues")
	fmt.Fprintln(w, "  perfscout [flags]                      search and inventory Qwen 3.8 & GLM 5.3 Flash repos")
	fmt.Fprintln(w, "  borrow-provenance <pin|verify> [flags] bind borrowed code to upstream evidence")
	fmt.Fprintln(w, "  customization-index [flags]            inspect agent customization research")
	fmt.Fprintln(w, "  version                               print fak-dev build identity")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "The serving/guard product surface is the separately buildable 'fak' artifact.")
}

func runVersion(w io.Writer, argv []string) int {
	if len(argv) != 0 {
		return 2
	}
	version := "(devel)"
	revision := "unknown"
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" {
			version = info.Main.Version
		}
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" && setting.Value != "" {
				revision = setting.Value
			}
		}
	}
	if appversion.BuildCommit != "" {
		revision = appversion.BuildCommit
	}
	fmt.Fprintf(w, "fak-dev %s (%s)\n", version, revision)
	fmt.Fprintf(w, "build: %s\n", revision)
	return 0
}
