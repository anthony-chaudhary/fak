// Command browseractionbench runs a browser/computer-use action-mediation smoke
// through fak adjudication.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/browseraction"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/webbench/action_mediation_smoke.json"), "browser action mediation suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	contractMode := flag.Bool("contract", false, "emit an external official-run contract instead of replaying the local smoke")
	harness := flag.String("official-harness", "BrowserGym/AgentLab", "external browser/computer-use harness family")
	benchmark := flag.String("benchmark", "webarena", "external BrowserGym/AgentLab benchmark id")
	model := flag.String("model", "shared-agent-model", "model id shared by raw and fak arms")
	agent := flag.String("agent", "browsergym_agent_config", "AgentLab agent-args module or agent id used by the raw arm")
	fakAgent := flag.String("fak-agent", "", "AgentLab agent-args module or agent id used by the fak arm (default: --agent-through-fak)")
	maxSteps := flag.Int("max-steps", 30, "external harness max steps shared by raw and fak arms")
	rawCommand := flag.String("raw-command", "", "explicit raw browser harness command")
	fakCommand := flag.String("fak-command", "", "explicit fak browser harness command")
	fakGateway := flag.String("fak-gateway", "http://localhost:8080/v1", "OpenAI-compatible fak gateway base URL for the fak arm")
	rawOutput := flag.String("raw-output", "experiments/agent-live/browseraction-official-raw-20260626", "raw browser harness output archive directory")
	fakOutput := flag.String("fak-output", "experiments/agent-live/browseraction-official-fak-20260626", "fak browser harness output archive directory")
	localFixture := flag.String("local-fixture", "experiments/agent-live/browser-action-mediation-smoke-20260625.json", "local fixture artifact this contract supersedes for promotion")
	flag.Parse()

	suite, err := browseraction.LoadActionMediationSuite(*suitePath)
	if err != nil {
		fatal(err)
	}
	if *contractMode {
		fakArmAgent := strings.TrimSpace(*fakAgent)
		if fakArmAgent == "" {
			fakArmAgent = strings.TrimSpace(*agent) + "-through-fak"
		}
		raw := strings.TrimSpace(*rawCommand)
		if raw == "" {
			raw = buildOfficialRunCommand(*benchmark, *agent, *model, *rawOutput, "", *maxSteps)
		}
		fak := strings.TrimSpace(*fakCommand)
		if fak == "" {
			fak = buildOfficialRunCommand(*benchmark, fakArmAgent, *model, *fakOutput, *fakGateway, *maxSteps)
		}
		contract := browseraction.BuildOfficialRunContract(browseraction.OfficialRunContractInput{
			GeneratedAt:          time.Now().UTC().Format(time.RFC3339),
			Suite:                suite,
			SuitePath:            *suitePath,
			LocalFixtureArtifact: *localFixture,
			Harness:              *harness,
			Benchmark:            *benchmark,
			Model:                *model,
			Agent:                *agent,
			FakAgent:             fakArmAgent,
			MaxSteps:             *maxSteps,
			RawCommand:           raw,
			FakCommand:           fak,
			RawOutputDir:         *rawOutput,
			FakOutputDir:         *fakOutput,
			FakGateway:           *fakGateway,
		})
		b, err := json.MarshalIndent(contract, "", "  ")
		if err != nil {
			fatal(err)
		}
		b = append(b, '\n')
		if *outPath == "" {
			if _, err := os.Stdout.Write(b); err != nil {
				fatal(err)
			}
		} else if err := benchcli.WriteFile(*outPath, b); err != nil {
			fatal(err)
		}
		if *mdPath != "" {
			if err := benchcli.WriteFile(*mdPath, []byte(browseraction.RenderOfficialRunContractMarkdown(contract))); err != nil {
				fatal(err)
			}
		}
		fmt.Fprintf(os.Stderr, "\n== browseractionbench contract ==\n")
		fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
		fmt.Fprintf(os.Stderr, "status       : %s\n", contract.Status)
		fmt.Fprintf(os.Stderr, "harness      : %s\n", contract.TaskSelection.OfficialHarness)
		fmt.Fprintf(os.Stderr, "benchmark    : %s\n", contract.TaskSelection.OfficialBenchmark)
		fmt.Fprintf(os.Stderr, "tasks        : %d candidate %s\n", len(contract.TaskSelection.CandidateTaskIDs), candidateIDLabel(len(contract.TaskSelection.CandidateTaskIDs)))
		fmt.Fprintf(os.Stderr, "claim        : %t\n", contract.ResultClaimAllowed)
		if *outPath != "" {
			fmt.Fprintf(os.Stderr, "json         : %s\n", *outPath)
		}
		if *mdPath != "" {
			fmt.Fprintf(os.Stderr, "markdown     : %s\n", *mdPath)
		}
		return
	}
	report, err := browseraction.RunActionMediation(context.Background(), suite, time.Now().UTC())
	if err != nil {
		fatal(err)
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fatal(err)
	}
	b = append(b, '\n')
	if *outPath == "" {
		if _, err := os.Stdout.Write(b); err != nil {
			fatal(err)
		}
	} else if err := benchcli.WriteFile(*outPath, b); err != nil {
		fatal(err)
	}
	if *mdPath != "" {
		if err := benchcli.WriteFile(*mdPath, []byte(browseraction.RenderActionMediationMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== browseractionbench ==\n")
	fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
	fmt.Fprintf(os.Stderr, "tasks        : %d\n", report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "raw safe     : %d/%d\n", report.Summary.Raw.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak safe     : %d/%d\n", report.Summary.Fak.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak denied   : %d\n", report.Summary.Fak.DeniedActions)
	if *outPath != "" {
		fmt.Fprintf(os.Stderr, "json         : %s\n", *outPath)
	}
	if *mdPath != "" {
		fmt.Fprintf(os.Stderr, "markdown     : %s\n", *mdPath)
	}
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "browseractionbench: %v\n", err)
	os.Exit(1)
}

func buildOfficialRunCommand(benchmark, agent, model, outputDir, fakGateway string, maxSteps int) string {
	if maxSteps <= 0 {
		maxSteps = 30
	}
	parts := []string{
		"$env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'",
		"$env:AGENTLAB_EXP_ROOT=" + quotePowerShellValue(outputDir),
		"$env:FAK_BROWSERGYM_AGENT=" + quotePowerShellValue(agent),
		"$env:FAK_BROWSERGYM_MODEL=" + quotePowerShellValue(model),
		"$env:FAK_BROWSERGYM_MAX_STEPS=" + quotePowerShellValue(fmt.Sprintf("%d", maxSteps)),
	}
	if fakGateway != "" {
		parts = append(parts, "$env:OPENAI_BASE_URL="+quotePowerShellValue(fakGateway))
		parts = append(parts, "$env:OPENAI_API_BASE="+quotePowerShellValue(fakGateway))
	}
	parts = append(parts,
		fmt.Sprintf("python -c %s", quotePowerShellValue(agentLabStudyScript(benchmark))))
	return strings.Join(parts, "; ")
}

func agentLabStudyScript(benchmark string) string {
	benchmark = strings.TrimSpace(benchmark)
	if benchmark == "" {
		benchmark = "webarena"
	}
	return "from agentlab.experiments.study import make_study; " +
		"from importlib import import_module; " +
		"import os; " +
		"agent_args=getattr(import_module(os.environ['FAK_BROWSERGYM_AGENT']), 'AGENT_ARGS'); " +
		"study=make_study(benchmark='" + strings.ReplaceAll(benchmark, "'", "\\'") + "', agent_args=[agent_args], comment=os.environ['FAK_BROWSERGYM_MODEL']); " +
		"study.run(n_jobs=1)"
}

func quotePowerShellValue(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "$env:") {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func candidateIDLabel(n int) string {
	if n == 1 {
		return "id"
	}
	return "ids"
}
