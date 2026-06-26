// Command terminalbench runs a Terminal-Bench-shaped command-boundary smoke
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

	"github.com/anthony-chaudhary/fak/internal/terminalbench"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/terminalbench/command_boundary_smoke.json"), "Terminal-Bench-shaped command suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	contractMode := flag.Bool("contract", false, "emit an external official-run contract instead of replaying the local smoke")
	datasetName := flag.String("dataset-name", "terminal-bench-core", "official Terminal-Bench dataset name")
	datasetVersion := flag.String("dataset-version", "0.1.1", "official Terminal-Bench dataset version")
	model := flag.String("model", "shared-agent-model", "model id shared by raw and fak arms")
	agent := flag.String("agent", "terminus", "Terminal-Bench agent used by the raw arm")
	fakAgent := flag.String("fak-agent", "", "Terminal-Bench agent used by the fak arm (default: --agent-through-fak)")
	nConcurrent := flag.Int("n-concurrent", 1, "Terminal-Bench concurrency shared by raw and fak arms")
	rawCommand := flag.String("raw-command", "", "explicit raw tb run command")
	fakCommand := flag.String("fak-command", "", "explicit fak tb run command")
	fakGateway := flag.String("fak-gateway", "http://localhost:8080/v1", "OpenAI-compatible fak gateway base URL for the fak arm")
	rawOutput := flag.String("raw-output", "experiments/agent-live/terminalbench-official-raw-20260626", "raw Terminal-Bench output archive directory")
	fakOutput := flag.String("fak-output", "experiments/agent-live/terminalbench-official-fak-20260626", "fak Terminal-Bench output archive directory")
	localFixture := flag.String("local-fixture", "experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json", "local fixture artifact this contract supersedes for promotion")
	flag.Parse()

	suite, err := terminalbench.Load(*suitePath)
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
			raw = buildOfficialRunCommand(*datasetName, *datasetVersion, *agent, *model, "", *nConcurrent)
		}
		fak := strings.TrimSpace(*fakCommand)
		if fak == "" {
			fak = buildOfficialRunCommand(*datasetName, *datasetVersion, fakArmAgent, *model, *fakGateway, *nConcurrent)
		}
		contract := terminalbench.BuildOfficialRunContract(terminalbench.OfficialRunContractInput{
			GeneratedAt:          time.Now().UTC().Format(time.RFC3339),
			Suite:                suite,
			SuitePath:            *suitePath,
			LocalFixtureArtifact: *localFixture,
			DatasetName:          *datasetName,
			DatasetVersion:       *datasetVersion,
			Model:                *model,
			Agent:                *agent,
			FakAgent:             fakArmAgent,
			NConcurrent:          *nConcurrent,
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
		} else if err := writeFile(*outPath, b); err != nil {
			fatal(err)
		}
		if *mdPath != "" {
			if err := writeFile(*mdPath, []byte(terminalbench.RenderOfficialRunContractMarkdown(contract))); err != nil {
				fatal(err)
			}
		}
		fmt.Fprintf(os.Stderr, "\n== terminalbench contract ==\n")
		fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
		fmt.Fprintf(os.Stderr, "status       : %s\n", contract.Status)
		fmt.Fprintf(os.Stderr, "dataset      : %s==%s\n", contract.TaskSelection.OfficialDataset, contract.TaskSelection.OfficialDatasetVersion)
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
	report, err := terminalbench.Run(context.Background(), suite, time.Now().UTC())
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
	} else if err := writeFile(*outPath, b); err != nil {
		fatal(err)
	}
	if *mdPath != "" {
		if err := writeFile(*mdPath, []byte(terminalbench.RenderMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== terminalbench ==\n")
	fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
	fmt.Fprintf(os.Stderr, "tasks        : %d\n", report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "raw safe     : %d/%d\n", report.Summary.Raw.SafeResolves, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak safe     : %d/%d\n", report.Summary.Fak.SafeResolves, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak denied   : %d\n", report.Summary.Fak.DeniedCommands)
	if *outPath != "" {
		fmt.Fprintf(os.Stderr, "json         : %s\n", *outPath)
	}
	if *mdPath != "" {
		fmt.Fprintf(os.Stderr, "markdown     : %s\n", *mdPath)
	}
}

func writeFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "terminalbench: %v\n", err)
	os.Exit(1)
}

func buildOfficialRunCommand(datasetName, datasetVersion, agent, model, fakGateway string, nConcurrent int) string {
	if nConcurrent <= 0 {
		nConcurrent = 1
	}
	dataset := terminalBenchDataset(datasetName, datasetVersion)
	parts := []string{"$env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'"}
	if fakGateway != "" {
		parts = append(parts, "$env:OPENAI_BASE_URL="+quotePowerShellValue(fakGateway))
		parts = append(parts, "$env:OPENAI_API_BASE="+quotePowerShellValue(fakGateway))
	}
	parts = append(parts,
		fmt.Sprintf("foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset %s --agent %s --model %s --task-id $task --n-concurrent %d }",
			quoteCLIArg(dataset), quoteCLIArg(agent), quoteCLIArg(model), nConcurrent))
	return strings.Join(parts, "; ")
}

func terminalBenchDataset(name, version string) string {
	name = strings.TrimSpace(name)
	version = strings.TrimSpace(version)
	if version == "" {
		return name
	}
	return name + "==" + version
}

func quoteCLIArg(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "''"
	}
	if strings.ContainsAny(s, " \t`\"'|()") {
		return "'" + strings.ReplaceAll(s, "'", "''") + "'"
	}
	return s
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
