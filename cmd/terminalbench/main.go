// Command terminalbench runs a Terminal-Bench-shaped command-boundary smoke
// through fak adjudication.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/terminalbench"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/terminalbench/command_boundary_smoke.json"), "Terminal-Bench-shaped command suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	contractMode := flag.Bool("contract", false, "emit an external official-run contract instead of replaying the local smoke")
	preflightMode := flag.Bool("preflight", false, "probe this host's readiness for the Terminal-Bench 2.1 raw/fak rehearsal and emit a result-claim-gated preflight artifact")
	preflightDataset := flag.String("preflight-dataset", "terminal-bench/terminal-bench-2-1", "official Harbor dataset slug the rehearsal targets")
	probeGateway := flag.Bool("probe-gateway", false, "in --preflight, GET the fak gateway /models endpoint to check the fak arm is reachable")
	oracleArtifact := flag.String("oracle-artifact", "", "in --preflight, require this official oracle-smoke artifact to exist before greenlighting any paid raw/fak run (acceptance criterion: oracle before paid)")
	officialContract := flag.String("official-contract", "experiments/agent-live/terminalbench-official-run-contract-20260626.json", "path to the official-run contract this preflight gates")
	submissionPacket := flag.String("submission-packet", "docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md", "path to the submission-packet index this preflight feeds")
	issueRef := flag.String("issue", "#900", "campaign issue reference recorded in the preflight artifact")
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
		} else if err := benchcli.WriteFile(*outPath, b); err != nil {
			fatal(err)
		}
		if *mdPath != "" {
			if err := benchcli.WriteFile(*mdPath, []byte(terminalbench.RenderOfficialRunContractMarkdown(contract))); err != nil {
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
	if *preflightMode {
		ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
		defer cancel()
		harborOK, harborVer := probeHarbor(ctx)
		dockerOK, dockerDetail := probeDocker(ctx)
		gatewayBase := strings.TrimSpace(*fakGateway)
		var gwChecked, gwReachable bool
		if *probeGateway && gatewayBase != "" {
			gwChecked = true
			gwReachable, _ = probeGatewayReachable(ctx, gatewayBase)
		}
		preflight := terminalbench.BuildRehearsalPreflight(terminalbench.RehearsalPreflightInput{
			GeneratedAt: time.Now().UTC().Format(time.RFC3339),
			Probe: terminalbench.PreflightProbe{
				HarborPresent:          harborOK,
				HarborVersion:          harborVer,
				DockerEngineUp:         dockerOK,
				DockerDetail:           dockerDetail,
				OpenAIKeyPresent:       strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != "",
				GatewayChecked:         gwChecked,
				GatewayReachable:       gwReachable,
				GatewayURL:             gatewayBase,
				OracleArtifactRequired: strings.TrimSpace(*oracleArtifact) != "",
				OracleArtifactPresent:  strings.TrimSpace(*oracleArtifact) != "" && fileExists(*oracleArtifact),
				OracleArtifactPath:     strings.TrimSpace(*oracleArtifact),
			},
			Dataset:          *preflightDataset,
			Issue:            *issueRef,
			OfficialContract: *officialContract,
			SubmissionPacket: *submissionPacket,
			CandidateTaskIDs: suiteTaskIDs(suite),
		})
		b, err := json.MarshalIndent(preflight, "", "  ")
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
			if err := benchcli.WriteFile(*mdPath, []byte(terminalbench.RenderRehearsalPreflightMarkdown(preflight))); err != nil {
				fatal(err)
			}
		}
		fmt.Fprintf(os.Stderr, "\n== terminalbench preflight ==\n")
		fmt.Fprintf(os.Stderr, "dataset      : %s\n", preflight.Dataset)
		fmt.Fprintf(os.Stderr, "status       : %s\n", preflight.Status)
		fmt.Fprintf(os.Stderr, "oracle ready : %t\n", preflight.OracleSmokeReady)
		fmt.Fprintf(os.Stderr, "raw ready    : %t\n", preflight.RawPaidSmokeReady)
		fmt.Fprintf(os.Stderr, "fak ready    : %t\n", preflight.FakPaidSmokeReady)
		fmt.Fprintf(os.Stderr, "claim        : %t\n", preflight.ResultClaimAllowed)
		if len(preflight.BlockingReasons) > 0 {
			fmt.Fprintf(os.Stderr, "blocked by   : %s\n", strings.Join(preflight.BlockingReasons, ", "))
		}
		fmt.Fprintf(os.Stderr, "next action  : %s\n", preflight.NextAction)
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
	} else if err := benchcli.WriteFile(*outPath, b); err != nil {
		fatal(err)
	}
	if *mdPath != "" {
		if err := benchcli.WriteFile(*mdPath, []byte(terminalbench.RenderMarkdown(report))); err != nil {
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

func suiteTaskIDs(s terminalbench.Suite) []string {
	ids := make([]string, 0, len(s.Tasks))
	for _, task := range s.Tasks {
		if id := strings.TrimSpace(task.ID); id != "" {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

func probeHarbor(ctx context.Context) (bool, string) {
	out, err := exec.CommandContext(ctx, "harbor", "--version").CombinedOutput()
	if err != nil {
		return false, ""
	}
	return true, firstLine(string(out))
}

func probeDocker(ctx context.Context) (bool, string) {
	out, err := exec.CommandContext(ctx, "docker", "info", "--format", "{{.ServerVersion}}").CombinedOutput()
	if err != nil {
		return false, "docker engine not reachable"
	}
	v := firstLine(string(out))
	if v == "" {
		return false, "docker info returned no server version"
	}
	return true, "docker engine " + v
}

func probeGatewayReachable(ctx context.Context, base string) (bool, string) {
	url := strings.TrimRight(strings.TrimSpace(base), "/") + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, err.Error()
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return false, err.Error()
	}
	defer resp.Body.Close()
	// Any HTTP round-trip proves the gateway process is listening, even a 401.
	return true, fmt.Sprintf("HTTP %d", resp.StatusCode)
}

func fileExists(path string) bool {
	info, err := os.Stat(strings.TrimSpace(path))
	return err == nil && !info.IsDir()
}

func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return ""
}

func candidateIDLabel(n int) string {
	if n == 1 {
		return "id"
	}
	return "ids"
}
