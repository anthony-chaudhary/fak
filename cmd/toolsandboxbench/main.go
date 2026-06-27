// Command toolsandboxbench runs a tau3/ToolSandbox-shaped policy-state adapter
// smoke through fak adjudication.
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
	"github.com/anthony-chaudhary/fak/internal/toolsandbox"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/toolsandbox/policy_state_smoke.json"), "ToolSandbox/tau3-shaped suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	contractMode := flag.Bool("contract", false, "emit an external official-run contract instead of replaying the local smoke")
	officialHarness := flag.String("official-harness", "tau3", "external harness family: tau3 or toolsandbox")
	domain := flag.String("domain", "retail", "official tau3 domain or ToolSandbox scenario family")
	model := flag.String("model", "shared-agent-model", "agent model id shared by raw and fak arms")
	userModel := flag.String("user-model", "", "user simulator model id shared by raw and fak arms (default: --model)")
	trials := flag.Int("trials", 1, "official harness trials per task")
	rawCommand := flag.String("raw-command", "", "explicit raw official-harness command")
	fakCommand := flag.String("fak-command", "", "explicit fak official-harness command")
	fakGateway := flag.String("fak-gateway", "http://localhost:8080/v1", "OpenAI-compatible fak gateway base URL for the fak arm")
	rawOutput := flag.String("raw-output", "experiments/agent-live/toolsandbox-official-raw-20260626", "raw official-harness output archive directory")
	fakOutput := flag.String("fak-output", "experiments/agent-live/toolsandbox-official-fak-20260626", "fak official-harness output archive directory")
	localFixture := flag.String("local-fixture", "experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json", "local fixture artifact this contract supersedes for promotion")
	flag.Parse()

	suite, err := toolsandbox.Load(*suitePath)
	if err != nil {
		fatal(err)
	}
	if *contractMode {
		user := *userModel
		if user == "" {
			user = *model
		}
		raw := strings.TrimSpace(*rawCommand)
		if raw == "" {
			raw = buildOfficialRunCommand(*officialHarness, *domain, *model, user, "", *trials)
		}
		fak := strings.TrimSpace(*fakCommand)
		if fak == "" {
			fak = buildOfficialRunCommand(*officialHarness, *domain, *model, user, *fakGateway, *trials)
		}
		contract := toolsandbox.BuildOfficialRunContract(toolsandbox.OfficialRunContractInput{
			GeneratedAt:          time.Now().UTC().Format(time.RFC3339),
			Suite:                suite,
			SuitePath:            *suitePath,
			LocalFixtureArtifact: *localFixture,
			OfficialHarness:      *officialHarness,
			Domain:               *domain,
			Model:                *model,
			UserModel:            user,
			Trials:               *trials,
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
			if err := benchcli.WriteFile(*mdPath, []byte(toolsandbox.RenderOfficialRunContractMarkdown(contract))); err != nil {
				fatal(err)
			}
		}
		fmt.Fprintf(os.Stderr, "\n== toolsandboxbench contract ==\n")
		fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
		fmt.Fprintf(os.Stderr, "status       : %s\n", contract.Status)
		fmt.Fprintf(os.Stderr, "harness      : %s\n", contract.TaskSelection.OfficialHarness)
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
	report, err := toolsandbox.Run(context.Background(), suite, time.Now().UTC())
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
		if err := benchcli.WriteFile(*mdPath, []byte(toolsandbox.RenderMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== toolsandboxbench ==\n")
	fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
	fmt.Fprintf(os.Stderr, "tasks        : %d\n", report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "raw safe     : %d/%d\n", report.Summary.Raw.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak safe     : %d/%d\n", report.Summary.Fak.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak denied   : %d\n", report.Summary.Fak.DeniedCalls)
	if *outPath != "" {
		fmt.Fprintf(os.Stderr, "json         : %s\n", *outPath)
	}
	if *mdPath != "" {
		fmt.Fprintf(os.Stderr, "markdown     : %s\n", *mdPath)
	}
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "toolsandboxbench: %v\n", err)
	os.Exit(1)
}

func buildOfficialRunCommand(harness, domain, model, userModel, fakGateway string, trials int) string {
	if trials <= 0 {
		trials = 1
	}
	h := strings.ToLower(strings.TrimSpace(harness))
	var parts []string
	if h == "toolsandbox" || h == "apple-toolsandbox" {
		parts = append(parts, "$env:TOOLSANDBOX_SCENARIO='<official scenario id>'")
		if fakGateway != "" {
			parts = append(parts, "$env:OPENAI_BASE_URL="+quotePowerShellValue(fakGateway))
		}
		parts = append(parts,
			"tool_sandbox --user "+quoteCLIArg(userModel)+
				" --agent "+quoteCLIArg(model)+
				" --scenario $env:TOOLSANDBOX_SCENARIO")
		return strings.Join(parts, "; ")
	}
	parts = append(parts, "$env:TAU3_TASK_IDS='<space-separated official task ids>'")
	if fakGateway != "" {
		parts = append(parts, "$env:OPENAI_BASE_URL="+quotePowerShellValue(fakGateway))
		parts = append(parts, "$env:OPENAI_API_BASE="+quotePowerShellValue(fakGateway))
	}
	parts = append(parts,
		fmt.Sprintf("tau2 run --domain %s --agent-llm %s --user-llm %s --num-trials %d --task-ids ($env:TAU3_TASK_IDS -split ' ')",
			quoteCLIArg(domain), quoteCLIArg(model), quoteCLIArg(userModel), trials))
	return strings.Join(parts, "; ")
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
