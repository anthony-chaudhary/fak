package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(argv []string) int {
	fs := flag.NewFlagSet("livecodebench", flag.ContinueOnError)
	fixture := fs.String("fixture", "internal/livecodebench/testdata/fixture.json", "path to committed LiveCodeBench smoke fixture")
	asJSON := fs.Bool("json", false, "print the smoke report as JSON")
	check := fs.Bool("check", false, "fail if the smoke report is not shape-valid or if result_claim_allowed is true")
	preflightMode := fs.Bool("preflight", false, "probe this host's readiness for a LiveCodeBench run and emit a result-claim-gated preflight artifact")
	datasetURL := fs.String("dataset-url", "https://huggingface.co/datasets/livecodebench/code_generation_lite", "HF LiveCodeBench dataset URL the preflight checks")
	probeDataset := fs.Bool("probe-dataset", false, "in --preflight, GET the HF dataset URL to check it is reachable")
	fakGateway := fs.String("fak-gateway", "http://localhost:18080/v1", "fak gateway base URL the preflight checks")
	probeGateway := fs.Bool("probe-gateway", false, "in --preflight, GET the fak gateway /models endpoint to check it is reachable")
	sandboxCmd := fs.String("sandbox-cmd", "docker", "executable on PATH the preflight treats as the code-execution sandbox")
	issueRef := fs.String("issue", "#2111", "issue reference recorded in the preflight artifact")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *preflightMode {
		return runPreflight(preflightInputFromFlags(*datasetURL, *probeDataset, *fakGateway, *probeGateway, *sandboxCmd, *issueRef), *asJSON)
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench: unexpected positional arguments")
		return 2
	}
	f, err := livecodebench.LoadFile(*fixture)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
		return 1
	}
	report := livecodebench.SmokeReport(f)
	if *check {
		if err := livecodebench.ValidateSmokeReport(report); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Printf("LiveCodeBench fixture smoke: %d question(s), %d scenario(s), result_claim_allowed=%v\n",
		report.Questions, len(report.Scenarios), report.ResultClaimAllowed)
	for _, s := range report.Scenarios {
		fmt.Printf("  - %s: %d\n", s.Scenario, s.Questions)
	}
	return 0
}

// preflightInputFromFlags probes the live host. The classifier
// (livecodebench.BuildPreflight) is pure and unit-tested separately; this is
// the only place that touches PATH lookups or the network.
func preflightInputFromFlags(datasetURL string, probeDataset bool, gatewayBase string, probeGateway bool, sandboxCmd string, issue string) livecodebench.PreflightInput {
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()

	uvOK, uvVer := probeExecVersion(ctx, "uv", "--version")
	pyOK, pyVer := probePython311(ctx)
	sandboxOK, sandboxDetail := probeExecVersion(ctx, sandboxCmd, "--version")

	var datasetChecked, datasetReachable bool
	if probeDataset && strings.TrimSpace(datasetURL) != "" {
		datasetChecked = true
		datasetReachable, _ = probeURLReachable(ctx, datasetURL)
	}
	var gatewayChecked, gatewayReachable bool
	if probeGateway && strings.TrimSpace(gatewayBase) != "" {
		gatewayChecked = true
		gatewayReachable, _ = probeURLReachable(ctx, strings.TrimRight(gatewayBase, "/")+"/models")
	}

	return livecodebench.PreflightInput{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Issue:       issue,
		Probe: livecodebench.PreflightProbe{
			UvPresent:        uvOK,
			UvVersion:        uvVer,
			PythonPresent:    pyOK,
			PythonVersion:    pyVer,
			DatasetChecked:   datasetChecked,
			DatasetReachable: datasetReachable,
			DatasetURL:       datasetURL,
			GatewayChecked:   gatewayChecked,
			GatewayReachable: gatewayReachable,
			GatewayURL:       gatewayBase,
			SandboxAvailable: sandboxOK,
			SandboxDetail:    sandboxDetail,
		},
	}
}

func runPreflight(in livecodebench.PreflightInput, asJSON bool) int {
	report := livecodebench.BuildPreflight(in)
	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(os.Stderr, "\n== livecodebench preflight ==\n")
	fmt.Fprintf(os.Stderr, "status       : %s\n", report.Status)
	fmt.Fprintf(os.Stderr, "claim        : %t\n", report.ResultClaimAllowed)
	for _, g := range report.Gates {
		fmt.Fprintf(os.Stderr, "  %-24s ok=%-5t %s\n", g.Name, g.OK, g.Detail)
	}
	if len(report.BlockingReasons) > 0 {
		fmt.Fprintf(os.Stderr, "blocked by   : %s\n", strings.Join(report.BlockingReasons, ", "))
	}
	fmt.Fprintf(os.Stderr, "next action  : %s\n", report.NextAction)
	return 0
}

func probeExecVersion(ctx context.Context, name string, args ...string) (bool, string) {
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	if err != nil {
		return false, ""
	}
	return true, firstLine(string(out))
}

// probePython311 tries python3.11 first (the exact interpreter LiveCodeBench
// pins), falling back to a bare "python"/"python3" whose reported version is
// then checked by the caller against the 3.11.x gate.
func probePython311(ctx context.Context) (bool, string) {
	for _, name := range []string{"python3.11", "python3", "python"} {
		if ok, ver := probeExecVersion(ctx, name, "--version"); ok {
			return true, strings.TrimPrefix(ver, "Python ")
		}
	}
	return false, ""
}

func probeURLReachable(ctx context.Context, url string) (bool, string) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, err.Error()
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return false, err.Error()
	}
	defer resp.Body.Close()
	// Any HTTP round-trip proves the endpoint is listening, even a 4xx.
	return true, fmt.Sprintf("HTTP %d", resp.StatusCode)
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
