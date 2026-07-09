package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
	if len(argv) > 0 && argv[0] == "export" {
		return runExport(argv[1:])
	}
	if len(argv) > 0 && argv[0] == "fetch" {
		return runFetch(argv[1:])
	}
	if len(argv) > 0 && argv[0] == "contract" {
		return runContract(argv[1:])
	}

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

// runExport implements `livecodebench export --format custom-evaluator`: it
// writes the exact input lcb_runner.runner.custom_evaluator consumes, so a
// fak generation run can be graded by the OFFICIAL LiveCodeBench checker.
// Grade the emitted file with:
//
//	python -m lcb_runner.runner.custom_evaluator \
//	    --custom_output_file <out> \
//	    --release_version <release_vN>
func runExport(argv []string) int {
	fs := flag.NewFlagSet("livecodebench export", flag.ContinueOnError)
	fixture := fs.String("fixture", "internal/livecodebench/testdata/fixture.json", "path to the LiveCodeBench fixture holding the generations to export")
	format := fs.String("format", "custom-evaluator", "export format; only \"custom-evaluator\" is supported")
	out := fs.String("out", "", "file to write the export to (default: stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench export: unexpected positional arguments")
		return 2
	}
	if *format != "custom-evaluator" {
		fmt.Fprintf(os.Stderr, "livecodebench export: unsupported --format %q (only \"custom-evaluator\")\n", *format)
		return 2
	}
	f, err := livecodebench.LoadFile(*fixture)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench export: %v\n", err)
		return 1
	}
	w := io.Writer(os.Stdout)
	if *out != "" {
		file, err := os.Create(*out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench export: %v\n", err)
			return 1
		}
		defer file.Close()
		w = file
	}
	if err := livecodebench.WriteCustomEvaluatorInput(w, f); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench export: %v\n", err)
		return 1
	}
	return 0
}

// runFetch implements `livecodebench fetch`: it turns upstream LiveCodeBench
// rows into a normalized, release-pinned Suite JSON with a provenance header.
// Two sources, exactly one required: --from FILE replays a committed/offline
// rows file deterministically (the CI path, no network); --fetch GETs the
// HuggingFace datasets-server rows for --release-version (the only networked
// path, opt-in). The normalize + provenance logic is pure in
// internal/livecodebench; this is the only place that reads a file or the wire.
func runFetch(argv []string) int {
	fs := flag.NewFlagSet("livecodebench fetch", flag.ContinueOnError)
	release := fs.String("release-version", "", "LCB dataset release to pin (release_v1..release_v6 or release_latest)")
	from := fs.String("from", "", "read upstream rows from a local JSON file (offline replay) instead of the network")
	doFetch := fs.Bool("fetch", false, "GET the HuggingFace datasets-server rows for --release-version (network; opt-in)")
	scenario := fs.String("scenario", "codegeneration", "LCB scenario to tag normalized problems with")
	dataset := fs.String("dataset", "livecodebench/code_generation_lite", "HuggingFace dataset id to record and (with --fetch) read")
	split := fs.String("split", "test", "dataset split to read")
	revision := fs.String("revision", "", "dataset revision to record in provenance (default: the resolved release)")
	limit := fs.Int("limit", 100, "with --fetch, number of rows to request (datasets-server caps per request)")
	offset := fs.Int("offset", 0, "with --fetch, row offset to request")
	model := fs.String("model", "", "optional model identity to record on the suite")
	fetchedAt := fs.String("fetched-at", "", "RFC3339 fetch timestamp to record (default: now for --fetch, empty for --from)")
	out := fs.String("out", "", "file to write the normalized suite to (default: stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench fetch: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*release) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench fetch: --release-version is required (pin the release, never implicit)")
		return 2
	}
	hasFrom := strings.TrimSpace(*from) != ""
	if hasFrom == *doFetch {
		fmt.Fprintln(os.Stderr, "livecodebench fetch: choose exactly one source: --from FILE (offline) or --fetch (network)")
		return 2
	}

	stamp := strings.TrimSpace(*fetchedAt)
	var raw []byte
	var err error
	if hasFrom {
		raw, err = os.ReadFile(*from)
	} else {
		if stamp == "" {
			stamp = time.Now().UTC().Format(time.RFC3339)
		}
		raw, err = fetchHFRows(*dataset, *release, *split, *offset, *limit)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fetch: %v\n", err)
		return 1
	}

	ups, err := livecodebench.ParseUpstreamRows(raw)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fetch: %v\n", err)
		return 1
	}
	suite, err := livecodebench.Normalize(ups, livecodebench.NormalizeOptions{
		Release:   *release,
		Scenario:  livecodebench.Scenario(*scenario),
		DatasetID: *dataset,
		Revision:  *revision,
		Split:     *split,
		FetchedAt: stamp,
		Model:     *model,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fetch: %v\n", err)
		return 1
	}

	w := io.Writer(os.Stdout)
	if *out != "" {
		file, cerr := os.Create(*out)
		if cerr != nil {
			fmt.Fprintf(os.Stderr, "livecodebench fetch: %v\n", cerr)
			return 1
		}
		defer file.Close()
		w = file
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(suite); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench fetch: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "livecodebench fetch: %d problem(s), release %s, dataset %s\n",
		len(suite.Problems), suite.ReleaseVersion, suite.Provenance.DatasetID)
	return 0
}

// runContract implements `livecodebench contract`: it emits the machine-readable
// official-run contract (#2110) that pins the raw lcb_runner and fak-native
// generation commands, the constants both arms share, and the official grading
// handoff. It performs NO run and claims no score — the emitted contract always
// carries result_claim_allowed=false. An optional --suite pins the exact
// question_ids so both arms provably score the same problems.
func runContract(argv []string) int {
	fs := flag.NewFlagSet("livecodebench contract", flag.ContinueOnError)
	release := fs.String("release-version", "release_v6", "LCB dataset release to pin (release_v1..release_v6; release_latest is refused for a published result)")
	scenario := fs.String("scenario", "codegeneration", "LCB scenario both arms run (codegeneration|selfrepair|testoutputprediction|codeexecution)")
	startDate := fs.String("start-date", "", "contest-date window start (YYYY-MM-DD); the contamination boundary")
	endDate := fs.String("end-date", "", "contest-date window end (YYYY-MM-DD)")
	model := fs.String("model", "", "model identity shared by raw and fak arms")
	engine := fs.String("engine", "", "serving engine id: inkernel for fak's own decode (the pure-kernel arm, no external engine in the path), otherwise sglang|vllm|the hosted proxy")
	servingBackend := fs.String("serving-backend", "", "serving engine + quantization the run is served through (e.g. SGLang W4AFP8)")
	gateway := fs.String("gateway", "http://127.0.0.1:8080/v1", "local fak gateway base URL both arms target")
	runDir := fs.String("run-dir", "experiments/livecodebench/<run-id>", "LCB_OUT run directory referenced by the arm and grading commands")
	suitePath := fs.String("suite", "", "optional normalized suite JSON to pin the exact candidate question_ids")
	issueRef := fs.String("issue", "#3060", "campaign issue reference recorded in the contract")
	out := fs.String("out", "", "write the contract JSON to this path (default: stdout)")
	mdPath := fs.String("md", "", "also write the contract markdown to this path")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench contract: unexpected positional arguments")
		return 2
	}

	in := livecodebench.OfficialRunContractInput{
		GeneratedAt:     time.Now().UTC().Format(time.RFC3339),
		Issue:           *issueRef,
		SuitePath:       strings.TrimSpace(*suitePath),
		ReleaseSelector: *release,
		Scenario:        livecodebench.Scenario(*scenario),
		StartDate:       *startDate,
		EndDate:         *endDate,
		Model:           *model,
		Engine:          *engine,
		ServingBackend:  *servingBackend,
		Gateway:         *gateway,
		RunDir:          *runDir,
	}
	if strings.TrimSpace(*suitePath) != "" {
		suite, err := livecodebench.LoadSuiteFile(*suitePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench contract: %v\n", err)
			return 1
		}
		in.Suite = &suite
	}

	contract := livecodebench.BuildOfficialRunContract(in)

	if strings.TrimSpace(*mdPath) != "" {
		if err := os.WriteFile(*mdPath, []byte(livecodebench.RenderOfficialRunContractMarkdown(contract)), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench contract: %v\n", err)
			return 1
		}
	}

	w := io.Writer(os.Stdout)
	if strings.TrimSpace(*out) != "" {
		file, err := os.Create(*out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench contract: %v\n", err)
			return 1
		}
		defer file.Close()
		w = file
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(contract); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench contract: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "livecodebench contract: status %s, release %s, scenario %s, result_claim_allowed=%t\n",
		contract.Status, contract.Constants.ReleaseVersion, contract.Constants.Scenario, contract.ResultClaimAllowed)
	return 0
}

// fetchHFRows GETs the HuggingFace datasets-server /rows endpoint for a dataset
// config (the LCB release) and split. It returns the raw JSON body for
// ParseUpstreamRows; the datasets-server rows shape is the envelope that parser
// already understands. This is the sole networked call in the fetch path.
func fetchHFRows(dataset, config, split string, offset, length int) ([]byte, error) {
	u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%d&length=%d",
		url.QueryEscape(dataset), url.QueryEscape(config), url.QueryEscape(split), offset, length)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 90 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("datasets-server HTTP %d for dataset=%s config=%s", resp.StatusCode, dataset, config)
	}
	return io.ReadAll(resp.Body)
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
